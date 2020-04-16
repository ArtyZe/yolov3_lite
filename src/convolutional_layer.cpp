#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include "eight_bit_int_gemm.h"
#undef max
#undef min
#include <iostream>
#include <string>
#include <memory>
using namespace std;

#ifdef AI2
#include "xnor_layer.h"
#endif

float prune_ratio[31] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, 
                                             int batch_normalize, int binary, int xnor, int adam, int quantized, int weight_quant_flag, int activ_quant_flag)
{
    int i;
    convolutional_layer l;
    
    
    l.type = CONVOLUTIONAL;
    init_layer(l);
    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.count = adam;
    adam = 0;
    l.batch_normalize = batch_normalize;
    l.layer_quantized = quantized;

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    l.weights = (float*)calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = (float*)calloc(c/groups*n*size*size, sizeof(float));	

    float scale = sqrt(2./(size*size*c/l.groups));
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;


#ifdef MASK
	l.weights_result = (float*)calloc(c*n*size*size, sizeof(float));
	l.weights_mask = (float*)calloc(c*n, sizeof(float));
	l.weights_mask_binary = (float*)calloc(c*n, sizeof(float));
	l.weight_mask_updates = (float*)calloc(c*n, sizeof(float));
    int j = 0;
	for(j = 0; j < c*n; j++){
		l.weights_mask[j] = 1;
		l.weights_mask_binary[j]= 1;
	}			
#endif

#ifdef QUANTIZATION_GOOGLE
    l.min_activ_value = (float*)calloc(1, sizeof(float));
    l.max_activ_value = (float*)calloc(1, sizeof(float));

    l.weight_quant_flag = weight_quant_flag;
    l.activ_quant_flag = activ_quant_flag;

	l.activ_data_int8_scales = (float*)calloc(1, sizeof(float));
    l.weight_data_int8_scales = (float*)calloc(1, sizeof(float));
	l.biases_data_int8_scales = (float*)calloc(n, sizeof(float));
	l.output_data_int8_scales = (float*)calloc(n, sizeof(float));

    l.input_sum_int = (uint32_t*)calloc(l.out_w*l.out_h, sizeof(uint32_t));
    l.weights_sum_int = (uint32_t*)calloc(l.n, sizeof(uint32_t));

    l.activ_data_int8_zero_point = (uint8_t*)calloc(1, sizeof(uint8_t));
    l.weight_data_int8_zero_point = (uint8_t*)calloc(1, sizeof(uint8_t));
    l.biases_data_int8_zero_point = (uint8_t*)calloc(n, sizeof(uint8_t));
    l.output_data_int8_zero_point = (uint8_t*)calloc(n, sizeof(uint8_t));

    l.weights_quant = (uint8_t*)calloc(l.nweights, sizeof(uint8_t));
	l.biases_quant = (uint32_t*)calloc(l.n, sizeof(uint32_t));
    l.input_quant = (uint8_t*)calloc(l.c*l.w*l.h, sizeof(uint8_t));
    l.input_backup = (float*)calloc(l.c*l.w*l.h, sizeof(float));

    l.weights_bn_backup = (float*)calloc(l.c*l.n*l.size*l.size, sizeof(float));
    l.output_bn_backup = (float*)calloc(l.n*l.out_w*l.out_h, sizeof(float));
    l.biases_bn_backup = (float*)calloc(l.n, sizeof(float));
    if(l.weight_quant_flag){
        // l.forward = forward_convolutional_layer_quant_google;
        l.forward = forward_convolutional_layer_quant_google_gemm_lowp;
        // l.forward = forward_convolutional_layer_nobn;
    }else{
        l.forward = forward_convolutional_layer;
    }
#elif QUANTIZATION
    if(l.layer_quantized > 0){
        l.weights_int8 = (int8_t*)calloc(c/groups*n*size*size, sizeof(int8_t));
        l.biases_int16 = (int16_t*)calloc(n, sizeof(int16_t));
        l.output_int8 = (int8_t*)calloc(l.batch*l.outputs, sizeof(int8_t));
        l.output_int16 = (int16_t*)calloc(l.batch*l.outputs, sizeof(int16_t));
        l.forward = forward_convolutional_layer_quant;
    }else{
        l.forward = forward_convolutional_layer;
    }
#else
    l.forward = forward_convolutional_layer;
#endif
    l.biases = (float*)calloc(n, sizeof(float));
    l.bias_updates = (float*)calloc(n, sizeof(float));

    l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.output_quant =(uint32_t*) calloc(l.batch*l.outputs, sizeof(uint32_t));
    l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.cweights = (char*)calloc(l.nweights, sizeof(char));
        l.scales = (float*)calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.binary_input = (float*)calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = (float*)calloc(n, sizeof(float));
        l.scale_updates = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(n, sizeof(float));
        l.variance = (float*)calloc(n, sizeof(float));

        l.mean_delta = (float*)calloc(n, sizeof(float));
        l.variance_delta = (float*)calloc(n, sizeof(float));

        l.rolling_mean = (float*)calloc(n, sizeof(float));
        l.rolling_variance = (float*)calloc(n, sizeof(float));
        l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = (float*)calloc(l.nweights, sizeof(float));
        l.v = (float*)calloc(l.nweights, sizeof(float));
        l.bias_m = (float*)calloc(n, sizeof(float));
        l.scale_m = (float*)calloc(n, sizeof(float));
        l.bias_v = (float*)calloc(n, sizeof(float));
        l.scale_v = (float*)calloc(n, sizeof(float));
    }

#ifdef GPU
#ifndef QUANTIZATION_GOOGLE
    l.forward_gpu = forward_convolutional_layer_gpu;
#else
    l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);
    l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);

    l.activ_data_int8_scales_gpu = cuda_make_array(l.activ_data_int8_scales, 1);
    l.weight_data_int8_scales_gpu = cuda_make_array(l.weight_data_int8_scales, 1);
    l.biases_data_int8_scales_gpu = cuda_make_array(l.biases_data_int8_scales, l.c*l.n);
    l.output_data_int8_scales_gpu = cuda_make_array(l.output_data_int8_scales, l.c*l.n);

    l.activ_data_int8_zero_point_gpu = cuda_make_array_uint8(l.activ_data_int8_zero_point, 1);
    l.weight_data_int8_zero_point_gpu = cuda_make_array_uint8(l.weight_data_int8_zero_point, 1);
    l.biases_data_int8_zero_point_gpu = cuda_make_array_uint8(l.biases_data_int8_zero_point, l.c*l.n);
    l.output_data_int8_zero_point_gpu = cuda_make_array_uint8(l.output_data_int8_zero_point, l.c*l.n);

    l.weights_quant_gpu = cuda_make_array_uint8(l.weights_quant, l.c*l.n*l.size*l.size);

    l.weights_bn_backup_gpu = cuda_make_array(l.weights_bn_backup, l.c*l.n*l.size*l.size);
    l.biases_bn_backup_gpu = cuda_make_array(l.biases_bn_backup, l.n);
    l.output_bn_backup_gpu = cuda_make_array(l.output_bn_backup, l.n*l.out_w*l.out_h);

    l.forward_gpu = forward_convolutional_layer_quant_gpu;
#endif
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
#ifdef MASK
		l.weights_result_gpu = cuda_make_array(l.weights_result, c*n*size*size);
		l.weights_mask_gpu = cuda_make_array(l.weights_mask, c*n);
		l.weights_mask_binary_gpu = cuda_make_array(l.weights_mask_binary, c*n);
		l.weight_mask_updates_gpu = cuda_make_array(l.weight_mask_updates, c*n);
#endif 

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
	l->output_bn_backup = (float*)realloc(l->output_bn_backup, l->batch*l->outputs*sizeof(float));
    l->delta  = (float*)realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_bn_backup_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	l->output_bn_backup_gpu = cuda_make_array(l->output_bn_backup, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

#ifdef QUANTIZATION
void forward_convolutional_layer_quant(convolutional_layer l, network net)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, j;
    int const out_size = out_h*out_w;

    net.input_int8 = (int8_t *)calloc(l.inputs, sizeof(int8_t));
    int z;
    for (z = 0; z < l.inputs; ++z) {
        //int16_t src = lround(state.input[k] * net.layers[0].input_quant_multipler);
        int16_t src = net.input[z] * l.input_quant_multipler;
        //int16_t src = net.input[z] * l.activ_quant_scale_google;
        net.input_int8[z] = max_abs(src, I_MAX_VAL);
    }

    ////////////////////////////////////
    // cudnnConvolutionBiasActivationForward()
    // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
    // int8 = activation( float * conv(int8) + float * int8 + float )
    // int8 = activation( conv(input_int8) + bias_float ) // X_INT8x4 or X_INT8
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
    ///////////////////////////////////
    // 1. Convolution !!!
    int fil,batch_index,groups_index;

    // y = conv(x)
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = out_h*out_w;
    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            int8_t *a = l.weights_int8 + groups_index*l.nweights/l.groups;
            int8_t *b = (int8_t *)net.workspace;
            int16_t *c = l.output_int16 + (batch_index*l.groups + groups_index)*n*m;    // int16_t
            int8_t *im =  net.input_int8 + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_int8(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);    // here
            }
	        int t;	  // multi-thread gemm
	        #pragma omp parallel for num_threads(8)
	        for (t = 0; t < m; ++t) {

#ifdef MASK
#if 0			
                //printf("c is: %d, n is: %d, batch is: %d, size is: %d\n", l.c, l.n, l.batch, l.size); 
                int m_1 = 0, n_1 = 0, zero_num_mask = 0, zero_weights = 0;
                for(m_1 = 0; m_1 < l.c*l.n; m_1++){
                    float sum = 0;
                    if(l.weights_mask[m_1] == 0){
                        zero_num_mask = zero_num_mask + 1;
                    }
                    for(n_1 = 0; n_1 < l.size*l.size; n_1++){
                        sum += l.weights[m_1*l.size*l.size + n_1];
                    }
                    if(sum == 0){
                        zero_weights = zero_weights + 1;
                    }
                }
                printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_num_mask, (float)zero_num_mask/(l.c*l.n));
                //printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_weights, (float)(zero_weights)/(l.c*l.n));
#endif	
                gemm_nn_int8_int16_mask(l.size*l.size, l.c/l.groups, 1, n, k, 1, a+t*k, k, b, n, l.weights_mask+t*l.c+groups_index, c+t*n, n);
                //gemm_nn_int8_int16_conv16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
                //gemm_nn_int8_int32(1, n, k, 1, a + t*k, k, b, n, c + t*n, n); // conv_t should be int32_t
#else    
                //printf("quantization 1; mask 0\n");
                gemm_nn_int8_int16(1, n, k, 1, a+t*k, k, b, n, c+t*n, n);
                //gemm_nn_int8_int16_conv16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
                //gemm_nn_int8_int32(1, n, k, 1, a + t*k, k, b, n, c + t*n, n); // conv_t should be int32_t
#endif
            }
        }
	}


    free(net.input_int8);

    float ALPHA1 = R_MULT / (l.input_quant_multipler * l.weights_quant_multipler);

    // y = alpha1 * conv(x)
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = l.output_int16[i] * ALPHA1;    //alpha1
    }

    //for (fil = 0; fil < l.n; ++fil) {
    //    for (j = 0; j < out_size; ++j) {
    //        l.output[fil*out_size + j] = l.output[fil*out_size + j] * ALPHA1;
    //    }
    //}

    //y = alpha1 * conv(x) + bias
    for (fil = 0; fil < l.n; ++fil) {
        for (j = 0; j < out_size; ++j) {
            l.output[fil*out_size + j] += l.biases[fil];
        }
    }
    
    //y = act ( alpha1 * conv(x) + bias )
    // bias is always FLOAT
    if (l.activation == LEAKY) {
        for (i = 0; i < l.n*out_size; ++i) {
            l.output[i] = (l.output[i]>0) ? l.output[i] : l.output[i] / 10; //leaky_activate(l.output[i]);
        }
    }
}
#elif QUANTIZATION_GOOGLE

void forward_convolutional_layer_quant_google_gemm_lowp(convolutional_layer l, network net)
{
    int i, j;
    int input_index;

    net.input_quant = (uint8_t *)calloc(l.inputs, sizeof(uint8_t));

    for (input_index = 0; input_index < l.c*l.w*l.h; ++input_index) {
        uint8_t input_quant_value = clamp(round(net.input[input_index] / l.input_data_int8_scales) + l.input_data_int8_zero_point, 0, 255);
        net.input_quant[input_index] = input_quant_value;
        // printf("the quant input is  %d\n", net.input_quant[input_index]);
    }
    // 1. Convolution !!!
    int batch_index,groups_index;

    // y = conv(x)
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_h*l.out_w;

    net.workspace_quant = (uint8_t*)calloc(1, l.out_h*l.out_w*l.size*l.size*l.c*sizeof(uint8_t));

    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            printf("the batch is %d, and group is %d\n", l.batch, l.groups);
            uint8_t *a = l.weights_quant + groups_index*l.nweights/l.groups;
            uint8_t *b = (uint8_t *)net.workspace_quant;
            uint32_t *c = l.output_quant + (batch_index*l.groups + groups_index)*n*m;  
            uint8_t *im =  net.input_quant + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_int8(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);    // here
            }

            // for (int ss = 0; ss < l.out_w*l.out_h; ++ss){ 
            //     for (int tt = 0; tt < l.c*l.size*l.size; ++tt){
            //         l.input_sum_int[ss] += b[tt*l.out_w*l.out_h+ss];
            //     }
            //     l.input_sum_int[ss] = l.input_sum_int[ss] * (*l.weight_data_int8_zero_point);
            // }

            // double time=what_time_is_it_now();
            // 0.29s for whole net forward
            // gemm_nn_uint8_uint32(m, n, k, 1, a, k, b, n, c, n);

            // 0.53s
            // gemm_nn_uint8_uint32_conv32(m, n, k, 1, a, k, b, n, c, n);

            // const int m = 4;
            // const int n = 2;
            // const int k = 3;
            // // Matrix A (LHS) is:
            // // |  7 | 10 | 13 | 16 |
            // // |  8 | 11 | 14 | 17 |
            // // |  9 | 12 | 15 | 18 |
            // const std::uint8_t a_data[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
            // std::uint8_t a_data_1[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
            // // Matrix B (RHS) is:
            // // |  1 |  3 |  5 |
            // // |  2 |  4 |  6 |
            // const std::uint8_t b_data[] = {1, 2, 3, 4, 5, 6};
            // std::uint8_t b_data_1[] = {1, 2, 3, 4, 5, 6};
            // // Here are the results we expect, from hand calculations:
            // // (1 * 7) + (3 * 8) + (5 * 9) = 76
            // // (2 * 7) + (4 * 8) + (6 * 9) = 100
            // // (1 * 10) + (3 * 11) + (5 * 12) = 103
            // // (2 * 10) + (4 * 11) + (6 * 12) = 136
            // // (1 * 13) + (3 * 14) + (5 * 15) = 130
            // // (2 * 13) + (4 * 14) + (6 * 15) = 172
            // // (1 * 16) + (3 * 17) + (5 * 18) = 157
            // // (2 * 16) + (4 * 17) + (6 * 18) = 208
            // // That means matrix C should be:
            // // |  76 | 103 | 130 | 157 |
            // // | 100 | 136 | 172 | 208 |
            // const std::uint8_t expected_data[] = {76, 100, 103, 136, 130, 172, 157, 208};

            // const int c_count = m * n;
            // std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);

            // const bool is_a_transposed = true;
            // const bool is_b_transposed = true;
            // const bool is_c_transposed = true;
            // const int lda = k;
            // const int ldb = n;
            // const int ldc = n;

            // const int a_offset = 0;
            // const int b_offset = 0;
            // const int c_offset = 0;
            // const int c_mult = 1;
            // const int c_shift = 0;

            // gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
            //     is_a_transposed, is_b_transposed, is_c_transposed, m, n, k, a_data,
            //     a_offset, lda, b_data, b_offset, ldb, output_data.get(), c_offset, c_mult,
            //     c_shift, ldc, gemmlowp::eight_bit_int_gemm::BitDepthSetting::A8B8);

            // uint32_t *test_output_int32 = (uint32_t*)calloc(1, m*n*sizeof(uint32_t));
            // gemm_nn_uint8_uint32(m, n, k, 1, a_data_1, k, b_data_1, n, test_output_int32, n);

            const int c_count = m * n;
            std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);
            // std::unique_ptr<std::uint32_t[]> output_data(new std::uint32_t[c_count]);
            const bool is_a_transposed = true;
            const bool is_b_transposed = true;
            const bool is_c_transposed = true;
            const int lda = k;
            const int ldb = n;
            const int ldc = n;

            const float real_multiplier = (*l.weight_data_int8_scales) * l.input_data_int8_scales / net.output_scale[l.count];
            // printf("layer:  %d, input quant scale: %f, weights quant scale: %f, otuput quant scale: %f\n", l.count, l.input_data_int8_scales, (*l.weight_data_int8_scales), net.output_scale[l.count]);
            // printf("real mul is %f\n", real_multiplier);
            std::int32_t quantized_multiplier;
            int right_shift;
            gemmlowp::eight_bit_int_gemm::QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier,
                                                                           &right_shift);

            const int a_offset = -l.input_data_int8_zero_point;
            const int b_offset = -(*l.weight_data_int8_zero_point);
            // const int c_offset = 0;
            const int c_offset = net.output_zero_point[l.count] - 90;
            const int c_mult = quantized_multiplier;
            const int c_shift = right_shift;

            // const int c_offset = 0;
            // const int c_mult = 1;
            // const int c_shift = 0;
            // printf("the weights offset is %d, input offset is %d\n", a_offset, b_offset);
            // printf("the quantized_multiplier is %d, right_shift is %d\n", c_mult, c_shift);
            // std::cout << "Offset RHS matrix:" << a_offset << std::endl;
            // std::cout << "Offset LHS matrix:" << b_offset << std::endl;
            // std::cout << "Offset Resut matrix:" << c_offset << std::endl;

            gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
                    is_a_transposed, is_b_transposed, is_c_transposed, m, n, k, a,
                    a_offset, lda, b, b_offset, ldb, output_data.get(), c_offset, c_mult,
                    c_shift, ldc, gemmlowp::eight_bit_int_gemm::BitDepthSetting::A8B8);

            // gemmlowp::eight_bit_int_gemm::EightBitIntGemm_gy(
            //         is_a_transposed, is_b_transposed, is_c_transposed, m, n, k, a,
            //         a_offset, lda, b, b_offset, ldb, output_data.get(), c_offset,
            //         ldc, gemmlowp::eight_bit_int_gemm::BitDepthSetting::A8B8);
            for (int s = 0; s < c_count; ++s){
                if (output_data[s] != 255 && output_data[s] != 0){
                    printf("layer: %d, the result is %d\n", l.count, output_data[s]);   
                }
            }
            // // for (int i = 0; i < m; ++i){
            // //     for (int j = 0; j < n; ++j){
            // //         if (c[i*n + j] != 255 && c[i*n + j] != 0){
            // //             printf("the optimise result is %d\n", c[i*n + j]); 
            // //         }
            // //     }
            // // }

            // // printf("the size of kernel is %d, time is %lf\n", l.size, what_time_is_it_now() - time);

            // y = alpha1 * conv(x)
            for (i = 0; i < l.n; ++i) {
                l.output_data_int8_scales[i] = l.input_data_int8_scales * (*l.weight_data_int8_scales);
                // l.output_data_int8_scales[i] = 0.00008;
                for (j = 0; j < l.out_w*l.out_h; ++j){
                    int out_index = i*l.out_w*l.out_h + j; 
                    // int32_t output_quant_value = l.output_quant[out_index] - l.weights_sum_int[i] - l.input_sum_int[j] \
                    //                              + l.mult_zero_point;
                    // l.output[out_index] = output_quant_value * l.output_data_int8_scales[i];   
                    // l.output[out_index] = output_data[out_index] * l.output_data_int8_scales[i];     
                    l.output[out_index] = (output_data[out_index] - c_offset)  * net.output_scale[l.count];
                }
            }

            // for (int i = 0; i < m; ++i){
            //     for (int j = 0; j < n; ++j){
            //         printf("the optimise result is %f\n", l.output[i*n + j]); 
            //     }
            // }
        }
	}

        
    //y = alpha1 * conv(x) + bias
    add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, l.out_c*l.out_h*l.out_w*l.batch, l.activation);

    free(net.input_quant);
    free(net.workspace_quant);
}

void forward_convolutional_layer_quant_google(convolutional_layer l, network net)
{
    int i, j;
    int input_index;

    net.input_quant = (uint8_t *)calloc(l.inputs, sizeof(uint8_t));

    for (input_index = 0; input_index < l.c*l.w*l.h; ++input_index) {
        uint8_t input_quant_value = clamp(round(net.input[input_index] / l.input_data_int8_scales) + l.input_data_int8_zero_point, 0, 255);
        net.input_quant[input_index] = input_quant_value;
    }
    // 1. Convolution !!!
    int batch_index,groups_index;

    // y = conv(x)
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_h*l.out_w;

    net.workspace_quant = (uint8_t*)calloc(1, l.out_h*l.out_w*l.size*l.size*l.c*sizeof(uint8_t));

    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            uint8_t *a = l.weights_quant + groups_index*l.nweights/l.groups;
            uint8_t *b = (uint8_t *)net.workspace_quant;
            uint32_t *c = l.output_quant + (batch_index*l.groups + groups_index)*n*m;  
            uint8_t *im =  net.input_quant + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_int8(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);    // here
            }

            for (int ss = 0; ss < l.out_w*l.out_h; ++ss){ 
                for (int tt = 0; tt < l.c*l.size*l.size; ++tt){
                    l.input_sum_int[ss] += b[tt*l.out_w*l.out_h+ss];
                }
                l.input_sum_int[ss] = l.input_sum_int[ss] * (*l.weight_data_int8_zero_point);
            }
            double time=what_time_is_it_now();
            // 0.29s for whole net forward
            gemm_nn_uint8_uint32(m, n, k, 1, a, k, b, n, c, n);

            // 0.53s
            // gemm_nn_uint8_uint32_conv32(m, n, k, 1, a, k, b, n, c, n);

            printf("the size of kernel is %d, time is %lf\n", l.size, what_time_is_it_now() - time);
            // 0.53s
	        // for (int t = 0; t < m; ++t) {
            //     gemm_nn_uint8_uint32_conv32(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
            // }
        }
	}

    // y = alpha1 * conv(x)
    for (i = 0; i < l.n; ++i) {
        l.output_data_int8_scales[i] = l.input_data_int8_scales * (*l.weight_data_int8_scales);
        for (j = 0; j < l.out_w*l.out_h; ++j){
            int out_index = i*l.out_w*l.out_h + j;
            int32_t output_quant_value = l.output_quant[out_index] - l.weights_sum_int[i] - l.input_sum_int[j] \
                                            + l.mult_zero_point;
  
            l.output[out_index] = output_quant_value * l.output_data_int8_scales[i];
        }
    }
        
    //y = alpha1 * conv(x) + bias
    add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, l.out_c*l.out_h*l.out_w*l.batch, l.activation);

    free(net.input_quant);
    free(net.workspace_quant);
}
#endif
void forward_convolutional_layer_nobn(convolutional_layer l, network net)
{
    int i, j;

    // char *file_name[100];
    // sprintf(file_name, "%d_input.txt", l.count);
    // FILE *fp = fopen(file_name, "w");
    // for(int s = 0; s < l.c*l.w*l.h; ++s){
    //     fprintf(fp, "%f\n", net.input[s]);
    // }

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
	
/* 	if(l.batch_normalize){
    	batch_normalize_weights(l.weights, l.rolling_variance, l.scales, l.out_c, l.size*l.size*l.c/l.groups); 
     	batch_normalize_bias(l.biases, l.rolling_mean, l.rolling_variance, l.scales, l.out_c); 
    } */
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
#ifdef MASK
	if((l.groups == 1) && (l.n != 30)){
            gemm_mask(l.size*l.size,l.c/l.groups,m,n,k,l.c,a,k,b,n,l.weights_mask+j,c,n);
#if 1			
            //printf("c is: %d, n is: %d, batch is: %d, size is: %d\n", l.c, l.n, l.batch, l.size); 
            int m = 0, n = 0, zero_num_mask = 0, zero_weights = 0;
            for(m = 0; m < l.c/l.groups*l.n; m++){
                float sum = 0;
                if(l.weights_mask[m] == 0){
                    zero_num_mask = zero_num_mask + 1;
                }
                for(n = 0; n < l.size*l.size; n++){
                    sum += l.weights[m*l.size*l.size + n];
                }
                if(sum == 0){
                    zero_weights = zero_weights + 1;
                }
            }
            printf("layer: %d  channel: %d  prune filters: %d, ration: %.2f%%\n", l.count, l.c/l.groups*l.n, zero_num_mask, (float)zero_num_mask*100/(l.c/l.groups*l.n));
            //printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_weights, (float)(zero_weights)/(l.c*l.n));
#endif	
	}else{
		gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
	}
#else		
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
#endif
        }
    }
    // float min_val = 0;
    // float max_val = 0;
    // float val = 0;
    // for(int s = 0; s < l.n*l.out_h*l.out_w; ++s){
    //     val = l.output[s];
    //     min_val = std::min(min_val, val);
    //     max_val = std::max(max_val, val);
    // }

    // const auto result_qparams = gemmlowp::eight_bit_int_gemm::ChooseQuantizationParams(min_val, max_val);
    // printf("layer: %d, output scale is %f, output zero point: %d\n", l.count, result_qparams.scale, result_qparams.zero_point);

    add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    //add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
    // char *file_name_o[100];
    // sprintf(file_name_o, "%d_outputput.txt", l.count);
    // FILE *fp_o = fopen(file_name_o, "w");
    // for(int s = 0; s < l.out_c*l.out_w*l.out_h; ++s){
    //     fprintf(fp_o, "%f\n", l.output[s]);
    // }
}


void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    // char *file_name[100];
    // sprintf(file_name, "%d_input.txt", l.count);
    // FILE *fp = fopen(file_name, "w");
    // for(int s = 0; s < l.c*l.w*l.h; ++s){
    //     fprintf(fp, "%f\n", net.input[s]);
    // }

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
	
/* 	if(l.batch_normalize){
    	batch_normalize_weights(l.weights, l.rolling_variance, l.scales, l.out_c, l.size*l.size*l.c/l.groups); 
     	batch_normalize_bias(l.biases, l.rolling_mean, l.rolling_variance, l.scales, l.out_c); 
    } */
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
#ifdef MASK
	if((l.groups == 1) && (l.n != 30)){
            gemm_mask(l.size*l.size,l.c/l.groups,m,n,k,l.c,a,k,b,n,l.weights_mask+j,c,n);
#if 1			
            //printf("c is: %d, n is: %d, batch is: %d, size is: %d\n", l.c, l.n, l.batch, l.size); 
            int m = 0, n = 0, zero_num_mask = 0, zero_weights = 0;
            for(m = 0; m < l.c/l.groups*l.n; m++){
                float sum = 0;
                if(l.weights_mask[m] == 0){
                    zero_num_mask = zero_num_mask + 1;
                }
                for(n = 0; n < l.size*l.size; n++){
                    sum += l.weights[m*l.size*l.size + n];
                }
                if(sum == 0){
                    zero_weights = zero_weights + 1;
                }
            }
            printf("layer: %d  channel: %d  prune filters: %d, ration: %.2f%%\n", l.count, l.c/l.groups*l.n, zero_num_mask, (float)zero_num_mask*100/(l.c/l.groups*l.n));
            //printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_weights, (float)(zero_weights)/(l.c*l.n));
#endif	
	}else{
		gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
	}
#else		
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
#endif
        }
    }
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    }else{
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
    //add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
    // char *file_name_o[100];
    // sprintf(file_name_o, "%d_outputput.txt", l.count);
    // FILE *fp_o = fopen(file_name_o, "w");
    // for(int s = 0; s < l.out_c*l.out_w*l.out_h; ++s){
    //     fprintf(fp_o, "%f\n", l.output[s]);
    // }
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image*)calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

