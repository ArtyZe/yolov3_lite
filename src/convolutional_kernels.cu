#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "algorithm"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "algorithm"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += abs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}





/**************************prune network weights*************************/

__global__ void prune_kernel(int N, float *weights,float *update_weights, float threshold, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if (fabs(weights[i*INCX])<threshold){
            weights[i*INCX]=0;
            update_weights[i*INCX] = 0;
        }
    }
}

void prune_gpu(int N, float * X, float * Y, float threhold,int INCY)
{
    prune_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X,  Y,threhold, INCY);
    check_error(cudaPeekAtLastError());
}


#ifdef MASK

__global__ void mask_weights_kernel(int N, int channel, int size, float *weights, float *weights_mask, float *weights_mask_binary, float threhold, int key)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
		int s;
		int zero_num = 0;
		int c = i%channel;  //channel
		i /= channel;
		int b = i;  //batch
		int mask_index = b*channel + c;  
		for(s = 0; s < size*size; s++){
			int weight_index = (b*channel + c)*size*size + s;
#if 0			
			if(weights_mask[mask_index] != 1){
					printf("the mask is %f, zero num is %d\n", weights_mask[mask_index], zero_num);
					zero_num = zero_num + 1;
			}
#endif			
			weights[weight_index] *= weights_mask[mask_index];
		}
	}
}

void mask_weights_gpu(int N, int channel, int size, float * X, float * Y, float * Z, float threhold, int key)
{
    //printf("conv forward\n");
    mask_weights_kernel<<<cuda_gridsize(N), BLOCK>>>(N, channel, size, X, Y, Z, threhold, key);
	//printf("all connection: %d, prune connection %d, rate: %f\n", channel, zero_num, zero_num/channel);
    check_error(cudaPeekAtLastError());
}

//l.c*l.n*l.batch, l.c*l.n, l.size, l.weights_result_gpu, l.	, l.weights_mask_gpu, l.weight_mask_updates_gpu

__global__ void mask_backward_kernel(int N, int channel, int size, float *weights_result, float *weight_updates, float *weights_mask, float *mask_updates)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
		int s;
		float sum =0;
		int c = i%channel;	//channel
		i /= channel;
		int b = i;	//batch

		int mask_index = b*channel + c;
		for(s = 0; s < size*size; s++){
				int weight_index = (b*channel + c)*size*size + s;
				sum += weight_updates[weight_index]*weights_mask[mask_index]*weights_result[weight_index];
		}
		mask_updates[mask_index] += sum;
		if(mask_updates[mask_index]<0){
			mask_updates[mask_index]= -1 * mask_updates[mask_index];
		}
    }
}

void mask_backward_gpu(int N, int channel, int size, float * X, float * Y, float * Z, float *updates)
{
    //printf("conv backward\n");
    mask_backward_kernel<<<cuda_gridsize(N), BLOCK>>>(N, channel, size, X, Y, Z, updates);
    check_error(cudaPeekAtLastError());
}

__global__ void mask_update_kernel(int N, int channel, int size, float *weights_mask, float *mask_updates, float threshold)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
		int c = i%channel;	//channel
		i /= channel;
		int b = i;	//batch

		int mask_index = b*channel + c;
		if(mask_updates[mask_index] >= threshold){
			weights_mask[mask_index] = 1;
			//if(weights_mask[mask_index] > 0.5) {
			//printf("the threshold is %f, mask updates value is %f\n", threshold, weights_mask[mask_index]);
			//}
		}else{
			weights_mask[mask_index] = 0;
			//if(weights_mask[mask_index] > 0.5){ 
			//printf("the mask updates value is %f\n", mask_updates[mask_index]);
			//}
		}
    }
    //if(threshold != 0) {printf("the threshold is %f\n", threshold);}
}

void mask_update_gpu(int N, int channel, int size, float * X, float * Y, float threshold)
{
    mask_update_kernel<<<cuda_gridsize(N), BLOCK>>>(N, channel, size, X, Y, threshold);
    check_error(cudaPeekAtLastError());
}

#endif


void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    //printf("this is %d layer\n", l.count);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else

#ifdef MASK
	if(l.count >= IGNORENUM){
		copy_gpu(l.c*l.n*l.size*l.size, l.weights_gpu, 1, l.weights_result_gpu, 1);
			  //	int N, int channel, int size, float *weights, float *weights_mask, float *weights_mask_binary, float threhold, int key)
		mask_weights_gpu(l.c*l.n*l.batch, l.c*l.n, l.size, l.weights_gpu, l.weights_mask_gpu, l.weights_mask_binary_gpu, 0.5, 0);
#if 0			
		//printf("c is: %d, n is: %d, batch is: %d, size is: %d\n", l.c, l.n, l.batch, l.size); 
		printf("this is %d layer\n", l.count);
		cuda_pull_array(l.weights_mask_gpu, l.weights_mask, l.c*l.n);
		cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
		int m = 0, n = 0, zero_num_mask = 0, zero_weights = 0;
		for(m = 0; m < l.c*l.n; m++){
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
		//printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_num, (float)(zero_num)/(l.c*l.n));
		printf("channel: %d, prune connection: %d, ration: %f\n", l.c*l.n, zero_weights, (float)(zero_weights)/(l.c*l.n));
#endif			
	}
#endif

    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        im2col_gpu(net.input_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.workspace);
        float * a = l.weights_gpu;
        float * b = net.workspace;
        float * c = l.output_gpu;
        gemm_gpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
#endif
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.);
    int h_offset = -(size/2.);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = net.workspace;
        float * c = l.weight_updates_gpu;

        im2col_gpu(net.input_gpu + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, net.workspace);
        gemm_gpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);
#if MASK
		//(int N, int channel, int size, float *weights_result, float *weight_updates, float *weights_mask, float *mask_updates)
		if(l.count >= IGNORENUM){
			//printf("this is %d layer BACKWARD\n", l.count);
			//mask_backward_weights_gpu(l.c*l.n*l.batch, l.c*l.n,  l.size, l.weight_updates_gpu, l.weights_mask_gpu); 
			mask_backward_gpu(l.c*l.n*l.batch, l.c*l.n, l.size, l.weights_result_gpu, l.weight_updates_gpu, l.weights_mask_gpu, l.weight_mask_updates_gpu);
			copy_gpu(l.c*l.n*l.size*l.size, l.weights_result_gpu, 1, l.weights_gpu, 1);
		}
#endif	
        if(net.delta_gpu){		
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = net.workspace;

            gemm_gpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_gpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);    
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);

#ifdef MASK    
    cuda_pull_array(layer.weights_mask_gpu, layer.weights_mask, layer.c*layer.n);
	cuda_pull_array(layer.weight_mask_updates_gpu, layer.weight_mask_updates, layer.c*layer.n);
#if 0		
	int i = 0;
	for(i = 0; i < layer.c*layer.n; i++)
	{
		//if(layer.weights_mask[i] != 1){
			printf("the mask value is %f\n", layer.weights_mask[i]);
		//}
	}
#endif	
#endif
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);

#ifdef MASK    
    cuda_push_array(layer.weights_mask_gpu, layer.weights_mask, layer.c*layer.n);
	cuda_push_array(layer.weight_mask_updates_gpu, layer.weight_mask_updates, layer.c*layer.n);
#if 0		
	int i = 0;
	for(i = 0; i < layer.c*layer.n; i++)
	{
		if(layer.weights_mask[i] != 1){
			printf("the mask value is %f\n", layer.weights_mask[i]);
		}
	}
#endif		
#endif

    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

int cmp( const void *a , const void *b )
{
		return *(float *)a > *(float *)b ? 1 : -1; 
}


void update_convolutional_layer_gpu(layer l, update_args a)
{

    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
#ifdef PRUNE
    prune_gpu(size,l.weights_gpu,l.weight_updates_gpu,0.001,1);

#endif
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
#ifdef MASK
		//printf("this is %d layer UPDATE\n", l.count);
		cuda_pull_array(l.weight_mask_updates_gpu, l.weight_mask_updates, l.c*l.n);
		qsort(l.weight_mask_updates, l.c*l.n, sizeof(float), cmp);
		float threshold = l.weight_mask_updates[(int)(l.c*l.n*0.1)];
#if 0				
		int i;
		for(i = 0; i < l.c*l.n; i++){
			if(l.weight_mask_updates[i] > 0) printf("the ort is %d, mask update value is %f\n", i, l.weight_mask_updates[i]);
		}
		if(threshold > 0){
			printf("the channel is %d, ort is %d, threshold value is %f\n", l.c*l.n, (int)(l.c*l.n*0.2), threshold);
		}
#endif
		if(l.count >= IGNORENUM){
			//mask_backward_gpu(l.c*l.n*l.batch, l.c*l.n, l.size, l.weights_result_gpu, l.delta_gpu, l.weights_mask_gpu, l.weight_mask_updates_gpu);
			mask_update_gpu(l.c*l.n*l.batch, l.c*l.n, l.size, l.weights_mask_gpu, l.weight_mask_updates_gpu, threshold);
		}
#endif        
        if(l.scales_gpu){
#ifdef SCALE_L1
			cuda_pull_array(l.scales_gpu, l.scales, l.n);
			int i;
			int zero_Num = 0;
			for(i = 0; i < l.n; i++)
			{
							l.sign_scales[i]= l.scales > 0 ? 1:-1;
					if(l.scales[i] < 0.5)
					{
							zero_Num++;
							//printf("Prune Number: %d, Value: %f\n", zero_Num, l.scales[i]);
						}
			}
#if 0
			printf("Prune Number: %d, All: %d\n", zero_Num, l.n);
#endif		      	
			cuda_push_array(l.sign_scales_gpu, l.sign_scales, l.n);

			axpy_gpu(l.n, -decay*batch, l.sign_scales_gpu, 1, l.scale_updates_gpu, 1);   //L1 regularization	
#endif		      	
		        //axpy_gpu(l.n, -decay*batch, l.scales_gpu, 1, l.scale_updates_gpu, 1);   //L2 regularization	
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}


