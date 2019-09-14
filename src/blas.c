#include "blas.h"
#include "omp.h"
#include <stdint.h>

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W_MAX_VAL (256/2 - 1)    // 7-bit (1-bit sign)
#define I_MAX_VAL (256/2 - 1)    // 7-bit (1-bit sign)
#define R_MAX_VAL (256*256/2 - 1)    // 31-bit (1-bit sign)
int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : (-1)*max_val;
    return src;
}

short int max_abs_short(short int src, short int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : (-1)*max_val;
    return src;
}
float entropy_calibration(float *src_arr, const size_t size, const float bin_width, const int max_bin)
{
    //const float bin_width = 1.0 / 4096;// 1.0F / 64.0F;
    //const int max_bin = 2048*2;// 2048;
    const int max_global_val = max_bin * bin_width;    // 1024    // 32
    float *m_array = (float*)calloc(max_bin, sizeof(float));
    float *H_histogram = (float*)calloc(max_bin, sizeof(float));
    float *P_array = (float*)calloc(max_bin, sizeof(float));
    float *Q_array = (float*)calloc(max_bin, sizeof(float));
    float *quant_Q_array = (float*)calloc(128, sizeof(float));    // 128 for INT8
    uint64_t *quant_Q_array_count = (uint64_t*)calloc(128, sizeof(uint64_t));    // 128 for INT8

    int i, j;
    {
        //uint64_t outliers = 0;
        const int last_bin = max_bin - 1;
        for (j = 0; j <= last_bin; ++j) P_array[j] = 0;
        for (j = 0; j < size; ++j) {
            int bin_num = lround(fabs(src_arr[j]) / bin_width);
            int bin_num_saturated = (bin_num >= last_bin) ? last_bin : bin_num;
            H_histogram[bin_num_saturated]++;

            //if (bin_num > last_bin) outliers++;
            //else H_histogram[bin_num]++;
        }
    }

    for (i = 128; i < max_bin; ++i) {    // [1/64; 1024] // [1/64; 32]
                                        //if (i > max_bin) printf(" i > max_bin = %d, ", i);
                                        //printf(" %d \r", i);
                                        // calculate bin histogram
        uint64_t outliers = 0;
        const int last_bin = i - 1;
        for (j = 0; j <= last_bin; ++j) P_array[j] = 0;
        /*for (j = 0; j < size; ++j) {
        int bin_num = lround(fabs(src_arr[j]) / bin_width);
        //int bin_num_saturated = (bin_num >= last_bin) ? last_bin : bin_num;
        if (bin_num > last_bin) outliers++;
        else P_array[bin_num]++;
        }*/
        for (j = 0; j < max_bin; ++j) {
            if (j <= last_bin) P_array[j] = H_histogram[j];
            else outliers += H_histogram[j];
        }
        // quantinization P-i-bins to Q-128-bins
        const float quant_expand_width = i / 128.0F;
        for (j = 0; j < 128; ++j) quant_Q_array[j] = 0, quant_Q_array_count[j] = 0;
        for (j = 0; j < i; ++j) {
            int quant_bin = lround(j / quant_expand_width);
            if (quant_bin > 127) quant_bin = 127; // printf(" quant_bin > 127 = %d \n", quant_bin);
            quant_Q_array[quant_bin] += P_array[j];
            if (P_array[j] != 0) quant_Q_array_count[quant_bin]++;
        }
        // expand 128-bins to i-bins
        for (j = 0; j < i; ++j) Q_array[j] = 0;
        for (j = 0; j < i; ++j) {
            int quant_bin = lround(j / quant_expand_width);
            if (quant_bin > 127) quant_bin = 127;// printf(" quant_bin > 127 = %d \n", quant_bin);
                                                 //Q_array[j] = llround(quant_Q_array[quant_bin] / quant_expand_width);
            if (P_array[j] != 0)    // preserve empty bins from original P
                Q_array[j] = quant_Q_array[quant_bin] / quant_Q_array_count[quant_bin];
            //printf(" quant_bin = %d, Q[j] = %f = q_Q %f / q_w %f, P = %f \n", quant_bin, Q_array[j], quant_Q_array[quant_bin], quant_expand_width, P_array[j]);
        }
        P_array[last_bin] += outliers;    // saturation
                                        // P /= SUM(P); Q /= SUM(Q);
        float sum_P = 0, sum_Q = 0, quant_sum_Q = 0;
        for (j = 0; j < 128; ++j) quant_sum_Q += quant_Q_array[j];
        for (j = 0; j < i; ++j) {
            sum_P += P_array[j];
            sum_Q += Q_array[j];
            //printf(" P_array = %f, Q_array = %f \n", P_array[j], Q_array[j]);
        }
        for (j = 0; j < i; ++j) {
            P_array[j] /= sum_P;
            Q_array[j] /= sum_Q;
        }
        // KL_divergence(P, Q);
        for (j = 0; j < i; ++j) {
            m_array[i] += P_array[j] * (log((P_array[j] + FLT_MIN) / (Q_array[j] + FLT_MIN)));
            //printf(" p = %f, q = %f, p/q = %f, log(p/q) = %f, m = %f \n", P_array[j], Q_array[j], P_array[j] / Q_array[j], log((P_array[j] + FLT_MIN) / (Q_array[j] + FLT_MIN)), m_array[i]);
        }
        //printf("\n i = %d, size = %zu, sum_P = %f, sum_Q = %f, q_sum_Q = %f, q_e_width = %f, m = %f \n", i, size, sum_P, sum_Q, quant_sum_Q, quant_expand_width, m_array[i]);
        //getchar();
    }

    float m_index = 128, min_m = FLT_MAX;
    for (i = 128; i < max_bin; ++i) {
        if (m_array[i] < min_m) {
            min_m = m_array[i];
            m_index = i;
        }
    }

    float threshold = (m_index + 0.5) * bin_width;
    float multiplier = 127 / threshold;
    printf(" mult = %g, threshold = %g, min_m = %g, m_index = %g \n", multiplier, threshold, min_m, m_index);

    free(H_histogram);
    free(P_array);
    free(Q_array);
    free(quant_Q_array);
    free(quant_Q_array_count);
    free(m_array);
    //getchar();

    return multiplier;
}
int * get_distribution(float *arr_ptr, int arr_size, int number_of_ranges, float start_range)
{
    //const int number_of_ranges = 32;
    //const float start_range = 1.F / 65536;
    printf("start to malloc memory for multipier\n");
    int *count = calloc(number_of_ranges, sizeof(int));
    float min_val = 10000, max_val = 0;

    int i, j;
    for (i = 0; i < arr_size; ++i) {
        float w = arr_ptr[i];

        float cur_range = start_range;
        for (j = 0; j < number_of_ranges; ++j) {
            if (fabs(cur_range) <= w && w < fabs(cur_range * 2))
                count[j]++;// , printf("found \n");
            cur_range *= 2;
            //printf("%f, ", w);
        }
    }

    return count;
}
float get_multiplier(float *arr_ptr, int arr_size, int bits_length)
{
    const int number_of_ranges = 32;
    const float start_range = 1.F / 65536;

    int i, j;
    int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

    int max_count_range = 0;
    int index_max_count = 0;
    for (j = 0; j < number_of_ranges; ++j) {
        int counter = 0;
        for (i = j; i < (j + bits_length) && i < number_of_ranges; ++i)
        {
            counter += count[i];
            //counter += log2(count[i]);
        }
        if (max_count_range < counter) {
            max_count_range = counter;
            index_max_count = j;
        }
    }
    //index_max_count = index_max_count + 2;    // optimal shift multipler
    float multiplier = 1 / (start_range * powf(2., (float)index_max_count));
    //printf(" max_count_range = %d, index_max_count = %d, multiplier = %g \n",
    //    max_count_range, index_max_count, multiplier);
    free(count);
    return multiplier;
}

// Quantinization and get multiplers for convolutional weights for quantinization
void quantinization_and_get_multipliers(network *net)
{

    // ----------- entropy_calibration(,, 1.0 / 16, 4096); - FULL ----------------------
    //float input_mult[] = { 256, 4,32,64,32,32,32,32,32,64,64,64,64,64,128,64,128,128,64,128,64,128,128 };    // divided 4 - full works
    int counter = 0;
    //const int input_mult_size = sizeof(input_mult) / sizeof(float);

    int j;
    for (j = 0; j < net->n; ++j) {
        layer *l = &net->layers[j];
        if (l->type == CONVOLUTIONAL) {
            size_t const weights_size = l->size*l->size*l->c/l->groups*l->n;
            size_t const filter_size = l->size*l->size*l->c/l->groups;
            if(l->batch_normalize){
                batch_normalize_weights(l->weights, l->rolling_variance, l->scales, l->out_c, l->size*l->size*l->c/l->groups); 
                batch_normalize_bias(l->biases, l->rolling_mean, l->rolling_variance, l->scales, l->out_c); 
            }
            int i, k, fil;
            float old_weight_mult = get_multiplier(l->weights, weights_size, 8) / 4;    // good [2 - 8], best 4
            float weights_multiplier_single = old_weight_mult;
            l->weights_quant_multipler = weights_multiplier_single;
            for (fil = 0; fil < l->n; ++fil) {
                for (i = 0; i < filter_size; ++i) {
                    float w = l->weights[fil*filter_size + i] * l->weights_quant_multipler;// [fil];
                    l->weights_int8[fil*filter_size + i] = max_abs(w, W_MAX_VAL);
                }
            }
            if (counter >= net->input_calibration_size) {
                printf("\n Warning: input_calibration= in the cfg-file has less values %d than convolutional layers %d \n",
                    net->input_calibration_size, counter);
            }
            //l->input_quant_multipler = 40;//(counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;    // best 40
            l->input_quant_multipler = (counter < net->input_calibration_size) ? net->input_calibration[counter] : 40;
            ++counter;

            //float current_input_mult = 40;//(counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;
            float current_input_mult = (counter < net->input_calibration_size) ? net->input_calibration[counter] : 40;

            for (fil = 0; fil < l->n; ++fil) {
                if (counter == 1) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
                if (counter == 2) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
                else if (counter >= 2) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
            }
            // quantinization Biases
            for (fil = 0; fil < l->n; ++fil) {
                // calculate optimal multipliers - for Biases
                float biases_multipler = (l->output_multipler * l->weights_quant_multipler * l->input_quant_multipler / R_MULT);

                l->biases_quant[fil] = l->biases[fil] * biases_multipler;
            }
            printf(" Multiplers: weights %g, input %g, output %g \n",
                l->weights_quant_multipler, l->input_quant_multipler, l->output_multipler);
        }
        else {
            printf(" Skip layer: %d \n", l->type);
        }
    }
}
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < spatial; ++i){
            float sum = 0;
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                sum += powf(x[index], 2);
            }
            sum = sqrtf(sum);
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                x[index] /= sum;
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}


void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void batch_normalize_weights(float *weights, float *variance, float *scales, int filters, int spatial)
{
	int i = 0,j = 0;
	//#pragma omp parallel for num_threads(8)
	for(i = 0; i < filters; i++){
		for(j = 0; j < spatial; j++){
			int weights_index = i*spatial + j;
			weights[weights_index] = weights[weights_index]*scales[i]/(sqrt(variance[i]) + .000001f);
		}
	}
}
	
void batch_normalize_bias(float *biases, float *rolling_mean, float *rolling_variance, float *scales, int filters)
{
	int i = 0;
	//#pragma omp parallel for num_threads(8)
	for(i = 0; i < filters; i++){
        biases[i] = biases[i]-scales[i]*rolling_mean[i]/(sqrt(rolling_variance[i]) + .000001f);
	}
}
void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}


