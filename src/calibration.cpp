#include "calibration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define GEMMCONV

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

// popcnt 32 bit
static inline int popcnt_32(uint32_t val32) {
#ifdef WIN32  // Windows MSVS
    int tmp_count = __popcnt(val32);
#else   // Linux GCC
    int tmp_count = __builtin_popcount(val32);
#endif
    return tmp_count;
}

static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

uint32_t reverse_32_bit(uint32_t a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

#ifdef AVX

#ifdef _WIN64
// Windows
#include <intrin.h>
#else
// Linux
#include <x86intrin.h>
#endif

#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=broad&expand=561

// https://software.intel.com/sites/landingpage/IntrinsicsGuide
void gemm_nn_256bit(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA*A[i*lda + k];
            __m256 a256, b256, c256, result256;    // AVX
            a256 = _mm256_set1_ps(A_PART);
            for (j = 0; j < N - 8; j += 8) {
                b256 = _mm256_loadu_ps(&B[k*ldb + j]);
                c256 = _mm256_loadu_ps(&C[i*ldc + j]);
                // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                //result256 = _mm256_fmadd_ps(a256, b256, c256);
                result256 = _mm256_mul_ps(a256, b256);
                result256 = _mm256_add_ps(result256, c256);
                _mm256_storeu_ps(&C[i*ldc + j], result256);
            }

            int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
            for (j = prev_end; j < N; ++j)
                C[i*ldc + j] += A_PART*B[k*ldb + j];
        }
    }
}

#if defined(_MSC_VER) && _MSC_VER <= 1900
static inline __int32 _mm256_extract_epi64(__m256i a, const int index) {
    return a.m256i_i64[index];
}

static inline __int32 _mm256_extract_epi32(__m256i a, const int index) {
    return a.m256i_i32[index];
}
#endif

static inline float _castu32_f32(uint32_t a) {
    return *((float *)&a);
}

#if defined(_MSC_VER)
// Windows
static inline float _mm256_extract_float32(__m256 a, const int index) {
    return a.m256_f32[index];
}
#else
// Linux
static inline float _mm256_extract_float32(__m256 a, const int index) {
    return _castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), index));
}
#endif

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    printf("======================\n");
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)// && is_fma_avx())
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    _mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        //printf("\n Error: is no non-optimized version \n");
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    }
}

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        {
            __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
            __m256 all256_01 = _mm256_set1_ps(0.1F);

            for (i = 0; i < n - 8; i += 8) {
                //x[i] = (x[i]>0) ? x[i] : .1*x[i];

                __m256 src256 = _mm256_loadu_ps(&x[i]);
                __m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

                __m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

                __m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
                _mm256_storeu_ps(&x[i], result256);
            }
        }

        for (; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{

    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;
    int b, k;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                //for (j = 0; j < out_w; ++j) {
                j = 0;

                if (stride == 1) {
                    for (j = 0; j < out_w - 8 - (size - 1); j += 8) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        __m256 max256 = _mm256_set1_ps(-FLT_MAX);
                        for (n = 0; n < size; ++n) {
                            for (m = 0; m < size; ++m) {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                max256 = _mm256_max_ps(src256, max256);
                            }
                        }
                        _mm256_storeu_ps(&dst[out_index], max256);

                    }
                }
                else if (size == 2 && stride == 2) {
                    for (j = 0; j < out_w - 4; j += 4) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        __m128 max128 = _mm_set1_ps(-FLT_MAX);

                        for (n = 0; n < size; ++n) {
                            //for (m = 0; m < size; ++m)
                            m = 0;
                            {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                __m256 src256_2 = _mm256_permute_ps(src256, (1 << 0) | (3 << 4));
                                __m256 max256 = _mm256_max_ps(src256, src256_2);

                                __m128 src128_0 = _mm256_extractf128_ps(max256, 0);
                                __m128 src128_1 = _mm256_extractf128_ps(max256, 1);
                                __m128 src128 = _mm_shuffle_ps(src128_0, src128_1, (2 << 2) | (2 << 6));

                                max128 = _mm_max_ps(src128, max128);
                            }
                        }
                        _mm_storeu_ps(&dst[out_index], max128);
                    }
                }

                for (; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}



#endif // AVX

static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}

static inline int popcnt_64(uint64_t val64) {
#ifdef WIN32  // Windows
#ifdef _WIN64 // Windows 64-bit
    int tmp_count = __popcnt64(val64);
#else         // Windows 32-bit
    int tmp_count = __popcnt(val64);
    tmp_count += __popcnt(val64 >> 32);
#endif
#else   // Linux
#ifdef __x86_64__  // Linux 64-bit
    int tmp_count = __builtin_popcountll(val64);
#else  // Linux 32-bit
    int tmp_count = __builtin_popcount(val64);
    tmp_count += __builtin_popcount(val64);
#endif
#endif
    return tmp_count;
}
// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_cpu(layer l, network state)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

    // fill zero (ALPHA)
    for (i = 0; i < l.outputs*l.batch; ++i) l.output[i] = 0;

    // 1. Convolution !!!
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    // convolution as GEMM (as part of BLAS)
    for (i = 0; i < l.batch; ++i) {
        printf("the input is %f, batch=%d, c=%d, h=%d, w=%d, size=%d, stride=%d, pad=%d\n", state.input[100], l.batch, l.c, l.h, l.w, l.size, l.stride, l.pad);
        im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2
        printf("the input is %f, batch=%f, c=%f, h=%f, w=%d, size=%d, stride=%d, pad=%d\n", b[100],  b[101],  b[102],  b[103], l.w, l.size, l.stride, l.pad);
/*         char *file_name[100];
        sprintf(file_name, "%d_im_output_3.txt", i);
        FILE *fp = fopen(file_name, "w");
        for(int s = 0; s < l.c*l.w*l.h; ++s){
            fprintf(fp, "%f\n", b[s]);
        }
        printf("---------------------\n");
        sleep(2); */
        int t;
        #pragma omp parallel for
        for (t = 0; t < m; ++t) {
            gemm_nn_256bit(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
        }
        c += n*m;
        state.input += l.c*l.h*l.w;

    }
    printf("the input is %f, batch=%d, c=%d, h=%d, w=%d, size=%d, stride=%d, pad=%d\n", l.output[100], l.batch, l.c, l.h, l.w, l.size, l.stride, l.pad);

    int const out_size = out_h*out_w;

    // 2. Batch normalization
    if (l.batch_normalize) {
        int b;
        for (b = 0; b < l.batch; b++) {
            for (f = 0; f < l.out_c; ++f) {
                for (i = 0; i < out_size; ++i) {
                    int index = f*out_size + i;
                    l.output[index+b*l.outputs] = (l.output[index+b*l.outputs] - l.rolling_mean[f]) / (sqrtf(l.rolling_variance[f]) + .000001f);
                }
            }

            // scale_bias
            for (i = 0; i < l.out_c; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j+b*l.outputs] *= l.scales[i];
                }
            }
        }
    }

    // 3. Add BIAS
    //if (l.batch_normalize)
    {
        int b;
        for (b = 0; b < l.batch; b++) {
            for (i = 0; i < l.n; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j + b*l.outputs] += l.biases[i];
                }
            }
        }
    }

    // 4. Activation function (LEAKY or LINEAR)
    //if (l.activation == LEAKY) {
    //    for (i = 0; i < l.n*out_size; ++i) {
    //        l.output[i] = leaky_activate(l.output[i]);
    //    }
    //}
    //activate_array_cpu_custom(l.output, l.n*out_size, l.activation);
    activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

}

void forward_maxpool_layer_cpu(const layer l, network state)
{
    forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
}

float entropy_calibration(float *src_arr, const size_t size, const float bin_width, const int max_bin)
{
    //const float bin_width = 1.0 / 4096;// 1.0F / 64.0F;
    //const int max_bin = 2048*2;// 2048;
    // size = l.w*l.h*l.c
    float *m_array = (float*)calloc(max_bin, sizeof(float));
    float *H_histogram = (float*)calloc(max_bin, sizeof(float));
    float *P_array = (float*)calloc(max_bin, sizeof(float));
    float *Q_array = (float*)calloc(max_bin, sizeof(float));
    float *quant_Q_array = (float*)calloc(128, sizeof(float));    // 128 for INT8
    uint64_t *quant_Q_array_count = (uint64_t*)calloc(128, sizeof(uint64_t));    // 128 for INT8

    int i, j;
    {
        //uint64_t outliers = 0;
        const int last_bin = max_bin - 1;  //4095
        for (j = 0; j <= last_bin; ++j) P_array[j] = 0;  //initialize the p array

        // process every value in input tensor, draw the histogram distribution image
        for (j = 0; j < size; ++j) {
            int bin_num = lround(fabs(src_arr[j]) / bin_width);   //the index of current input
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
        const float quant_expand_width = i / 128.0F; //the quant scale for current i
        for (j = 0; j < 128; ++j) quant_Q_array[j] = 0, quant_Q_array_count[j] = 0;
        for (j = 0; j < i; ++j) {
            int quant_bin = lround(j / quant_expand_width);
            if (quant_bin > 127) quant_bin = 127; // printf(" quant_bin > 127 = %d \n", quant_bin);
            quant_Q_array[quant_bin] += P_array[j];
            if (P_array[j] != 0) quant_Q_array_count[quant_bin]++;
        }

        // expand 128-bins to i-bins
        ///////////////////////////////////////////////////////////////////
        // for example, float 1.5, 2.8. 4.9 quant to int 20
        // quant_Q_array[20] = P_array[1.5] + P_array[2.8] + P_array[4.9] = 5+ 4 + 6 = 15
        // quant_Q_array_count[20] = 3
        // after expand to i-bins, Q_array[1.5] = Q_array[2.8] = Q_array[4.9] = quant_Q_array[20]/3 = 5
        ///////////////////////////////////////////////////////////////////
        for (j = 0; j < i; ++j) Q_array[j] = 0;
        for (j = 0; j < i; ++j) {
            int quant_bin = lround(j / quant_expand_width);

            // acutually this condition should not happen
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

float *network_calibrate_cpu(network net, float *input)
{
    net.input = input;
    net.train = 0;
    // input calibration - for quantinization
    static int max_num = 100;
    static int counter = 0;
    static float *input_mult_array = NULL;

    max_num = net.calibrate_round;
    if (input_mult_array == NULL) {
        input_mult_array = (float *)calloc(net.n * max_num, sizeof(float));
    }
    ++counter;
    // save calibration coefficients
    if (counter > max_num) {
        printf("\n\n Saving coefficients to the input_calibration.txt file... \n\n");
        FILE* fw = fopen("input_calibration.txt", "wb");
        char buff[1024];
        //printf("\n float input_mult[] = { ");
        char *str1 = "input_calibration = ";
        printf("%s", str1);
        fwrite(str1, sizeof(char), strlen(str1), fw);
        int i;
        for (i = 0; i < net.n; ++i)
            if (net.layers[i].type == CONVOLUTIONAL) {
                printf("%g, ", input_mult_array[0 + i*max_num]);
                sprintf(buff, "%g, ", input_mult_array[0 + i*max_num]);
                fwrite(buff, sizeof(char), strlen(buff), fw);
            }
        char *str2 = "16";
        printf("%s \n ---------------------------", str2);
        fwrite(str2, sizeof(char), strlen(str2), fw);
        fclose(fw);
        getchar();
        exit(0);
    }

    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
                float multiplier = entropy_calibration(net.input, l.inputs, 1.0 / 16, 4096);
 
                printf(" multiplier = %f, l.inputs = %d \n\n", multiplier, l.inputs);
                input_mult_array[counter + i*max_num] = multiplier;
                if (counter >= max_num) {
                    int j;
                    float res_mult = 0;
                    for (j = 0; j < max_num; ++j)
                        res_mult += input_mult_array[j + i*max_num];
                    res_mult = res_mult / max_num;
                    input_mult_array[0 + i*max_num] = res_mult;
                    printf(" res_mult = %f, max_num = %d \n", res_mult, max_num);
                }
            //forward_convolutional_layer_cpu(l, net);
            forward_convolutional_layer(l, net);
            //printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == MAXPOOL) {
            //forward_maxpool_layer_cpu(l, net);
            forward_maxpool_layer(l, net);
            //printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == ROUTE) {
            forward_route_layer(l, net);
            //printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
        }
        else if (l.type == REORG) {
            forward_reorg_layer(l, net);
            //printf("\n REORG \n");
        }
        else if (l.type == REGION) {
            forward_region_layer(l, net);
            //printf("\n REGION \n");
        }
        else {
            printf("\n layer: %d \n", l.type);
        }


        net.input = l.output;
    }

    //int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}

void yolov2_fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - l->scales[f] * l->rolling_mean[f] / (sqrtf(l->rolling_variance[f]) + .000001f);

                    const size_t filter_size = l->size*l->size*l->c;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] = l->weights[w_index] * l->scales[f] / (sqrtf(l->rolling_variance[f]) + .000001f);
                    }
                }
                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else {
            //printf(" Skip layer: %d \n", l->type);
        }
    }
}
// parser.c
void load_convolutional_weights_cpu(layer l, FILE *fp)
{
    int num = l.n*l.c*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)) {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights, sizeof(float), num, fp);
    /*    if (l.adam) {
    fread(l.m, sizeof(float), num, fp);
    fread(l.v, sizeof(float), num, fp);
    }
    if (l.flipped) {
    transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }*/
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if (gpu_index >= 0) {
        push_convolutional_layer(l);
    }
#endif
}


void load_weights_upto_cpu(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0) {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        fread(net->seen, sizeof(uint64_t), 1, fp);
    }
    else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    //int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i) {
        layer l = net->layers[i];
        if (l.dontload) continue;
        if (l.type == CONVOLUTIONAL) {
            load_convolutional_weights_cpu(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void validate_calibrate_valid(char *datacfg, char *cfgfile, char *weightfile, int calibrate_round)
{
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    printf("valid=%s \n", valid_images);

    network *net = parse_network_cfg(cfgfile, 0);    // batch=1, quantized=0

    if (!calibrate_round) {
        printf("\n -calibrate_round <number> - isn't specified in command line, will be used 1000 images \n\n");
        calibrate_round = 1000;
    }
    net->calibrate_round = calibrate_round;

    if (weightfile) {
        load_weights_upto_cpu(net, weightfile, net->n);
    }
    //set_batch_network(&net, 1);
    yolov2_fuse_conv_batchnorm(*net);
    srand(time(0));

#ifdef GPU
    size_t workspace_size = 0;
    for (int j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        size_t cur_workspace_size = (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
        if (cur_workspace_size > workspace_size) workspace_size = cur_workspace_size;
    }
    cudaFree(net->workspace);
    net->workspace = (float*)calloc(1, workspace_size);
#endif // GPU

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    int m = plist->size;
    int i = 0;
    int t;

    int nthreads = 4;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net->w;
    args.h = net->h;
    args.type = IMAGE_DATA;
    //args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            float *X = val_resized[t].data;

            network_calibrate_cpu(*net, X);
        }
    }
}