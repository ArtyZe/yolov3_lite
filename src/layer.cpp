#include "layer.h"
#include "cuda.h"

#include <stdlib.h>

void init_layer(layer& l){
    if(l.type == DROPOUT){
        l.rand=nullptr;
#ifdef GPU
       l.rand_gpu=nullptr;
#endif
        return;
    }
    l.cweights=nullptr;
	l.indexes=nullptr;
    l.input_layers=nullptr;
    l.input_sizes=nullptr;
    l.map=nullptr;
    l.rand=nullptr;
    l.cost=nullptr;
    l.state=nullptr;
    l.prev_state=nullptr;
    l.forgot_state=nullptr;
    l.forgot_delta=nullptr;
    l.state_delta=nullptr;
    l.concat=nullptr;
    l.concat_delta=nullptr;
    l.binary_weights=nullptr;
    l.biases=nullptr;
    l.bias_updates=nullptr;
    l.scales=nullptr;
    l.scale_updates=nullptr;
    l.weights=nullptr;
    l.weight_updates=nullptr;
    l.delta=nullptr;
    l.output=nullptr;
    l.squared=nullptr;
    l.norms=nullptr;
    l.spatial_mean=nullptr;
    l.mean=nullptr;
    l.variance=nullptr;
    l.mean_delta=nullptr;
    l.variance_delta=nullptr;
    l.rolling_mean=nullptr;
    l.rolling_variance=nullptr;
    l.x=nullptr;
    l.x_norm=nullptr;
    l.m=nullptr;
    l.v=nullptr;
    l.z_cpu=nullptr;
    l.r_cpu=nullptr;
    l.h_cpu=nullptr;
    l.binary_input=nullptr;
    l.softmax_tree=nullptr;

#ifdef GPU
    l.indexes_gpu=nullptr;

    l.z_gpu=nullptr;
    l.r_gpu=nullptr;
    l.h_gpu=nullptr;
    l.m_gpu=nullptr;
    l.v_gpu=nullptr;
    l.prev_state_gpu=nullptr;
    l.forgot_state_gpu=nullptr;
    l.forgot_delta_gpu=nullptr;
    l.state_gpu=nullptr;
    l.state_delta_gpu=nullptr;
    l.gate_gpu=nullptr;
    l.gate_delta_gpu=nullptr;
    l.save_gpu=nullptr;
    l.save_delta_gpu=nullptr;
    l.concat_gpu=nullptr;
    l.concat_delta_gpu=nullptr;
    l.binary_input_gpu=nullptr;
    l.binary_weights_gpu=nullptr;
    l.mean_gpu=nullptr;
    l.variance_gpu=nullptr;
    l.rolling_mean_gpu=nullptr;
    l.rolling_variance_gpu=nullptr;
    l.variance_delta_gpu=nullptr;
    l.mean_delta_gpu=nullptr;
    l.x_gpu=nullptr;
    l.x_norm_gpu=nullptr;
    l.weights_gpu=nullptr;
    l.weight_updates_gpu=nullptr;
    l.biases_gpu=nullptr;
    l.bias_updates_gpu=nullptr;
    l.scales_gpu=nullptr;
    l.scale_updates_gpu=nullptr;
    l.output_gpu=nullptr;
    l.delta_gpu=nullptr;
    l.rand_gpu=nullptr;
    l.squared_gpu=nullptr;
    l.norms_gpu=nullptr;
#endif
}
void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
#ifdef GPU
        if(l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);

#ifdef GPU
    if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if(l.z_gpu)                   cuda_free(l.z_gpu);
    if(l.r_gpu)                   cuda_free(l.r_gpu);
    if(l.h_gpu)                   cuda_free(l.h_gpu);
    if(l.m_gpu)                   cuda_free(l.m_gpu);
    if(l.v_gpu)                   cuda_free(l.v_gpu);
    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if(l.state_gpu)               cuda_free(l.state_gpu);
    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if(l.gate_gpu)                cuda_free(l.gate_gpu);
    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if(l.save_gpu)                cuda_free(l.save_gpu);
    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if(l.concat_gpu)              cuda_free(l.concat_gpu);
    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if(l.mean_gpu)                cuda_free(l.mean_gpu);
    if(l.variance_gpu)            cuda_free(l.variance_gpu);
    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if(l.x_gpu)                   cuda_free(l.x_gpu);
    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if(l.weights_gpu)             cuda_free(l.weights_gpu);
    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if(l.biases_gpu)              cuda_free(l.biases_gpu);
    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if(l.scales_gpu)              cuda_free(l.scales_gpu);
    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if(l.output_gpu)              cuda_free(l.output_gpu);
    if(l.delta_gpu)               cuda_free(l.delta_gpu);
    if(l.rand_gpu)                cuda_free(l.rand_gpu);
    if(l.squared_gpu)             cuda_free(l.squared_gpu);
    if(l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
}
