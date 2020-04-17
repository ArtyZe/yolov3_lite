GPU := 0
CUDNN := 0
OPENCV := 0
OPENMP := 1
DEBUG := 0
PRUNE :=0
MASK :=0
AVX :=1
#as far can not use QUANTIZATION_GOOGLE and QUANTIZATION at same time
QUANTIZATION_GOOGLE :=1
QUANTIZATION :=0

ARCH :=  -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH := ./src/:./examples:./gemm_lowp/eight_bit_int_gemm
SLIB := libdarknet.so
ALIB := libdarknet.a
EXEC := darknet
OBJDIR := ./obj/

CC := g++
NVCC := nvcc 
AR := ar
ARFLAGS := rcs
OPTS := -Ofast
LDFLAGS := -lm -lpthread -ldl
COMMON := -Iinclude/ -Isrc/ -Igemm_lowp/eight_bit_int_gemm/
CFLAGS := -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC --std=c++11

ifeq ($(OPENMP), 1) 
CFLAGS += -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS += $(OPTS)

ifeq ($(OPENCV), 1) 
COMMON += -DOPENCV
CFLAGS += -DOPENCV
LDFLAGS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
COMMON += `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON += -DGPU -I/usr/local/cuda/include/
CFLAGS += -DGPU
LDFLAGS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON += -DCUDNN 
CFLAGS += -DCUDNN
LDFLAGS += -lcudnn
endif

ifeq ($(AVX), 1) 
CFLAGS+= -ffp-contract=fast -msse3 -msse4.1 -msse4.2 -msse4a -mavx -mavx2 -mfma -DAVX
endif

ifeq ($(PRUNE), 1) 
COMMON+= -DPRUNE
CFLAGS+= -DPRUNE
endif

ifeq ($(QUANTIZATION_GOOGLE), 1) 
COMMON+= -DQUANTIZATION_GOOGLE
CFLAGS+= -DQUANTIZATION_GOOGLE
endif

ifeq ($(QUANTIZATION), 1) 
COMMON+= -DQUANTIZATION
CFLAGS+= -DQUANTIZATION
endif

ifeq ($(SCALE_L1), 1) 
COMMON+= -DSCALE_L1
CFLAGS+= -DSCALE_L1
endif

ifeq ($(MASK), 1) 
COMMON+= -DMASK
CFLAGS+= -DMASK
endif

ifeq ($(LAYER_MASK), 1) 
COMMON+= -DMASK
CFLAGS+= -DMASK
endif

ifeq ($(MULTI_CORE), 1) 
CFLAGS+= -fopenmp
LDFLAGS+= -lgomp
endif
OBJ :=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o calibration.o eight_bit_int_gemm.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  yolo_layer.o image_opencv.o list.o
EXECOBJA :=segmenter.o detector.o darknet.o
ifeq ($(GPU), 1) 
# LDFLAGS+= -lstdc++ 
OBJ += convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ  :=  $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS  :=  $(addprefix $(OBJDIR), $(OBJ))
DEPS  :=  $(wildcard src/*.h) Makefile include/darknet.h

# CFLAGS += -std=c++11
#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@ -std=c++11

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)

