#ifndef CALIBRATION
#define CALIBRATION

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include "darknet.h"
#include "blas.h"
#include "activations.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "yolo_layer.h"
#include "route_layer.h"
#include "im2col.h"

void validate_calibrate_valid(char *datacfg, char *cfgfile, char *weightfile, int calibrate_round);

#endif