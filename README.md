# yolo_prune

1. set weights &lt; 0.01 to 0 (set PRUNE = 1 in makefile); 

2. pruning output feature maps according to BN scales (set SCALE_L1 = 1 in makefile)

3. conv connection sparsification by using a mask of 0 and 1 (set MASK = 1 in makefile)

4. parallel computation (set MULTI_CORE = 1 in makefile)

The first function is based on the work of https://github.com/hjimce/compress_yolo

The second function is pruning feature maps according to BN scales parameter,

and the regularization method is L1 NORM

test result is:
YOLO2
	Before   Runtime: 15s, mAP: 0.62
	After    Runtime: 2s,  mAP: 0.57

If you want to use my code, please let me know
