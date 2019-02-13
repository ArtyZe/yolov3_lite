# compress_yolo 对yolo模型进行网络压缩、修剪等实验

 1.git clone https://github.com/hjimce/compress_yolo
 
 2.```cd compress_yolo```
 
 3.vim Makefile and  set``` PRUNE=1  MASK=1```
 
 4.start prune  tiny-yolo:
 
 ```./darknet detector train cfg/coco.data cfg/tiny-yolo.cfg  pretrain/tiny-yolo.weights  -gpus 0```
 
 5、copy backup file trained weights and test:
 
 ``` ./darknet detector test cfg/coco.data cfg/tiny-yolo-test.cfg pretrain/tiny-yolo_prune.weights Original.png```

 6、test mAP:

 ```./darknet detector valid cfg/coco.data cfg/tiny-yolo-test.cfg pretrain/tiny-yolo_prune.weights```
 
 There are some visulization Marcos in my code, if you want to see the prune result, just read my code and open them
