# MaskFlownet-Pytorch
Unofficial PyTorch implementation of MaskFlownet (https://github.com/microsoft/MaskFlownet).

Tested with:
* PyTorch 1.5.0
* CUDA 10.1

### Install
The correlation package must be installed first:
```
cd model/correlation_package
python setup.py install
```

### Inference
Right now, I implemented the inference script only for KITTI 2012/2015.

```
python predict.py CONFIG -c CHECKPOINT --dataset_cfg DATASET [-b BATCH_SIZE]
```

For example:
* ``` python predict.py MaskFlownet.yaml --c 5adNov03-0005_1000000.pth --dataset_cfg kitti.yaml -b 4```
* ``` python predict.py MaskFlownet_S.yaml --c 771Sep25-0735_500000.pth --dataset_cfg kitti.yaml -b 4 ```
* ``` python predict.py MaskFlownet_S.yaml --c dbbSep30-1206_1000000.pth --dataset_cfg kitti.yaml -b 4 ```

### Difference with the original implementation
The results are slightly different from the original implementation:

| Checkpoint | Network | Original Implementation AEPE KITTI2012 / KITTI2015 | This Implementation AEPE KITTI2012 / KITTI2015 |
| --- | --- | --- | ---|
| 771Sep25 | MaskFlownet_S | 4.12 / 11.52 | 4.18 / 11.83 |
| dbbSep30 | MaskFlownet_S | 1.27 / 1.92 | 1.28 / 1.93 |
| 5adNov03 | MaskFlownet   | 1.16 / 1.66 | 1.21 / 1.74 |
 
#### Example
Original implementation:
![original_visualization](./data/original-implementation.png)

This implementation:
![this_visualization](./data/this-implementation.png)

### Acknowledgement
[correlation_package](model/correlation_package) was taken from [flownet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
