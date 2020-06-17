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
Right now, I implemented the inference script for KITTI 2012/2015, MPI Sintel and FlyingChairs.

```
python predict.py CONFIG -c CHECKPOINT --dataset_cfg DATASET -f ROOT_FOLDER [-b BATCH_SIZE]
```

For example:
* ``` python predict.py MaskFlownet.yaml -c 5adNov03-0005_1000000.pth --dataset_cfg sintel.yaml -f ./SINTEL -b 4```
* ``` python predict.py MaskFlownet.yaml -c 8caNov12-1532_300000.pth --dataset_cfg kitti.yaml -f ./KITTI -b 4```
* ``` python predict.py MaskFlownet_S.yaml -c 771Sep25-0735_500000.pth --dataset_cfg chairs.yaml -f ./FLYINGCHAIRS -b 4 ```
* ``` python predict.py MaskFlownet_S.yaml -c dbbSep30-1206_1000000.pth --dataset_cfg sintel.yaml -f ./SINTEL -b 4 ```

### Differences with the original implementation
The results are slightly different from the original implementation:

| Checkpoint | Network | Implementation | KITTI2012 | KITTI2015 | Sintel Clean | Sintel Final | FlyingChairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 771Sep25 | MaskFlownet_S | <p>Original AEPE: <br> PyTorch AEPE:</p> | <p>4.12<br>4.18</p> | <p>11.52<br>11.82</p> | <p>3.38<br>3.38</p> | <p>4.71<br>4.70</p> | <p>1.84<br>1.83</p> |
| dbbSep30 | MaskFlownet_S | <p>Original AEPE: <br> PyTorch AEPE:</p> | <p>1.27<br>1.28</p> | <p>1.92<br>1.93</p> | <p>2.76<br>2.78</p> | <p>3.29<br>3.32</p> | <p>2.36<br>2.36</p> |
| 5adNov03 | MaskFlownet   | <p>Original AEPE: <br> PyTorch AEPE:</p> | <p>1.16<br>1.18</p> | <p>1.66<br>1.68</p> | <p>2.58<br>2.59</p> | <p>3.14<br>3.17</p> | <p>2.23<br>2.23</p> |
| 8caNov12 | MaskFlownet   | <p>Original AEPE: <br> PyTorch AEPE:</p> | <p>0.82<br>0.82</p> | <p>1.38<br>1.38</p> | <p>4.34<br>4.40</p> | <p>5.27<br>5.33</p> | <p>4.01<br>3.99</p> |
 
#### Examples

KITTI Original implementation:

![original_visualization](./data/original-implementation.png)

KITTI This implementation:

![this_visualization](./data/this-implementation.png)

Sintel Original implementation:

![original_visualization](./data/original-sintel.png)

Sintel This implementation:

![this_visualization](./data/this-sintel.png)

FlyingChairs Original implementation:

![original_visualization](./data/original-chairs.png)

FlyingChairs This implementation:

![this_visualization](./data/this-chairs.png)

### Acknowledgment
Original MXNet implementation: [here](https://github.com/microsoft/MaskFlownet)

[correlation_package](model/correlation_package) was taken from [flownet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
