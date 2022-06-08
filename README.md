# FEATURIZED Query R-CNN

Paper Link: [Featurized Query R-CNN]()

## Introduction

Recently, query-based deep neural networks have achieved great object detection performance. However, these methods suffer from two issues about the queries. Firstly, object queries are learnable parameters that require multi-stage decoders to optimize, incurring a large computation burden. Secondly, the queries are fixed after training, leading to unsatisfying generalization capability. To remedy the above issues, we present featurized object queries predicted by a query generation network for better object detection. We integrate the featurized queries into the mainstream two-stage detection frameworks, e.g., Faster R-CNN, and develop a Featurized Query R-CNN. Extensive experiments on the COCO dataset show that our Featurized Query R-CNN obtains the best speed-accuracy trade-off among all R-CNN detectors, including the recent state-of-the-art Sparse R-CNN detector.

![1654667943617](C:\Users\WenqiangZhang\AppData\Roaming\Typora\typora-user-images\1654667943617.png)

## Installation and Training

Our methods are based on [detectron2](), please refer to [here]() for more details.

Install the detectron2:

```
git clone https://github.com/facebookresearch/detectron2.git

python setup.py build develop
```

For training, run:

```
python train_net.py --config-file <config-file> --num-gpus <num-gpus>
```

## Main Results

|                                          | Backbone   | Epoch | AP   | FPS  | Weights |
| ---------------------------------------- | ---------- | ----- | ---- | ---- | ------- |
| FEATURIZED QR-CNN(100 Queries)           | ResNet-50  | 36    | 41.3 | 26   |         |
| FEATURIZED QR-CNN(2 Stages, 100 Queries) | ResNet-50  | 36    | 43.0 | 24   |         |
| FEATURIZED QR-CNN(2 Stages, 300 Queries) | ResNet-50  | 36    | 44.6 | 24   |         |
| FEATURIZED QR-CNN(2 Stages, 100 Queries) | ResNet-101 | 36    | 43.9 | 18   |         |
| FEATURIZED QR-CNN(2 Stages, 300 Queries) | ResNet-101 | 36    | 45.8 | 18   |         |

- The speed is tested on a single RTX 2080Ti GPU on COCO val set.

## Acknowledgements

Our implementation is based on [detectron2](https://github.com/facebookresearch/detectron2) and [Sparse R-CNN](), we thank for their open-source code.

