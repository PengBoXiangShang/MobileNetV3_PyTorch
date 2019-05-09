
# MobileNetV3_PyTorch_pretrained_model

<div align=center><img src="https://github.com/PengBoXiangShang/net_pytorch/blob/master/figures/MobileNet_V3_block.png"/></div>

This is an unofficial PyTorch implementation for [MobileNetV3](https://arxiv.org/abs/1905.02244). Multi-GPUs training is supported. We trained it on ImageNet-1K and released the model parameters. This work was   implemented by **[Peng Xu](http://www.pengxu.net)** and **[Jin Feng](https://github.com/JinDouer)**.

This project is designed with these goals:
- [x] Train MobileNetV3-Small 1.0 on ImageNet-1K dataset.
- [ ] Train MobileNetV3-Large 1.0 on ImageNet-1K dataset.

-----
## Requirements

Ubuntu 14.04

Python 2.7

PyTorch 0.4.0

## Our Hardware Environment

Our server details:
2 Intel(R) Xeon(R) CPUs (E5-2620 v3 @ 2.40GHz), 128 GB RAM,
4 GTX 1080 Ti GPUs.

For fast IO, ImageNet-1K dataset is stored in our SSD.

## Experimental Results
We report the performance (Top-1 accuracy) on ImageNet-1K validation set.

| Network | Top-1 Accuracy | Pretrained Model|
| ------ | ------ | ------ |
|MobileNetV3-Small 1.0 (Official Implementation)|67.4%|none|
|MobileNetV3-Small 1.0 (Our Implementation)|todo|9.8 MB, [Baiduyun Disk](https://pan.baidu.com/s/1MNb0oTrFkcnw-GD3O2Ys1Q：), [Google Drive] ()|

## Our Training Details
Optimizer
Learning rate
Batch Size = 2560
Running costs
|GPU RAM|RAM|Running Time|
| ------ | ------ | ------ |
|x|x|x|
Please see more details in our training file train.py.


## How to Use
```
from MobileNetV2 import MobileNetV2

net = MobileNetV2(model_mode="SMALL", num_classes=1000)
state_dict = torch.load('xx')["network"]
net.load_state_dict(state_dict)

```


