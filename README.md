# 3D Object Detection and Attribute Prediction

## Authors:
Yuqing Qin, Feng(Jason) Xiang, Jingxiang Lin

## Introduction

This project was created for the development and grading of the 16-824: Visual Learning and Recognition course at Carnegie Mellon University. The following members contributed to the development of this semester project:

* Yuqing Qin
* Jingxiang Lin
* Feng Xiang

## Installation

### TODO: create installation steps

## How to Run

To create the dataset for mmdetection to create the dataloader on, make sure to download the NuScenes dataset from the NuScenes website:

[NuScenes Website](https://www.nuscenes.org)

Untar the dataset files into the data/nuscenes directory.

To create the dataset files, run the following command from the mmdetection3d directory:

```python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes```

If the dataset is the NuScenes Mini dataset, create the dataset by running the following command from the mmdetection3d directory:

'python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version='v1.0-mini'

To train the Centerpoint model (based on the Point Pillar backbone), run the following command from the mmdetection3d directory:

'python tools/train.py configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py'

To train the Centerpoint model with DCN and using pretrained weights, run the following command from the mmdetection3d directory:

'python tools/train.py configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py --load-from ./demo/centerpoint_pointpillars_dcn.pth'

To test:

'python tools/test.py configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4/latest.pth --eval mAP'


## References

This custom repository was based on the *mmdetection3d* repository.

[mmdetection3d Github Repository](https://github.com/open-mmlab/mmdetection3d)

The creaters' documentation was referenced during the development of this project.

[mmdetection3d ReadTheDocs Website](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html)
