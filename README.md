# Surgical OmniMotion

This repository contains the code for training neural fields for 3D tracks in surgical videos, as described in the publication "Neural Fields for 3D Tracking of Anatomy and Surgical Instruments in Monocular Laparoscopic Video Clips".

## ⚙️ Installation
You can use a standard environment with python and pytorch. Install additional packages with:
```
pip3 install matplotlib tensorboard scipy opencv-python-headless tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## 📄 Datasets

### SCARED Dataset
This dataset is available at [grand-challenge](https://endovissub2019-scared.grand-challenge.org/). The data consists of 9 static surgical scenes captured from several angles.

The dataset itself does not come with depth maps. Luckily, these can be generated with the help of [this repository](https://github.com/dimitrisPs/scared_toolkit). When the dataset is downloaded, you can generate depth maps, rectified images and undistored images with the command:
```
python3 -m scripts.extract_sequence_dataset /data/SCARED --recursive --depth --undistort --disparity
```
For using this package, the additional installation of ```plyfile``` is necessary.

### CholecSeg8k Dataset
This dataset is available at [Kaggle](https://www.kaggle.com/datasets/newslab/cholecseg8k). The data includes 80-frame video clips with segmentation masks of tissues and instruments.

## 🚀 Usage

### Pre-processing
Before training OmniMotion on a video clip, the data needs to be pre-processed. For example, when using the SCARED dataset, process a video by renaming the folder ```data/left``` to ```data/color``` such that OmniMotion can find it. Then, start the following script:
```
cd preprocessing
python3 main_processing --data_dir /data/SCARED/dataset_{number}/keyframe_{number}/data
```

### Training
After pre-processing, start the training with:
```
python3 train.py --config configs/{config_file}.txt --data_dir /data/SCARED/dataset_{number}/keyframe_{number}/data
```

#### Common issues
A common issue (```ValueError: ImageIO does not generally support reading folders.```) is that Jupyer notebooks create ```.ipynb_checkpoints``` files. You can remove these files recursively with:
```
rm -rf `find -type d -name .ipynb_checkpoints`
```

### Visualization
To create visualizations you can run:
```
python3 viz.py --config configs/{config_file}.txt --data_dir /data/SCARED/dataset_{number}/keyframe_{number}/data
```

When you have a foreground-mask, add ```--foreground_mask_path {path_to_file}.png``` to the command. This file should be a RGBA file, where the alpha values of background are zeros.

## ⭐ Acknowledgements
- This repository was cloned and adapted from [the original OmniMotion implementation](https://github.com/qianqianwang68/omnimotion). We thank the authors for publishing their work.
