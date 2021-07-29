# yolo_ros

## Prerequisites
---
- Ubuntu LTS 18.04

- ROS Melodic 

- [Setup PX4, QGC and MavROS](https://github.com/dylantzx/PX4)

- JupyterLab

- Anaconda3

- Python 3.7+

- TensorFlow-gpu 2.5.0

- OpenCV-python 4.1.2.30

- Numpy, scipy, wget, tqdm, seaborn, Pillow, pandas, awscli, urllib3, mss

- NVIDIA Drivers 

- CUDA 11.3

- CuDNN 

## About
---

This is a ROS package of [YOLO with Tensorflow 2.x](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) with [DeepSORT](https://github.com/nwojke/deep_sort) for object detection and object tracking.

It contains ROS nodes for object detection and object detection with tracking.

The current repository is for a drone tracking another drone on PX4 but you should be able to adapt it for your own use case.

## Installation Guide
---

### NVIDIA Drivers, CUDA and CuDNN 

1. [Guide for NVIDIA Drivers and CuDNN installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux)

2. [Guide for CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### Anaconda3, JupyterLab and Conda environment

1. [Anaconda3 Installation](https://www.anaconda.com/products/individual) 

2. [JupyterLab Installation](https://jupyter.org/)

3. [Conda environment guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### The rest of the prerequisites ###
Clone the repository then change to the `TensorFlow_Yolo` directory in terminal and use `pip install -r requirements.txt` 
```
cd ~/catkin_ws/src
git clone https://github.com/dylantzx/yolo_ros.git --recursive
cd yolo_ros/src/TensorFlow_Yolo
conda activate <your_env>
pip install -r requirements.txt
```

You can also install them individually with `pip install <requirement>` in your virtual environment 

## Getting Started
---

[FPS.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/FPS.py) - Contains a simple FPS class for FPS calculation 

[ImageConverter.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/ImageConverter.py) - Contains ImageConverter class that converts images received via GazeboROS Plugin of `sensor_msgs` type to a usable type for object detection. 

[ObjectTracker.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/ObjectTracker.py) - Contains class to utilize DeepSORT for object tracking.

[yolo_ros_detect_node.py](https://github.com/dylantzx/yolo_ros/blob/main/src/yolo_ros_detect_node.py) - Main script that runs YOLO and TensorFlow 2.x with ROS for object detection

[yolo_ros_track_node.py](https://github.com/dylantzx/yolo_ros/blob/main/src/yolo_ros_track_node.py) - Main script that runs YOLO and TensorFlow 2.x with ROS for object detection with tracking.

## How to run codes
---
1. Go into your virtual environment

    ```conda activate <your_env>```

2. To run object detection, run the launch file
    
    ```roslaunch yolo_ros yolo_detect.launch```

<!-- ![Object detection only](images/maskRCNN_detect.png) -->

3. To run object detection with tracking, run the launch file
    
    ```roslaunch yolo_ros yolo_track.launch```

<!-- ![Object detection with tracking](images/maskRCNN_track.png) -->

## Training YOLO 
---

To train the YOLO model on your own custom dataset, you can refer to [this repository](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3).

Place your images for training, validation and testing under `TensorFlow_Yolo/IMAGES/`

After getting your checkpoint files from training, place them under the `TensorFlow_Yolo/checkpoints` directory.

## Modifying to use it on your own dataset
---

1. Go to `TensorFlow_Yolo/yolov3/configs.py`.
2. Ensure that the configs are suited to your use case. The important ones are:
    ```
    YOLO_TYPE (line 13)
    YOLO_CUSTOM_WEIGHTS (line 20)
    TRAIN_CLASSES (line 40)
    TRAIN_ANNOT_PATH (line 41)
    TRAIN_MODEL_NAME (line 44)
    TEST_ANNOT_PATH (line 57)
    ```
3. Go into the script that you will be using, for example `yolo_ros_detect.py` and change lines 32 -34 if required
4. Go to the launch file that you will be using
5. Remap the `image_topic` to your own ROStopic that is publishing the images

## Evaluation
---
The evaluation script is found in the `TensorFlow_Yolo/evaluation/main.py`.

Change the path and names on lines 34 - 36 into your own.

In your terminal, change directory into `TensorFlow_Yolo/evaluation` and run `python main.py`

Upon successful execution, you should get an excel spreadsheet under `TensorFlow_Yolo/exports/` with your evaluation results.

