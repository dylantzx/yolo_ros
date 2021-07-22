#!/usr/bin/env python
from __future__ import print_function

from FPS import *
from ImageConverter import *
from ObjectTracker import *

import os
# 0 for GPU, -1 for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from TensorFlow_Yolo.yolov3.utils import image_preprocess, postprocess_boxes, nms, Load_Yolo_model
from TensorFlow_Yolo.yolov3.configs import *

import sys
import rospy
import cv2

from yolo_ros.msg import Bbox_values

from tensorflow.python.client import device_lib
import numpy as np

def main(args):

    print(device_lib.list_local_devices())
    rospy.init_node('drone_detector')
    ic = ImageConverter()
    fps = FPS()
    ot = ObjectTracker()
    yolo=Load_Yolo_model()

    input_size = YOLO_INPUT_SIZE
    score_threshold=0.8
    iou_threshold=0.45

    while not rospy.is_shutdown():

        image_data = image_preprocess(np.copy(ic.cv_img), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        fps.start()
        batched_input = tf.constant(image_data)
        pred_bbox = yolo(batched_input)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, ic.cv_img, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        boxes, scores, names = [], [], []
        for bbox in bboxes:
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
            scores.append(bbox[4])
            names.append('target')

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)

        ot.track_object(ic.cv_img, boxes, names, scores, fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)