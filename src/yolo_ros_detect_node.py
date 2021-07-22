#!/usr/bin/env python
from __future__ import print_function

from FPS import *
from ImageConverter import *

import os
# 0 for GPU, -1 for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from TensorFlow_Yolo.yolov3.utils import image_preprocess, postprocess_boxes, nms, draw_bbox, Load_Yolo_model, Create_Yolo
from TensorFlow_Yolo.yolov3.configs import *

import sys
import rospy
import cv2

from yolo_ros.msg import Bbox_values

from tensorflow.python.client import device_lib
import numpy as np
import time

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  fps = FPS()
  yolo=Load_Yolo_model()

  input_size = YOLO_INPUT_SIZE
  score_threshold=0.3
  iou_threshold=0.45

  while not rospy.is_shutdown():

    image_data = image_preprocess(np.copy(ic.cv_img), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # t1 and t2 used to calculate the time taken to predict a bbox
    fps.start()
    batched_input = tf.constant(image_data)
    pred_bbox = yolo(batched_input)
    fps.stop()

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, ic.cv_img, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    
    # get bbox values that are within the thresholds
    # if prediction not within thresholds, bbox list will be empty
    # print(f"{len(bboxes) != 0}")
    if len(bboxes) !=0:
      bbox_pub = Bbox_values()
      for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        x1, y1, x2, y2 = coor[0], coor[1], coor[2], coor[3]
        w = x2 - x1
        h = y2 - y1
        score = bbox[4]
      # print(f"Bbox values: {x1, y1, w, h} Score: {round(score,2)}")
      bbox_pub.x = x1.item()
      bbox_pub.y = y1.item()
      bbox_pub.w = w.item()
      bbox_pub.h = h.item()
      ic.image_pub.publish(bbox_pub)

    frame = draw_bbox(ic.cv_img, bboxes, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    cv2.putText(frame, f"FPS: {fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)