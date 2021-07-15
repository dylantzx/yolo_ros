#!/usr/bin/env python
from __future__ import print_function

import os
# 0 for GPU, -1 for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from TensorFlow_Yolo.yolov3.utils import image_preprocess, postprocess_boxes, nms, draw_bbox, Load_Yolo_model, Create_Yolo
from TensorFlow_Yolo.yolov3.configs import *

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from yolo_ros.msg import Bbox_values

from tensorflow.python.client import device_lib
import numpy as np
import time


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("bbox_output",Bbox_values, queue_size=10)
    self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
    self.cv_img = None

  def callback(self,data):
        
    width = data.width
    height = data.height
    channels = int(len(data.data) / (width * height))

    encoding = None
    if data.encoding.lower() in ['rgb8', 'bgr8']:
        encoding = np.uint8
    elif data.encoding.lower() == 'mono8':
        encoding = np.uint8
    elif data.encoding.lower() == '32fc1':
        encoding = np.float32
        channels = 1

    # Have to use a copy as the original image is read-only which will result in an error when
    # trying to modify the image
    self.cv_img = np.ndarray(shape=(data.height, data.width, channels), dtype=encoding, buffer=data.data).copy()

    if data.encoding.lower() == 'mono8':
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2GRAY)
    else:
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2BGR)

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = image_converter()
  yolo=Load_Yolo_model()

  input_size = YOLO_INPUT_SIZE
  score_threshold=0.8
  iou_threshold=0.1

  times = []

  while not rospy.is_shutdown():

    image_data = image_preprocess(np.copy(ic.cv_img), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # t1 and t2 used to calculate the time taken to predict a bbox
    t1 = time.time()
    batched_input = tf.constant(image_data)
    pred_bbox = yolo(batched_input)
    t2 = time.time()

    # Store the time taken to predict and get the average of the last 20 predictions
    times.append(t2-t1)
    times = times[-20:]
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms

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
    cv2.putText(frame, f"FPS: {fps:.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)