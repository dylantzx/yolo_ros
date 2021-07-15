#!/usr/bin/env python
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from TensorFlow_Yolo.yolov3.utils import image_preprocess, postprocess_boxes, nms, draw_bbox, Load_Yolo_model, detect_image
from TensorFlow_Yolo.yolov3.configs import *

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

from tensorflow.python.client import device_lib
import numpy as np
import time


class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("bbox_output",Bbox_values, queue_size=10)
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

class FPS:

  def __init__(self):
    self.prev_frame_time = 0
    self.curr_frame_time = 0
    self.sum_of_fps = 0
    self.count = 0
    self.fps =0

  def calculateFPS(self):
    self.curr_frame_time = time.time()
    self.fps = 1/ (self.curr_frame_time - self.prev_frame_time)
    self.count+=1
    self.sum_of_fps += self.fps
    self.prev_frame_time = self.curr_frame_time

  def getAvgFPS(self, img):
    avg_fps= round(self.sum_of_fps/self.count,2)
    # print(f"-----------------Avg FPS: {round(self.sum_of_fps/self.count,2)}-----------------")
    cv2.putText(img, f"FPS: {avg_fps}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = image_converter()
  yolo=Load_Yolo_model()
  input_size = YOLO_INPUT_SIZE
  score_threshold=0.3
  iou_threshold=0.45
  count = 0
  fps = FPS()

  while not rospy.is_shutdown():
    
    image_data = image_preprocess(np.copy(ic.cv_img), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
          pred_bbox = yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, ic.cv_img, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    # get bbox values
    for i, bbox in enumerate(bboxes):
      coor = np.array(bbox[:4], dtype=np.int32)
      x1, y1, x2, y2 = coor[0], coor[1], coor[2], coor[3]
      w = x2 - x1
      h = y2 - y1
      score = bbox[4]
    print(f"Bbox values: {x1, y1, w, h} Score: {round(score,2)}")

    frame = draw_bbox(ic.cv_img, bboxes, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

    # Calculate FPS
    fps.calculateFPS()
    fps.getAvgFPS(frame)

    # # publish bbox values when they are available
    # # bbox values are in y1,x1,y2,x2
    # # have to reformat to x,y,w,h
    # if len(r['rois']):
    #   bbox_str = np.array_str(r['rois'][0])
    #   bbox_ls = bbox_str[1:-1].strip().replace("   ", " ").replace("  ", " ").split(" ")
    #   bbox = Bbox_values()
    #   bbox.x = int(bbox_ls[1])
    #   bbox.y = int(bbox_ls[0])
    #   bbox.w = int(bbox_ls[3]) - int(bbox_ls[1])
    #   bbox.h = int(bbox_ls[2]) - int(bbox_ls[0])
    #   ic.image_pub.publish(bbox)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)