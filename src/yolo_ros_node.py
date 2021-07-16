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

from TensorFlow_Yolo.deep_sort import nn_matching
from TensorFlow_Yolo.deep_sort.detection import Detection
from TensorFlow_Yolo.deep_sort.tracker import Tracker
from TensorFlow_Yolo.deep_sort import generate_detections as gdet

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
    iou_threshold=0.45

    Track_only = ['target']
    NUM_CLASS = {0: 'target'}
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = '/home/dylan/catkin_ws/src/yolo_ros/src/TensorFlow_Yolo/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

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
        
        print(f"{bboxes}")

        if len(bboxes) !=0:
            boxes, scores, names = [], [], []
            for bbox in bboxes:
                if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                    boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                    scores.append(bbox[4])
                    names.append(NUM_CLASS[int(bbox[5])])

            print(f"{boxes}, {scores}, {names}")

            # Obtain all the detections for the given frame.
            boxes = np.array(boxes) 
            names = np.array(names)
            scores = np.array(scores)
            features = np.array(encoder(ic.cv_img, boxes))
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

            # Pass detections to the deepsort object and obtain the track information.
            tracker.predict()
            tracker.update(detections)

            # Obtain info from the tracks
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue 
                bbox = track.to_tlbr() # Get the corrected/predicted bounding box
                class_name = track.get_class() #Get the class name of particular object
                tracking_id = track.track_id # Get the ID for the particular track
                index = key_list[val_list.index(class_name)] # Get predicted object index by object name
                tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

            frame = draw_bbox(ic.cv_img, tracked_bboxes, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), tracking=True)
            
            t3 = time.time()
            times_2.append(t3-t1)
            times_2 = times_2[-20:]
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)