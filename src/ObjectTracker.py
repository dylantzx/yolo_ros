import numpy as np
import cv2

from TensorFlow_Yolo.yolov3.utils import draw_bbox
from TensorFlow_Yolo.yolov3.configs import *

from TensorFlow_Yolo.deep_sort import nn_matching
from TensorFlow_Yolo.deep_sort.detection import Detection
from TensorFlow_Yolo.deep_sort.tracker import Tracker
from TensorFlow_Yolo.deep_sort import generate_detections as gdet

class ObjectTracker:

    def __init__(self):
        self.rectangle_colors=(255,0,0)

        self.NUM_CLASS = {0: 'target'}
        self.key_list = list(self.NUM_CLASS.keys()) 
        self.val_list = list(self.NUM_CLASS.values())

        self.max_cosine_distance = 0.7
        self.nn_budget = None

        self.model_filename = '/home/dylan/catkin_ws/src/yolo_ros/src/TensorFlow_Yolo/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

    def track_object(self, img, boxes, names, scores, fps):
        features = np.array(self.encoder(img, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        self.tracker.predict()
        self.tracker.update(detections)
        fps.stop()
        self.show_tracked_object(img, fps)

    def show_tracked_object(self, img, fps):
        # Obtain info from the tracks
        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = self.key_list[self.val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        frame = draw_bbox(img, tracked_bboxes, CLASSES=TRAIN_CLASSES, rectangle_colors=self.rectangle_colors, tracking=True)

        cv2.putText(frame, f"FPS: {fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Prediction", frame)