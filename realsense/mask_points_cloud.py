import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# class_names = ['person']

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# rs.config.enable_device_from_file(config, args.input)
rs.config.enable_device_from_file(config, '/home/xipeng/Documents/20190304_011637.bag')
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
# Start streaming
profile = pipeline.start(config)

# Read the bag frame by frame, not by real time !!!!!!!!
device = profile.get_device()
playback = rs.playback(device)
playback.set_real_time(False)

# videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 1, (1280, 720))
ii = 0


try:

    while True:
        ii += 1

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if ii != 705:
            continue

        # Points Cloud
        pc = rs.pointcloud()
        points = rs.points
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asarray(points.get_vertices())
        points.export_to_ply("1.ply", color_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        print(np.shape(color_frame))
        print(color_frame.dtype)
        print(color_frame)

        # Run detection
        results = model.detect([color_image], verbose=1)

        # Visualize results
        r = results[0]

        print(ii)

        print(r['class_ids'])

        while True:

            premask = np.zeros((np.shape(color_image)[0], np.shape(color_image)[1]), dtype=np.uint8)

            for i in range(len(r['class_ids'])):
                if r['class_ids'][i] != 1:
                    continue
                for j in range(np.shape(color_image)[0]):
                    for k in range(np.shape(color_image)[1]):
                        if r['masks'][j][k][i]:
                            premask[j][k] = 1

            for j in range(np.shape(color_image)[0]):
                for k in range(np.shape(color_image)[1]):
                    for i in range(3):
                        color_image[j][k][i] *= premask[j][k]

            a = color_image[:, :, ::-1].astype(np.uint8)

            cv2.namedWindow('RealSense', a)
            cv2.imshow('RealSense', a)

            # # Points Cloud
            # pc = rs.pointcloud()
            # points = rs.points
            # pc.map_to(color_frame)
            # points = pc.calculate(depth_frame)
            # vtx = np.asarray(points.get_vertices())
            # points.export_to_ply("1.ply", color_frame)




            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.imwrite('color.jpg', a)
                cv2.destroyAllWindows()
                break


        break
        # a = a[:,:,::-1].astype(np.uint8)



finally:
    # Stop streaming
    pipeline.stop()
