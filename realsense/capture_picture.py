# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
# rs.config.enable_device_from_file(config, args.input)
rs.config.enable_device_from_file(config, '/home/xipeng/Documents/20190304_011637.bag')
# Configure the pipeline to stream the depth stream, recorded file have to have the same format
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# Start streaming from file
profile  = pipeline.start(config)

frame_namber = 0
last_frame = 0

# Read the bag frame by frame, not by real time !!!!!!!!
device = profile.get_device()
playback = rs.playback(device)
playback.set_real_time(False)


# depth_vedio = np.array()
# color_vedio = np.array()


# Streaming loop
try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        # frames.keep()

        frame_namber +=1
        if last_frame > frames.get_timestamp():
            print(frame_namber)
            break
        else:
            last_frame = frames.get_timestamp()


        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = rs.colorizer().colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Wait for a coherent pair of frames: depth and color
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_vedio = np.r_(depth_vedio, depth_image)
        # color_vedio = np.r_(color_vedio, color_image)

        if frame_namber == 705:

            # Convert image from RGB to BGR
            color_image = color_image[:, :, ::-1]
            cv2.imwrite('demo_presentation.jpg', color_image)

            break



finally:
    # Stop streaming
    pipeline.stop()