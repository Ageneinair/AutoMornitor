# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2


output_fps = 30
input_fps = 30

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
# rs.config.enable_device_from_file(config, args.input)
rs.config.enable_device_from_file(config, '/home/xipeng/Documents/20190304_011637.bag')
# Configure the pipeline to stream the depth stream, recorded file have to have the same format
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, input_fps)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, input_fps)

# Start streaming from file
profile  = pipeline.start(config)

frame_namber = 0
last_frame = 0

# Read the bag frame by frame, not by real time !!!!!!!!
device = profile.get_device()
playback = rs.playback(device)
playback.set_real_time(False)

# Create alignment primitive with depth as its target stream:
align_to = rs.stream.color
align = rs.align(align_to)



videoWriter = cv2.VideoWriter('xinyun.mp4', cv2.VideoWriter_fourcc(*'MJPG'), output_fps, (1280*2, 720))

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
            videoWriter.release()
            # np.save("depth_vedio.npy", depth_vedio)
            # np.save("color_vedio.npy", color_vedio)
            print(frame_namber)
            break
        else:
            last_frame = frames.get_timestamp()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Get depth frame
        # depth_frame = frames.get_depth_frame()

        # # Colorize depth frame to jet colormap
        # depth_color_frame = rs.colorizer().colorize(depth_frame)
        #
        # # Convert depth_frame to numpy array to render image in opencv
        # depth_color_image = np.asanyarray(depth_color_frame.get_data())
        #
        # # Wait for a coherent pair of frames: depth and color
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()


        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print(color_image.dtype)
        # print(np.shape(color_image))
        # print(color_image)

        # depth_vedio = np.r_(depth_vedio, depth_image)
        # color_vedio = np.r_(color_vedio, color_image)

        if frame_namber % (input_fps / output_fps) == 0:

            # Convert image from RGB to BGR
            color_image = color_image[:, :, ::-1]

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            cv2.putText(images, str(frame_namber), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

            # Save to a video
            videoWriter.write(images)

finally:
    # Stop streaming
    pipeline.stop()