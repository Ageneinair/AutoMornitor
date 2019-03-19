import dlib
import cv2
import pyrealsense2 as rs
import numpy as np


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    detector = dlib.get_frontal_face_detector()
    color_green = (0, 255, 0)
    line_width = 3
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            rgb_image = np.asanyarray(color_frame.get_data())
            # print(rgb_image)
            # while True :
            #     pass
            # 检测
            dets = detector(rgb_image)
            for det in dets:
                cv2.rectangle(rgb_image, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
            cv2.imshow('my webcam', rgb_image)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
