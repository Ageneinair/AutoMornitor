import cv2

def videocut(start, end, output_fps, input_address, output_address = '../cuted_video.mp4'):

    vidcap = cv2.VideoCapture(input_address)

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv2.VideoWriter(output_address, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), output_fps, (width,height))

    count = 0
    success, frame = vidcap.read()

    print (fps/output_fps)
    while success:

        if (count % int(fps / output_fps) == 0) & (count >= start*fps) & (count <= end* fps):
            videoWriter.write(frame)
            print('Count',count)

        count += 1
        success, frame = vidcap.read()

    vidcap.release()
    videoWriter.release()
    print('Video cutting finished!')




