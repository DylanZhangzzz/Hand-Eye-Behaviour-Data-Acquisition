import cv2
import os



def video_gen():

    fps = 30  # 视频每秒30帧
    size = (1920, 1080)
    video = cv2.VideoWriter("./OriginVideo.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    total_frame = os.listdir('./skeleton_img')
    print(total_frame)
    for frame_num in total_frame:

        img_path = F'./skeleton_img/{frame_num}'  # 图片路径

        read_img = cv2.imread(img_path)
        video.write(read_img)
    video.release()

video_gen()