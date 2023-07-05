import index_finger_tip_module as htm
import cv2
# from utils import *
import numpy as np
import os


def image_loader(root, j):
    # print(F'{root}/hand/rgb/image_{j}.png')
    # rgb = cv2.flip(cv2.imread(F'{root}/hand/rgb/image_{j}.png'),0)
    img =cv2.imread(F'{root}/hand/rgb/image_{j}.png')

    tmp = img.copy()

    src = np.float32([[70, 13], [503, 11], [2, 432], [570, 428]])
    dst = np.float32([[0, 0], [1080, 0], [0, 1080], [1080, 1080]])
    m = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(tmp, m, (1080, 1080))

    return result


def skeleton_gen(root, index, handness='right'):
    pixel_righthand, pixel_lefthand = [], []
    pixel_rightindex, pixel_leftindex = [], []
    process = htm.handDetector()
    count = 0
    for j in range(1, index.shape[0]):
        rgb = image_loader(root, int(index[j]))
        #print(depth[120, 240])
        process.findHands(rgb)
        all_pixel_r,  all_pixel_l, index_pixel_r, index_pixel_l = process.findPosition(rgb)

        # print(np.asarray(pixel_l).shape)

        if handness == 'right':
            # print(len(pixel_r))
            if len(index_pixel_r) == 1:
                # print('lm_right', lm_right)
                pixel_rightindex.append(np.asarray(index_pixel_r))
                # print('lm_right', pixel_r)
            else:
                pixel_rightindex.append(np.zeros((1, 2)))
                count = count + 1
        else:
            if len(index_pixel_l) == 1:
                # print('lm_right', lm_right)
                pixel_leftindex.append(np.asarray(index_pixel_l))
            else:
                pixel_leftindex.append(np.zeros((1, 2)))
                count = count + 1

        pixel_righthand.append(np.asarray(all_pixel_r))
        pixel_lefthand.append(np.asarray(all_pixel_l))



    print(F'sequence len {count}', index.shape[0], len(pixel_rightindex), len(pixel_leftindex))

    # if len(pixel_righthand) !=0:
    #     righthand = np.asarray(pixel_righthand)
    #
    # if len(pixel_lefthand) !=0:
    #     lefthand = np.asarray(pixel_lefthand)
    # print(np.array(pixel_lefthand).shape)

    return pixel_lefthand, pixel_righthand, pixel_rightindex, pixel_leftindex



def video_gen(img_root,root, index, fps, mode='hand'):

    size = (640, 480)
    media_root =F'{root}/HandVideo.avi'
    video = cv2.VideoWriter(media_root, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for j in index:
        imgroot = F'{img_root}/hand/rgb/image_{int(j)}.png'
        rgb = cv2.imread(imgroot)
        # 可以使用cv2.resize()进行修改
        video.write(rgb)
    video.release()
    print('Done')


if __name__ == '__main__':
    root = 'D:/Mydataset/scenariosS1/Child_mimi/complexity_level_L1/4/'
    skeleton_gen(root, 9)
