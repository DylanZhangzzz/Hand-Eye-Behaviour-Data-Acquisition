import pyxdf
import numpy as np
import os
import cv2
# print(data[0],header)
def sort_seq(data):
    seq = {}
    for id, stream in enumerate(data):
        seq[stream['info']['name'][0]] = id
    return seq

def sort_path(data):
    for id, stream in enumerate(data):
        if stream['info']['name'][0] == 'Hand_camera':
            Hand_camera_path = stream['info']['desc'][0]['videoFile'][0]

        if stream['info']['name'][0] == 'Face_camera':
            Face_camera_path  = stream['info']['desc'][0]['videoFile'][0]

    if Hand_camera_path  == Face_camera_path :
        return Hand_camera_path

def check_path_file(file, create_if_missing=True):
  path_file = os.path.dirname(file)
  # print(path_file)
  if not os.path.exists(path_file):
    os.makedirs(path_file)
    return False
  else:
    return True

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def seq_main(data):
    start_stamp = {}
    seq = sort_seq(data)
    hand_seq = data[seq['Hand_camera']]
    face_seq = data[seq['Face_camera']]
    gaze_seq = data[seq['Tobii']]

    # print('gaze_seq', gaze_seq)
    # print(left,right)
    # print(hand_seq['time_stamps'].shape, hand_seq['time_series'].shape)
    # print(face_seq['time_stamps'].shape, face_seq['time_series'].shape)
    # print(gaze_seq['time_stamps'].shape, gaze_seq['time_series'].shape)

    list_seq = [hand_seq['time_stamps'][0], face_seq['time_stamps'][0], gaze_seq['time_stamps'][0]]
    start_time = max(list_seq)
    hand_start = find_nearest(hand_seq['time_stamps'], start_time)
    face_start = find_nearest(face_seq['time_stamps'], start_time)
    gaze_start = find_nearest(gaze_seq['time_stamps'], start_time)


    start_stamp['hand'] = np.where(hand_seq['time_stamps'] == hand_start)[0][0]
    start_stamp['face'] = np.where(face_seq['time_stamps'] == face_start)[0][0]
    start_stamp['gaze'] = np.where(gaze_seq['time_stamps'] == gaze_start)[0][0]

    dur = min([hand_seq['time_stamps'].shape[0]-start_stamp['hand'], face_seq['time_stamps'].shape[0]-start_stamp['face'], gaze_seq['time_stamps'].shape[0]-start_stamp['gaze']])

    gaze_timestamp = gaze_seq['time_stamps'][start_stamp['gaze']: start_stamp['gaze']+dur]
    # print(start_stamp)

    # print([hand_seq['time_stamps'].shape[0]-start_stamp['hand'], face_seq['time_stamps'].shape[0]-start_stamp['face'], gaze_seq['time_stamps'].shape[0]-start_stamp['gaze']])
    # print('==',start_stamp, dur)
    # print(gaze_timestamp.shape)
    hand_data = hand_seq['time_series'][start_stamp['hand']: start_stamp['hand']+dur]

    return hand_data

def image_loader(root, j):
    # print(F'{root}/hand/rgb/image_{j}.png')
    # rgb = cv2.flip(cv2.imread(F'{root}/hand/rgb/image_{j}.png'),0)
    img =cv2.imread(F'{root}/hand/rgb/image_{j}.png')

    tmp = img.copy()

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imwrite("canny.jpg", edges)

    src = np.float32([[70, 13], [503, 11], [2, 432], [570, 428]])
    dst = np.float32([[0, 0], [1080, 0], [0, 1080], [1080, 1080]])
    m = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(tmp, m, (1080, 1080))
    # cv2.imwrite('result.png',result)

    cv2.imshow('image', result)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    return result


def main():
    name = 'mimi'
    # try:
    for com in (0,):
        for num in (1, ):

            if num < 10:
                num = "00%d" % num
            elif num < 100:
                num = "0%d" % num
            else:
                num = str(num)
            print(num)
            save_path = F'F:/Dataset_new/{name}/complexity00{com}/trial_{num}/'
            data, header = pyxdf.load_xdf(F'G:/dataset/{name}/complexity{com}/trials_{num}.xdf')

            path = sort_path(data)
            print(path)
            img_root = F'G:/dataset/{path}'

            hand_data = seq_main(data)

            print(hand_data)
            img = image_loader(img_root, int(hand_data[-1]))




    # except:
    #     pass












if __name__ == '__main__':
    main()