import pyxdf
import numpy as np
import save_skeleton
import os
import HandTrackingModule
import utils

# print(data[0],header)
def sort_seq(data):
    seq = {}
    for id, stream in enumerate(data):
        seq[stream['info']['name'][0]] = id
    print(seq)
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
    print(gaze_seq['time_series'].shape)
    print(gaze_seq['time_series'][:, 0].shape)

    left = gaze_seq['time_series'][:, 23:25]
    right = gaze_seq['time_series'][:, 25:27]

    left_vector = gaze_seq['time_series'][:, 17:20]
    right_vector = gaze_seq['time_series'][:, 20:23]

    point_validity = gaze_seq['time_series'][:, 15]

    # print('=====',left.shape, right.shape, left_vector.shape, right_vector.shape)
    #
    # for i in range(left.shape[0]):
    #     print(left[i], left_vector[i], point_validity[i])
    # print(left,right)
    # print(hand_seq['time_stamps'].shape, hand_seq['time_series'].shape)
    # print(face_seq['time_stamps'].shape, face_seq['time_series'].shape)
    # print(gaze_seq['time_stamps'].shape, gaze_seq['time_series'].shape)

    # print(hand_seq['info'])
    # print(hand_seq['info']['effective_srate'])
    fps = hand_seq['info']['effective_srate']

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
    face_data = face_seq['time_series'][start_stamp['face']: start_stamp['face']+dur]
    left_data = left[start_stamp['gaze']: start_stamp['gaze']+dur]
    right_data = right[start_stamp['gaze']: start_stamp['gaze']+dur]

    print(hand_data.shape, face_data.shape, left_data.shape, right_data.shape, gaze_timestamp.shape)
    return hand_data, face_data, left_data, right_data, gaze_timestamp, fps


def main():
    names = ['Carrie']
    for name in names:
        for com in (6,):
            for num in range(1, 11):

                if num < 10:
                    num = "00%d" % num
                elif num < 100:
                    num = "0%d" % num
                else:
                    num = str(num)
                save_path = F'F:/All_Level_Final_Dataset/{name}/complexity00{com}/trial_{num}/'
                data, header = pyxdf.load_xdf(F'F:/Dataset/{name}/complexity00{com}/trial_{num}.xdf')
                # data, header = pyxdf.load_xdf(F'F:/ROG dataset/dataset/{name}/complexity{com}/trials_{num}.xdf')
                # data, header = pyxdf.load_xdf(F'F:/tmp/xml_result/old/{name}/complexity{1}/trial_{num}.xdf')

                path = sort_path(data)
                print(path)
                # img_root = F'F:/ROG dataset/dataset/{path}'
                img_root = F'F:/Dataset/{path}'

                hand_data, face_data, left_data, right_data, gaze_timestamp, fps = seq_main(data)

                if check_path_file(save_path) is not True:

                    save_skeleton.video_gen(img_root, save_path, hand_data, fps, mode='hand')

                    # print(hand_data)

                    pixel_lefthand, pixel_righthand, pixel_rightindex, pixel_leftindex = save_skeleton.skeleton_gen(img_root, hand_data, 'left')


                    # save_skeleton.video_gen(img_root, save_path, hand_data, mode='face')


                    if len(pixel_righthand) !=0:
                        righthand = np.asarray(pixel_righthand)
                        np.save(F'{save_path}Raw_righthand.npy', righthand)

                    if len(pixel_rightindex) != 0:
                        pixel_rightindex = np.asarray(pixel_rightindex)
                        pixel_rightindex = utils.skeleton_filter(pixel_rightindex)
                        np.save(F'{save_path}pixel_skeleton_rightindex.npy', pixel_rightindex)


                    if len(pixel_lefthand) !=0:
                        lefthand = np.asarray(pixel_lefthand)
                        np.save(F'{save_path}Raw_lefthand.npy', lefthand)

                    if len(pixel_leftindex) != 0:
                        pixel_leftindex = np.asarray(pixel_leftindex)
                        pixel_leftindex = utils.skeleton_filter(pixel_leftindex)
                        np.save(F'{save_path}pixel_skeleton_leftindex.npy', pixel_leftindex)

                    np.save(F'{save_path}gaze_left_eye.npy', left_data)
                    np.save(F'{save_path}gaze_right_eye.npy', right_data)
                    np.save(F'{save_path}gaze_timestamp.npy', gaze_timestamp)

    # except:
    #     pass












if __name__ == '__main__':
    main()