import sys
from PyQt5.QtCore import Qt, QThread, QSize, pyqtSignal
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
from pylsl import StreamInfo, StreamOutlet, local_clock
import sys
import pyrealsense2 as rs
import datetime
import cv2
import numpy as np
import keyboard


ft = tr.find_all_eyetrackers()
if len(ft) == 0:
    print("No Eye Trackers found!?")
    exit(1)

# Pick first tracker
mt = ft[0]
print("Found Tobii Tracker at '%s'" % (mt.address))

my_eyeTracker = tr.find_all_eyetrackers()[0]

connect_device = []
for d in rs.context().devices:
    print('Found device: ',
          d.get_info(rs.camera_info.name), ' ',
          d.get_info(rs.camera_info.serial_number))
    if d.get_info(rs.camera_info.name).lower() != 'platform camera':
        connect_device.append(d.get_info(rs.camera_info.serial_number))

if len(connect_device) < 2:
    print('Registrition needs two camera connected.But got one.')
    exit()


rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

rs_config_facial = rs.config()
rs_config_facial.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_config_facial.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print('trun on realsense')

rs_config.enable_device('013222071729') #013222071729 #hand
pipeline1 = rs.pipeline()
pipeline1.start(rs_config)

rs_config_facial.enable_device('008222072729') #008222072729
pipeline2 = rs.pipeline()
pipeline2.start(rs_config_facial)

def Turn_Off_Realsense():
    pipeline1.stop()
    pipeline2.stop()

def check_path_file(file, create_if_missing=True):
  path_file = os.path.dirname(file)
  # print(path_file)
  if not os.path.exists(path_file):
    os.makedirs(path_file)
    return False
  else:
    return True


channels = 31  # count of the below channels, incl. those that are 3 or 2 long
gaze_stuff = [
    ('device_time_stamp', 1),

    ('left_gaze_origin_validity', 1),
    ('right_gaze_origin_validity', 1),

    ('left_gaze_origin_in_user_coordinate_system', 3),
    ('right_gaze_origin_in_user_coordinate_system', 3),

    ('left_gaze_origin_in_trackbox_coordinate_system', 3),
    ('right_gaze_origin_in_trackbox_coordinate_system', 3),

    ('left_gaze_point_validity', 1),
    ('right_gaze_point_validity', 1),

    ('left_gaze_point_in_user_coordinate_system', 3),
    ('right_gaze_point_in_user_coordinate_system', 3),

    ('left_gaze_point_on_display_area', 2),
    ('right_gaze_point_on_display_area', 2),

    ('left_pupil_validity', 1),
    ('right_pupil_validity', 1),

    ('left_pupil_diameter', 1),
    ('right_pupil_diameter', 1)
]

def unpack_gaze_data(gaze_data):
    x = []
    for s in gaze_stuff:
        d = gaze_data[s[0]]
        if isinstance(d, tuple):
            x = x + list(d)
        else:
            x.append(d)
    return x


def setup_lsl():
    global channels
    global gaze_stuff

    info = lsl.StreamInfo('Tobii', 'ET', channels, 30, 'float32', mt.address)
    info.desc().append_child_value("manufacturer", "Tobii")
    channels = info.desc().append_child("channels")
    cnt = 0
    for s in gaze_stuff:
        if s[1] == 1:
            cnt += 1
            channels.append_child("channel") \
                .append_child_value("label", s[0]) \
                .append_child_value("unit", "device") \
                .append_child_value("type", 'ET')
        else:
            for i in range(s[1]):
                cnt += 1
                channels.append_child("channel") \
                    .append_child_value("label", "%s_%d" % (s[0], i)) \
                    .append_child_value("unit", "device") \
                    .append_child_value("type", 'ET')

    outlet = lsl.StreamOutlet(info)

    return outlet

last_report = 0
N = 0
hand_count_num = 1
eye_count_num = 1

def gaze_data_callback(gaze_data):
    '''send gaze data'''

    # for k in sorted(gaze_data.keys()):
    #     print ' ' + k + ': ' +  str(gaze_data[k])

    try:
        global last_report
        global tobii_outlet
        global N
        global hand_count_num, eye_count_num
        # left = gaze_data['left_gaze_point_on_display_area']
        # right = gaze_data['right_gaze_point_on_display_area']

        # sts = gaze_data['system_time_stamp'] / 1000000.
        # print(sts,lsl.local_clock())
        # tobii_outlet.push_sample(unpack_gaze_data(gaze_data), local_clock())

        # if sts > last_report + 5:
        #     sys.stdout.write("%14.3f: %10d packets\r" % (sts, N))
        #     last_report = sts
        # N += 1
        # print(N)

        if hand_count_num > last_report:
            tobii_outlet.push_sample(unpack_gaze_data(gaze_data), local_clock())
            last_report = hand_count_num
            N += 1
            print('gaze_num', N)

        # print unpack_gaze_data(gaze_data)
    except:
        print("Error in callback: ")
        print(sys.exc_info())

        halted = True


def push_save(pipeline, root, outlet, type):
    global hand_count_num, eye_count_num, N
    frameCounter = 1
    align1 = rs.align(rs.stream.color)
    while True:
        no_error, frames = pipeline.try_wait_for_frames(100)
        if not no_error:
            continue
        frames = align1.process(frames)
        # Update color and depth frames:
        color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = cv2.flip(np.asanyarray(color_frame.get_data()), 1)
        aligned_depth = cv2.flip(np.asanyarray(aligned_depth_frame.get_data()), 1)
        if N > 0:
            sts = local_clock()
            outlet.push_sample([frameCounter], sts)
            #cv2.imshow('color',color_image)
            cv2.imwrite(F'{root}/{type}/rgb/image_{frameCounter}.png', color_image)
            cv2.imwrite(F'{root}/{type}/depth/depth_{frameCounter}.png', np.asarray(aligned_depth, np.uint16))

            frameCounter += 1
            if type == 'hand':
                hand_count_num += 1
            else:
                eye_count_num += 1
            print(F'frameCounter{type}', frameCounter)


def start_gaze_tracking():
    mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    return True


def end_gaze_tracking():
    mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    return True


class RealSenseThread1(QThread):
    def __init__(self):
        super(RealSenseThread1, self).__init__()

    def run(self):
        global data_path, hand_outlet
        push_save(pipeline1, data_path, hand_outlet, 'hand')


class RealSenseThread2(QThread):
    def __init__(self):
        super(RealSenseThread2, self).__init__()

    def run(self):
        global data_path, face_outlet
        push_save(pipeline2, data_path, face_outlet, 'face')


class TobiiThread(QThread):
    def __init__(self):
        super(TobiiThread, self).__init__()

    def run(self):
        start_gaze_tracking()

### outlet

info1 = StreamInfo(name='Hand_camera', type='videostream', channel_format='float32', channel_count=1,
                       nominal_srate=30,
                       source_id=str('013222071729'))
info2 = StreamInfo(name='Face_camera', type='videostream', channel_format='float32', channel_count=1,
                       nominal_srate=30,
                       source_id=str('008222072729'))



if __name__ == '__main__':
    global tobii_outlet, hand_outlet, face_outlet
    halted = False
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # info1.desc().append_child_value("videoFile", filename)
    # info2.desc().append_child_value("videoFile", filename)
    tobii_outlet = setup_lsl()
    hand_outlet = StreamOutlet(info1)
    face_outlet = StreamOutlet(info2)

    data_path = F'C:/tmp/{filename}/'
    check_path_file(F'C:/tmp/{filename}/hand/rgb/', True)
    check_path_file(F'C:/tmp/{filename}/hand/depth/', True)
    check_path_file(F'C:/tmp/{filename}/face/rgb/', True)
    check_path_file(F'C:/tmp/{filename}/face/depth/', True)

    Rworkthread1 = RealSenseThread1()
    Rworkthread1.start()

    Rworkthread2 = RealSenseThread2()
    Rworkthread2.start()

    EyeThread = TobiiThread()
    EyeThread.start()
    # Main loop; run until escape is pressed
    print("%14.3f: LSL Running; press CTRL-C repeatedly to stop" % lsl.local_clock())
    try:
        while not halted:
            time.sleep(1)
            keys = keyboard.read_key()
            print(keys)

            if len(keys) != 0:

                if keys == 'esc':
                    halted = True

            if halted:
                break

            # print lsl.local_clock()

    except:
        print("Halting...")

    print("terminating tracking now")

    end_gaze_tracking()
    Turn_Off_Realsense()