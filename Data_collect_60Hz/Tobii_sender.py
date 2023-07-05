################################
# Preface here
#
# from psychopy import prefs, visual, core, event, monitors, tools, logging
import numpy as np
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
import sys
import keyboard

# Find Eye Tracker and Apply License (edit to suit actual tracker serial no)

ft = tr.find_all_eyetrackers()

initial_gaze_output_frequency = ft.get_gaze_output_frequency()
print("The eye tracker's initial gaze output frequency is {0} Hz.".format(initial_gaze_output_frequency))
try:
    for gaze_output_frequency in ft.get_all_gaze_output_frequencies():
        ft.set_gaze_output_frequency(gaze_output_frequency)
        print("Gaze output frequency set to {0} Hz.".format(gaze_output_frequency))
finally:
    ft.set_gaze_output_frequency(initial_gaze_output_frequency)
    print("Gaze output frequency reset to {0} Hz.".format(initial_gaze_output_frequency))


if len(ft) == 0:
    print("No Eye Trackers found!?")
    exit(1)

# Pick first tracker
mt = ft[0]
print("Found Tobii Tracker at '%s'" % (mt.address))


last_report = 0
N = 0

channels = 4 #31  # count of the below channels, incl. those that are 3 or 2 long
gaze_stuff = [
    # ('device_time_stamp', 1),
    #
    # ('left_gaze_origin_validity', 1),
    # ('right_gaze_origin_validity', 1),
    #
    # ('left_gaze_origin_in_user_coordinate_system', 3),
    # ('right_gaze_origin_in_user_coordinate_system', 3),
    #
    # ('left_gaze_origin_in_trackbox_coordinate_system', 3),
    # ('right_gaze_origin_in_trackbox_coordinate_system', 3),
    #
    # ('left_gaze_point_validity', 1),
    # ('right_gaze_point_validity', 1),

    # ('left_gaze_point_in_user_coordinate_system', 3),
    # ('right_gaze_point_in_user_coordinate_system', 3),

    ('left_gaze_point_on_display_area', 2),
    ('right_gaze_point_on_display_area', 2),

    # ('left_pupil_validity', 1),
    # ('right_pupil_validity', 1),
    #
    # ('left_pupil_diameter', 1),
    # ('right_pupil_diameter', 1)
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



def gaze_data_callback(gaze_data):
    '''send gaze data'''

    # for k in sorted(gaze_data.keys()):
    #     print ' ' + k + ': ' +  str(gaze_data[k])

    try:
        global last_report
        global outlet
        global N
        global halted

        # left = gaze_data['left_gaze_point_on_display_area']
        # right = gaze_data['right_gaze_point_on_display_area']

        sts = gaze_data['system_time_stamp'] / 1000000.
        # print(sts,lsl.local_clock())
        outlet.push_sample(unpack_gaze_data(gaze_data), lsl.local_clock())

        if sts > last_report + 5:
            sys.stdout.write("%14.3f: %10d packets\r" % (sts, N))
            last_report = sts
        N += 1

        # print unpack_gaze_data(gaze_data)
    except:
        print("Error in callback: ")
        print(sys.exc_info())

        halted = True


def start_gaze_tracking():
    mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    return True


def end_gaze_tracking():
    mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    return True


halted = False


# Set up lsl stream
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


if __name__ == '__main__':
    outlet = setup_lsl()

    # Main loop; run until escape is pressed
    print("%14.3f: LSL Running; press CTRL-C repeatedly to stop" % lsl.local_clock())

    start_gaze_tracking()
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