import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np


class handDisplay():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, imgBGR, depth_map, draw=True):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(depth_colormap, handLms, self.mpHands.HAND_CONNECTIONS)
        return depth_colormap



class handDetector():
    def __init__(self, mode=False, maxHand=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHand, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, imgBGR):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        return

    def findPosition(self, img):
        lmList_right, lmList_left = [], []
        h, w, c = img.shape
        # print('======',img.shape)
        depth_result_r, depth_result_l = [], []
        pixel_r, pixel_l = [], []
        if self.result.multi_hand_landmarks:
            #print('###', len(self.result.multi_handedness), self.result.multi_handedness)
            for idx, hand_handedness in enumerate(self.result.multi_handedness):
                handedness = hand_handedness.classification[0].label
                myhand = self.result.multi_hand_landmarks[idx]

                if len(myhand.landmark) == 21:
                    for id, lm in enumerate(myhand.landmark):
                        cx = min(int(lm.x * w), img.shape[1] - 1)
                        cy = min(int(lm.y * h), img.shape[0] - 1)

                        if handedness == 'Right':
                            pixel_r.append([cx, cy])

                        if handedness == 'Left':
                            pixel_l.append([cx, cy])
        return pixel_r, pixel_l
