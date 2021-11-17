import threading
import time

import cv2
import pyautogui
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from cvzone.HandTrackingModule import HandDetector
from ITSS_Util import *
from ITSS_Util import fingersUp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import operator
import numpy as np

'''
上下左右，大拇指
音量 调整    gun
两只手  gun 截图
上下滑动 手掌
zoom 两个食指
'''
gestrue_dict = {0: "fist", 1: "five", 2: "gundown", 3: "gunup", 4: "one", 5: "thumbdown", 6: "thumbleft",
                7: "thumbright", 8: "thumbup"}


# 静态姿势
def Isthumbup(hand):
    """

    :return:
    """
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 0, 0, 0, 0]
    type_hand = hand["type"]
    AB = [lmList[4][0], lmList[4][1], lmList[0][0], lmList[0][1]]
    CD = [1, 0, 0, 0]
    ang = angle_full(AB, CD)
    flag_s = 0
    if type_hand == "Right":
        if 90 > ang > 30:
            flag_s = 1
    else:
        if 140 > ang > 90:
            flag_s = 1

    if operator.eq(fingers, upList1) and flag_s == 1:
        return True

    return False


def Isthumbright(hand):
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 0, 0, 0, 0]
    type_hand = hand["type"]

    AB = [lmList[4][0], lmList[4][1], lmList[17][0], lmList[17][1]]
    CD = [1, 0, 0, 0]
    ang = angle(AB, CD)
    flag_s = 0
    if 180 > ang > 120:
        flag_s = 1
    if operator.eq(fingers, upList1) and flag_s == 1:
        return True
    return False


def Isthumbleft(hand):
    """

    :return:
    """
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 0, 0, 0, 0]
    type_hand = hand["type"]

    AB = [lmList[4][0], lmList[4][1], lmList[0][0], lmList[0][1]]
    CD = [1, 0, 0, 0]
    ang = angle_full(AB, CD)
    flag_s = 0

    if type_hand == "Right":
        if 360 > ang > 330 or 30 > ang > 0:
            flag_s = 1
    else:
        if 60 > ang > 0:
            flag_s = 1

    if operator.eq(fingers, upList1) and flag_s == 1:
        return True
    return False


def Isthumbdown(hand):
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 0, 0, 0, 0]
    AB = [lmList[4][0], lmList[4][1], lmList[0][0], lmList[0][1]]
    CD = [1, 0, 0, 0]
    type_hand = hand["type"]
    ang = angle_full(AB, CD)
    flag_s = 0
    if type_hand == "Right":
        if 330 > ang > 270:
            flag_s = 1
    else:
        if 270 > ang > 230:
            flag_s = 1

    if operator.eq(fingers, upList1) and flag_s == 1:
        return True
    return False


def Isgunup(hand):
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 1, 0, 0, 0]
    if operator.eq(fingers, upList1) and lmList[0][1] > lmList[8][1]:
        return True
    return False


def Isgundown(hand):
    lmList = hand["lmList"]
    fingers = fingersUp(lmList)
    upList1 = [1, 1, 0, 0, 0]
    if operator.eq(fingers, upList1) and lmList[0][1] < lmList[8][1]:
        return True
    return False


# 动态姿势
def Isfive(finger):
    if (finger[0] == 1 and finger[1] == 1 and finger[2] == 1 and finger[3] == 1 and finger[4] == 1):
        return True
    return False


def Isone(finger):
    # print(finger)
    if (finger[0] == 0 and finger[1] == 1 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0):
        # print("true")
        return True
    else:
        # print("FF")
        return False


def Isfist(hand):
    return False


# 静态姿势
def op_move(detector, allHands):  # 输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    # 模拟指令输出
    # 右手
    # 向左 330 - 360 | 0 - 30
    # 向右 120 - 180
    # 向上 90  - 30
    # 向下 270 - 330
    # 左手
    # 向左 0-60
    # 向上 90-140
    # 向右 150-200
    # 向下 230-270
    if len(allHands) == 1:
        hand = allHands[0]
        if Isthumbdown(hand):
            print("thumbdown")
            # down
            pyautogui.press('down')
            return True
        elif Isthumbup(hand):
            # up
            pyautogui.press('up')
            print("thumbup")
            return True
        elif Isthumbleft(hand):
            # right
            pyautogui.press('right')
            print("thumbleft")
            return True
        elif Isthumbright(hand):
            # left
            pyautogui.press('left')
            print("thumbright")
            return True
        else:
            return False
    return False


def vol_adjust(detector, allHands, volume):  # 输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    # 当前音量
    vl = volume.GetMasterVolumeLevel()
    # 获取音量范围
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    print(volRange)
    # 插值获取长度
    length = np.interp(vl, [minVol, maxVol], [50, 300])



    """
    右手:  向上gun 拳心朝摄像头 [1,1,0,0,0] 朝面[0,1,0,0,0] 向下gun(仅拳心朝面) [1,0,1,1,1] | [0,0,1,1,1]
    左手:  向上gun 拳心朝摄像头 [1,1,0,0,0] 朝面[0,1,0,0,0] 向下gun(仅拳心朝面) [1,0,1,1,1] | [0,0,1,1,1]
    写的有问题 相当于一根食指确定是否调节音量 || 已修复
    :param detector:
    :param allHands:
    :return:
    """
    if len(allHands) == 1:
        hand = allHands[0]
        if Isgunup(hand):
            print("gunup")
            vol = np.interp(length+1, [50, 300], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            print(int(length), vol)
            # volume_up
            return True
        elif Isgundown(hand):
            # volume_down
            print("gundown")
            vol = np.interp(length-1, [50, 300], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            print(int(length), vol)
            return True
        else:
            return False
    return False


def screen_shot(detector, allHands, img=None):
    """
        通过比较两只手的交错食指拇指的距离来判断是否是截屏手势
    """
    dot = []
    if len(allHands) == 2:
        for hand in allHands:
            lmList = hand["lmList"]
            dot.append(lmList[4])
            dot.append(lmList[8])
        if img is None:
            length1, info1 = detector.findDistance(dot[0], dot[3])
            length2, info2 = detector.findDistance(dot[1], dot[2])
            # print(length1, info1, length2, info2, (length1 < 50 and length2 < 50))
            if length1 < 50 and length2 < 50:
                # screenshot
                # im1 = pyautogui.screenshot()
                # im1.save('my_screenshot.png')
                return True
        else:
            length1, info1, img = detector.findDistance(dot[0], dot[3], img)
            length2, info2, img = detector.findDistance(dot[1], dot[2], img)
            # print(length1, info1, length2, info2, (length1 < 50 and length2 < 50))
            if length1 < 50 and length2 < 50:
                # screenshot
                # im1 = pyautogui.screenshot()
                # im1.save('my_screenshot.png')
                return True, img
    elif img is None:
        return False
    else:
        return False, img


# 动态姿势
def five_move():  # 手掌移动
    return True, 1


# def five_move_deamon():             #five_pos_dict = {0:"no", 1:"left", 2:"right", 3: "up", 4: "down"}
#     global cur_five_pos
#     global end
#     pre_five_pos = 0
#     while True:
#         if(end == 1):
#             return
#         if(pre_five_pos != cur_five_pos):
#             if(cur_five_pos == 1 and pre_five_pos == 2):
#                 print("press right")
#                 pyautogui.press('right')
#             elif(cur_five_pos == 2 and pre_five_pos == 1):
#                 print("press left")
#                 pyautogui.press('left')
#             elif(cur_five_pos == 3 and pre_five_pos == 4):
#                 print("press pagedown")
#                 pyautogui.press('pagedown')
#             elif(cur_five_pos == 4 and pre_five_pos == 3):
#                 print("press pageup")
#                 pyautogui.press('pageup')
#             pre_five_pos = cur_five_pos
#         time.sleep(0.5)
#
# def zoom_deamon():
#     global lock
#     global g_z_op
#     global end
#     while True:
#         if(end == 1):
#             return
#         if(g_z_op == 1):
#             print("rolllllllllll upppppppppp")
#             pyautogui.keyDown('ctrl')
#             #time.sleep(0.1)
#             pyautogui.scroll(150)
#             pyautogui.keyUp('ctrl')
#         elif(g_z_op == 2):
#             print("rolllllllllll downnnnnnnnnn")
#             pyautogui.keyDown('ctrl')
#             #time.sleep(0.1)
#             pyautogui.scroll(-150)
#             pyautogui.keyUp('ctrl')
#         else:
#             time.sleep(0.01)
#             pass
#
# def zoom(op): #缩放 1: 放大 2：缩小
#     global g_z_op
#     g_z_op = op
#     return True, 1


def posture(hand, finger):
    pos = -1
    time = 0
    if (Isthumbup(hand) == True):
        return pos, time
    elif (Isthumbright(hand) == True):
        return pos, time
    elif (Isthumbleft(hand) == True):
        return pos, time
    elif (Isthumbdown(hand) == True):
        return pos, time
    elif (Isgunup(hand) == True):
        return pos, time
    elif (Isgundown(hand) == True):
        return pos, time
    elif (Isfist(hand) == True):
        return pos, time
    return pos, time


#
#
#
#
# hand_pos_list = []
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# ###这里初始化全局遍历
# is_fst = 1
# pre_length = 0
# pre_five = 0# 无 0: 左 1: 右 2: 上 3: 下 4
# five_pos_dict = {0:"no", 1:"left", 2:"right", 3: "up", 4: "down"}
# cur_five_pos = 0
# #def main():
# #init
# g_z_op = 0
# end = 0
# zoom_d = threading.Thread(target=zoom_deamon)
# zoom_d.start()
# move_d = threading.Thread(target=five_move_deamon)
# move_d.start()
# if(1):
#     cap = cv2.VideoCapture(0)
#     detector = HandDetector(detectionCon=0.8, maxHands=2)
#     while True:
#         success, img = cap.read() #get hand map
#         hands, img = detector.findHands(img)  # with draw#每一帧得到手的21点
#         # hands = detector.findHands(img, draw=False)
#         # without draw
#         if hands:
#             # Hand 1
#             hand1 = hands[0]
#             lmList1 = hand1["lmList"]  # List of 21 Landmark points
#             bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
#             centerPoint1 = hand1['center']  # center of the hand cx,cy
#             handType1 = hand1["type"]  # Handtype Left or Right
#             fingers1 = fingersUp(lmList1)
#             if len(hands) == 2:
#                 # Hand 2
#                 hand2 = hands[1]
#                 lmList2 = hand2["lmList"]  # List of 21 Landmark points
#                 bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
#                 print("bbox1")
#                 print(bbox1)
#                 print("bbox2")
#                 print(bbox1)
#                 centerPoint2 = hand2['center']  # center of the hand cx,cy
#                 handType2 = hand2["type"]  # Hand Type "Left" or "Right"
#                 fingers2 = fingersUp(lmList2)
#                 # Find Distance between two Landmarks. Could be same hand or different hands
#                 # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
#                 # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
# ######################################2只手手势代码##############################################################
#             if(len(hands) == 2):
#                 ###################################两只手控制缩放代码
#                 if(Isone(fingers1) == True and Isone(fingers2) == True):     #进入zoom in out
#                     if(is_fst == 1):    ###判断是否是第一次
#                         is_fst = 0
#                         length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
#                         pre_length = length
#                     else:
#                         length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
#                         cur_length = length
#                         #diff_length = (cur_length*1.0)/pre_length
#                         if(pre_length - cur_length < -5):
#                             zoom(1)
#                             print("zoom out")
#                             pre_length = cur_length
#                         elif(pre_length - cur_length > 5):
#                             zoom(2)
#                             pre_length = cur_length
#                             print("zoom in")
#                         else:
#                             zoom(0)
#                         #print(diff_length)
#                 else:
#                     is_fst = 1
#                     zoom(0)
#                     print("zoom end")
# #####################################一只手手势代码####################################################
#             elif(len(hands) == 1):
#                 #print(fingers1)
#                 #####################手掌手势是动态动作，特殊处理
#                 if(Isfive(fingers1) == True):
#                     five_pos_temp = reg_five(lmList1)
#                     cur_five_pos = five_pos_temp
#                 else:
#                     cur_five_pos = 0
#                 #stop_time = -1
#                 Is_stop = -1 #stop要进行处理
#                 ######################其余静态手势动作
#                 if(Is_stop == 0):
#                     pos,time = posture(hand1, fingers1)
#                     stop_time = time                        #静态姿势
#                     try:
#                         act_func = gestrue_switch[pos]
#                         act_func()
#                     except:
#                         pass
#         cv2.imshow("Image", img)
#         k = cv2.waitKey(1) & 0xff
#         if k == 27:
#             end = 1
#             break
#
# move_d.join()
# zoom_d.join()
# cap.release()
# cv2.destroyAllWindows()

wCam, hCam = 240, 360


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = HandDetector(detectionCon=0.7)

    # volume setting
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    mute = volume.GetMute()

    print(mute)


    while True:
        success, img = cap.read()
        allHands, img = detector.findHands(img)

        if len(allHands) != 0:
             op_move(detector, allHands)
             vol_adjust(detector, allHands,volume)
             if screen_shot(detector, allHands):
                 print("screenshot")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
