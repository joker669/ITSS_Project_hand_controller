from cvzone.HandTrackingModule import HandDetector
import threading,time
import cv2
from math import *
import pyautogui
from ITSS_Util import *
'''
上下左右，大拇指
音量 调整    gun
两只手  gun 截图
上下滑动 手掌
zoom 两个食指
'''
gestrue_dict = {0: "fist", 1: "five", 2: "gundown", 3: "gunup", 4: "one", 5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"}


#静态姿势
def Isthumbup():

    return False

def Isthumbright():
    return False

def Isthumbleft():
    return False


def Isthumbdown():
    return False
    
def Isgunup():
    return False

def Isgundown():
    return False

#动态姿势
def Isfive(finger):
    if(finger[0] == 1 and finger[1] == 1 and finger[2] == 1 and finger[3] == 1 and finger[4] == 1):
        return True
    return False
    
def Isone(finger):
    #print(finger)
    if(finger[0] == 0 and finger[1] == 1 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0):
        #print("true")
        return True
    else:
        #print("FF")
        return False

def Isfist(hand):
    return False


#静态姿势
def op_move():#输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    #模拟指令输出
    return True, 1

def vol_adjust():#输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    ##模拟指令输出
    return True, 1

def screen_shot():
    return True, 1

#动态姿势
def five_move():#手掌移动
    return True, 1

def zoom_deamon():
    global lock
    global g_z_op
    global end
    while True:
        if(end == 1):
            return
        if(g_z_op == 1):
            print("rolllllllllll upppppppppp")
            pyautogui.keyDown('ctrl')
            #time.sleep(0.1)
            pyautogui.scroll(150)
            pyautogui.keyUp('ctrl')
        elif(g_z_op == 2):
            print("rolllllllllll downnnnnnnnnn")
            pyautogui.keyDown('ctrl')
            #time.sleep(0.1)
            pyautogui.scroll(-150)
            pyautogui.keyUp('ctrl')
        else:
            time.sleep(0.01)
            pass

def zoom(op): #缩放 1: 放大 2：缩小
    global g_z_op
    g_z_op = op
    return True, 1


def posture(hand, finger):
    pos = -1
    time = 0
    if(Isthumbup(hand) == True):
        return pos,time
    elif(Isthumbright(hand) == True):
        return pos,time
    elif(Isthumbleft(hand) == True):
        return pos,time
    elif(Isthumbdown(hand) == True):
        return pos,time
    elif(Isgunup(hand) == True):
        return pos,time
    elif(Isgundown(hand) == True):
        return pos,time
    elif(Isfist(hand) == True):
        return pos,time
    return pos,time



###  0: "fist", 1: "five", 2: "gundown", 3: "gunup", 4: "one", 5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
gestrue_switch = { 1: five_move, 2: vol_adjust, 3: vol_adjust, 4: zoom, 5: op_move, 6: op_move, 7: op_move, 8: op_move}

hand_pos_list = []

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
###这里初始化全局遍历
is_fst = 1
pre_length = 0
pre_five = 0# 无 0: 左 1: 右 2: 上 3: 下 4
five_pos = {0:"no", 1:"left", 2:"right", 3: "up", 4: "down"}
#def main():
#init
g_z_op = 0
end = 0
zoom_d = threading.Thread(target=zoom_deamon)
zoom_d.start()
if(1):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        success, img = cap.read() #get hand map
        hands, img = detector.findHands(img)  # with draw#每一帧得到手的21点
        # hands = detector.findHands(img, draw=False)  
        # without draw
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            fingers1 = detector.fingersUp(hand1)
            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                fingers2 = detector.fingersUp(hand2)
                # Find Distance between two Landmarks. Could be same hand or different hands
                # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
######################################2只手手势代码##############################################################
            if(len(hands) == 2):
                ###################################两只手控制缩放代码
                if(Isone(fingers1) == True and Isone(fingers2) == True):     #进入zoom in out
                    if(is_fst == 1):    ###判断是否是第一次
                        is_fst = 0
                        length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                        pre_length = length
                    else:
                        length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                        cur_length = length
                        #diff_length = (cur_length*1.0)/pre_length
                        if(pre_length - cur_length < -5):
                            zoom(1)
                            print("zoom out")
                            pre_length = cur_length
                        elif(pre_length - cur_length > 5):
                            zoom(2)
                            pre_length = cur_length
                            print("zoom in")
                        else:
                            zoom(0)
                        #print(diff_length)
                else:
                    is_fst = 1
                    zoom(0)
                    print("zoom end")
#####################################一只手手势代码####################################################
            elif(len(hands) == 1):
                print(fingers1)
                #####################手掌手势是动态动作，特殊处理
                if(Isfive(fingers1) == True):
                    if(is_fst == 1):
                        is_fst = 0
                        pre_five = reg_five(lmList1)
                        print(five_pos[pre_five])
                    else:
                        cur_five = reg_five(lmList1)
                        print(five_pos[cur_five]) 
                else:
                    is_fst = 1
                #stop_time = -1
                Is_stop = -1 #stop要进行处理，触发后保持一段时间(1s)stop是1
                ######################其余静态手势动作
                if(Is_stop == 0):
                    pos,time = posture(hand1, fingers1)
                    stop_time = time                        #静态姿势
                    try:
                        act_func = gestrue_switch[pos]
                        act_func()
                    except:
                        pass
        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            end = 1
            break
        
        
zoom_d.join()
cap.release()
cv2.destroyAllWindows()
            