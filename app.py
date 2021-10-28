import time
from cvzone.HandTrackingModule import HandDetector
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

    return bool, time

def Isthumbright():
    return bool, time

def Isthumbleft():
    return bool, time


def Isthumbdown():
    return bool, time
    
def Isgunup():
    return bool, time

def Isgundown():
    return bool, time

#动态姿势
def Isfive():
    return bool, time
    
def Isone():
    return bool, time

def Isfist():
    return bool, time


#静态姿势
def op_move():#输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    #模拟指令输出

def vol_adjust():#输入姿势  5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
    ##模拟指令输出

def screen_shot():


#动态姿势
def five_move():#手掌移动

def zoom(): #缩放



def posture(hand):
    if(Isthumbup(hand) == 1):
        return pos,time
    elif(Isthumbright(hand) == 1):
        return pos,time
    elif(Isthumbleft(hand) == 1):
        return pos,time
    elif(Isthumbdown(hand) == 1):
        return pos,time
    elif(Isgunup(hand) == 1):
        return pos,time
    elif(Isgundown(hand) == 1):
        return pos,time
    elif(Isone(hand) == 1):
        return pos,time
    elif(Isfive(hand) == 1):
        return pos,time
    elif(Isfist(hand) == 1):
        return pos,time
    return pos,time

###  0: "fist", 1: "five", 2: "gundown", 3: "gunup", 4: "one", 5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"
gestrue_switch = { 1: five_move, 2: vol_adjust, 3: vol_adjust, 4: zoom, 5: op_move, 6: op_move, 7: op_move, 8: op_move}

hand_pos_list = []

def main():
    while True:
        #get hand map
        #每一帧得到手的21点
        #一秒50帧
        lmlist #21点的list
        hand1
        fingers1 #手指状态
        stop_time
        Is_stop = 0 #stop要进行处理，触发后保持一段时间(1s)stop是1
        if(Is_stop == 0):
            pos,time = posture(hand1)
            stop_time = time
            if(pos == 1 or pos == 4):         #动态姿势
                
            else:                             #静态姿势
                act_func = gestrue_switch[pos]
                act_func()

            