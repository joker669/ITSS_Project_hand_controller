from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import sklearn.metrics as metrics
from yolo import YOLO
import time
from cvzone.HandTrackingModule import HandDetector


def preprocess(image):
    image = cv2.resize(image,(224,224))
    image = image.astype("float32")
    image = image.reshape(1,224,224,3)
    return image
    
    
model = ResNet50(weights=None,
                include_top=False,
                input_tensor=Input(shape=(224, 224, 3)))
def createModel1():
    global model
    h   = model.output
    h   = AveragePooling2D(pool_size=(5, 5))(h)
    h   = Flatten(name="flatten")(h)
    #h   = Dense(512, activation="relu")(h)
    h   = Dense(128, activation="relu")(h)
    h   = Dropout(0.5)(h)
    h   = Dense(9, activation="softmax")(h)
    model = Model(inputs=model.input, outputs=h)
    return model
model = createModel1()
model.load_weights("modelsensing1.hdf5")
def gesture_detect(frame, model, yolo):
    width, height, inference_time, results = yolo.inference(frame)
    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)
    #print("44")
    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    #if args.hands != -1:
    hand_count = int(1)
    # display hands
    box = tuple()
    #print("55")
    ok = 0
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)
        roi_color = frame[y:y+h, x:x+w, :]
        roi_color = preprocess(roi_color)
        #print("66")
        predicts_m = model(roi_color,training = False)
        #print("77")
        predicts = np.array(predicts_m)
        maxindex1 = int(np.argmax(predicts))
        #print("88")
        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        cv2.putText(frame, gestrue_dict[maxindex1], (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        box =  tuple([x, y, w, h])
        ok = 1
    return frame,box,ok

class general_pose_model(object):
    def __init__(self, modelpath):
        self.num_points = 22
        self.point_pairs = [[0,1],[1,2],[2,3],[3,4],
                            [0,5],[5,6],[6,7],[7,8],
                            [0,9],[9,10],[10,11],[11,12],
                            [0,13],[13,14],[14,15],[15,16],
                            [0,17],[17,18],[18,19],[19,20]]
        # self.inWidth = 368
        self.inHeight = 360
        self.threshold = 0.1
        self.hand_net = self.get_hand_model(modelpath)


    def get_hand_model(self, modelpath):

        prototxt   = os.path.join(modelpath, "./pose_deploy.prototxt")
        caffemodel = os.path.join(modelpath, "./pose_iter_102000.caffemodel")
        hand_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return hand_model


    def predict(self, imgfile):
        #img_cv2 = cv2.imread(imgfile)
        img_cv2 = imgfile
        [img_height, img_width, _ ]= img_cv2.shape
        aspect_ratio = img_width / img_height

        inWidth = int(((aspect_ratio * self.inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(img_cv2, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.hand_net.setInput(inpBlob)
        output = self.hand_net.forward()
        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :] # confidence map.
            probMap = cv2.resize(probMap, (img_width, img_height))
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        return points

    def vis_pose(self, imgfile, points):
        img_cv2 = imgfile
        # Draw Skeleton
        for pair in self.point_pairs:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(img_cv2, points[partA], points[partB], (0, 255, 255), 3)
                cv2.circle(img_cv2, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        return img_cv2



def ellipse_detect(image):
    """YCrCb颜色空间的Cr分量+Otsu阈值分割
    :param image: 图片路径
    :return: None
    """
    img = image
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
 
    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    skin = cv2.cvtColor(skin,cv2.COLOR_GRAY2BGR)
    return skin
    

#if args.network == "normal":
#    print("loading yolo...")
#    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
#elif args.network == "prn":
#    print("loading yolo-tiny-prn...")
#    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
#elif args.network == "v4-tiny":
#    print("loading yolov4-tiny-prn...")
#    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
#else:
#    print("loading yolo-tiny...")
#    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
yolo.size = int(416)
yolo.confidence = float(0.2)
#pose model init
modelpath = "./models"

#gestrue_dict
gestrue_dict = {0: "fist", 1: "five", 2: "gundown", 3: "gunup", 4: "one", 5: "thumbdown", 6: "thumbleft", 7: "thumbright", 8: "thumbup"}
print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
WW = int(vc.get(3))
HH = int(vc.get(4))

detector = HandDetector(maxHands=2,detectionCon=0.8)
#tracker init
tracker = cv2.TrackerKCF_create()

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

op_mode = 1 #0: gestrue 1: 21 points
hand_track = 0
while rval:
    if(op_mode == 0):
        if(hand_track == 0):
            skin = ellipse_detect(frame)
            #print("1")
            try:
                frame,bbox,ok = gesture_detect(frame, model, yolo)
            except:
                #print("2")
            if(ok == 1):
                #print("13")
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(skin, bbox)
                #print("23")
                hand_track = 1
            #cv2.imshow("preview", frame)
            #cv2.waitKey(0)
        else:
            start_time = time.time()
            skin = ellipse_detect(frame)
            #print("11")
            ok, tbox = tracker.update(skin)
            #print("22")
            if ok:
                p1 = (max(int(tbox[0] - 20),0), max(int(tbox[1] - 20),0))
                p2 = (min(int(tbox[0] + tbox[2]+20), WW), min(int(tbox[1] + tbox[3]+20), HH))
                #print(p1, p2, WW, HH)
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                roi_color = frame[p1[1]:p2[1], p1[0]:p2[0], :]
                roi_color = preprocess(roi_color)
                #roi_color = preprocess(frame)
                predicts_m = model(roi_color,training = False)
                predicts = np.array(predicts_m)
                maxindex1 = int(np.argmax(predicts))
                cv2.putText(frame, gestrue_dict[maxindex1], (p1[0], p2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
            else:
                hand_track = 0
            fpp = 1 / (time.time() - start_time)
            cv2.putText(frame, "Tracker, Frame " + str(fpp), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    elif(op_mode == 1):
        start_time = time.time()
        hand, frame = detector.findHands(frame)
        fpp = 1 / (time.time() - start_time)
        cv2.putText(frame, "Tracker, Frame " + str(fpp), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(1)
    if key == ord("q"):  # exit on ESC
        if(op_mode == 1):
            op_mode = 0
        elif(op_mode == 0):
            op_mode = 1
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()