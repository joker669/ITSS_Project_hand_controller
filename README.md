# ITSS_Project_hand_controller

don't forget to install requirements before runing this demo
run "pip install -r ./requirements.txt"
##APP
If you want to try the app
click APP ->> app.bat

Operation:
press "q" to quit

##Gesture recognition
Click APP->> demo.bat

Operation:
press "q" to switch modes(21points mode & DTC mode)(DTC is the system we proposed means “Detection + Tracking + Classification”)
press "esc" to quit


###motion detection demo
Motion_HOF_SVM ->> detect_motion.ipynb

###recognition based on skin detection + HOG + SVM
Skin_tone_detection_HOG_SVM ->> handclassification.ipynb

###Hand Tracker demo
hand_tracker ->> tracker_kcf.ipynb
when you start to run this process, fisrt it will take a picture and you should draw the area of your hand, then it will track your hand.
 
