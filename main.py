import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import time
import mediapipe as mp
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array


#Load the model
model = load_model('My_model.h5')
#Predcitions dictionary
labels_to_char = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'del',27:'space',28:'nothing'}

#Media pipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
padding = 50 #Padding so that whole hand fits

# image_path = '/home/szramstar/Desktop/SW/test_b.jpg'
# image = cv2.imread(image_path)
# image = cv2.resize(image,(64,64))
# image = img_to_array(image)
# image = np.expand_dims(image,axis=0)
# image = image/255.0
# prediction = model.predict(image)
# label = np.argmax(prediction)


# print(labels_to_char[label])



video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    if not ret:
        break
    #Convert to rgb
    frame = cv2.cvtColor(cv2.flip(frame,1) ,cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Calculate the bounding box of the hand
            x_min = max(0,min([landmark.x for landmark in hand_landmarks.landmark])*frame.shape[1]-padding)
            x_max = min(frame.shape[1],max([landmark.x for landmark in hand_landmarks.landmark])*frame.shape[1]+padding)
            y_min = max(0,min([landmark.y for landmark in hand_landmarks.landmark])*frame.shape[0]-padding)
            y_max = min(frame.shape[0],max([landmark.y for landmark in hand_landmarks.landmark])*frame.shape[0]+padding)

            #Crop the hand
            hand_img = frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            #Resize the hand for the model
            hand_img = cv2.resize(hand_img,(64,64))
            hand_img = hand_img/255
            hand_img = np.expand_dims(hand_img,axis=0)
            #Predict the letter
            # pred = model.predict(hand_img)
            # label = np.argmax(pred)
    
    
            # print(labels_to_char[label])

            #Draw the bounding box
            cv2.rectangle(frame,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),2)
            hand_frame = frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            hand_frame = cv2.resize(hand_frame,(64,64))
            hand_frame = hand_frame/255
            pred = model.predict(np.expand_dims(hand_frame,axis=0))
            label = np.argmax(pred)
            print(labels_to_char[label])
            # cv2.imshow("Hand",hand_frame)
    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
