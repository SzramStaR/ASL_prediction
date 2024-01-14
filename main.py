import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import time
import mediapipe as mp
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
import collections
from scipy.spatial.distance import pdist
import txt_reader

#Load the model
model = load_model('My_model_landmarks_final2.h5')
#Predcitions dictionary
labels_to_char = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
7:'H',8:'I',9:'K',10:'L',11:'M',12:'N',
13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',
20:'V',21:'W',22:'X',23:'Y',24:'Z',25:'space',26:'nothing'}

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


last_prediction_time = None
predictions=[]
output_text = ""
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

            #Draw hand landmarks
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    
            landmarks = np.array([[landmark.x,landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
            distances = pdist(landmarks,'euclidean')
            #Normalize landmarks
            # landmarks = landmarks - np.mean(landmarks,axis=0)
            #landmarks = landmarks/np.std(landmarks,axis=0)
            #RESHAPE
            landmarks = np.reshape(landmarks,(1,21,3))
            pred = model.predict(distances.reshape(1,-1), verbose=0)
            label = np.argmax(pred)
            predictions.append(label)
            last_prediction_time = time.time()
            if(last_prediction_time is not None and time.time() - last_prediction_time > 2):
                predictions.clear()
                output_text+=" "   
            if len(predictions) == 30:
                most_common = collections.Counter(predictions).most_common(1)
                print(labels_to_char[most_common[0][0]])
                output_text+=labels_to_char[most_common[0][0]]
                predictions.clear()

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

print(output_text)
txt_reader.read_string(output_text)
video.release()
cv2.destroyAllWindows()
