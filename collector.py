import os
import cv2
import mediapipe as mp
import numpy as np

#Media pipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
padding = 50 #Padding so that whole hand fits

save_path = '/home/szramstar/Desktop/SW/ASL/SPACE'
os.makedirs(save_path,exist_ok=True)

img_counter = 0;
capturing = False

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
            # hand_img = hand_img/255
            # hand_img = np.expand_dims(hand_img,axis=0)
            cv2.rectangle(frame,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),2)
            


            if cv2.waitKey(1) % 256 == ord('c'):
                capturing = True
            if capturing and img_counter <= 100:
                img_name = "SPACE_asl_{:d}.jpg".format(img_counter)
                cv2.imwrite(os.path.join(save_path,img_name),hand_img)
                print("{} written!".format(img_name))
                img_counter += 1
            if img_counter == 101:
                capturing = False
                img_counter = 0
                break

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()