import cv2
import time
import os
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import mediapipe as mp
import PIL
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard


org = 'ISL'
for i in os.listdir(org):
    img_lst = os.path.join(org,i)
    print(f"Name = {i}")
    for num in os.listdir(img_lst):
        num_images = len(os.listdir(os.path.join(img_lst,num)))
        print(f"images in {num} = {num_images}")
    print("\n\n")

actions = os.listdir('ISL')
actions = ['are you free today','are you hiding something','bring water for me','can i help you','comb your hair']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    
def extract_keypoints(results):
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])

no_sequences = 30

for action in actions:
    img_lst = os.path.join("ISL",action)
    for lst in os.listdir(img_lst):
        final_lst = os.path.join(img_lst,lst)
        cnt=0
        pth = os.path.join("ISL_data_2",action,lst)
        os.makedirs(pth)
        for image in range(no_sequences):
            with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.8) as hands:
                img = cv2.imread(os.path.join(final_lst,f"{str(image)}.jpg"))
                print(os.path.join(final_lst,str(image)))
                results = hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                draw_landmarks(img,results)
            a = extract_keypoints(results)
            path = os.path.join(pth,str(image))
            np.save(path,a)
#             print(f'file saved as {path}')
            cnt = cnt+1

label_map = {label:num for num,label in enumerate(actions)}

no_sequences = 30
sequences , labels = [],[]
sequence_length = [1,2,3,4,5,6,7]
for action in actions:
    for seq in sequence_length:
        window = []
        for frame_num in range(no_sequences):
            res = np.load(os.path.join("ISL_data",action,str(seq),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
print(x.shape)
y = to_categorical(labels).astype(int)
print(y.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64,return_sequences=True,activation="relu",input_shape=(30,126)))
model.add(LSTM(128,return_sequences=True,activation="relu"))
# model.add(LSTM(256,return_sequences=True,activation="relu"))
# model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=True,activation="relu"))
model.add(LSTM(64,return_sequences=False,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(5,activation="softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=['categorical_accuracy'])
model.fit(x,y,epochs=500,callbacks = [tb_callback])

model.summary()

sequence=[]
sentence=[]
threshold = 0.8

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while(True):
        ret, frame = cap.read()
        image,results = mediapipe_detection(frame,holistic)
        
        draw_landmarks(image,results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
#         sequence.insert(0,keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])
        
            if res[np.argmax(res)] > threshold:
                if len(sentence) >0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 2:
                sentence = sentence[-2:]
            
        cv2.rectangle(image, (0,0),(640,40),(245,117,16),-1)
        cv2.putText(image, ' | '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
#         cv2.putText(img=image, text=actions[np.argmax(res)], fontFace = cv2.FONT_HERSHEY_SIMPLEX, org=(40,70),thickness=2,fontScale = 1.5,color = (255, 0, 0))
        
        cv2.imshow('Sign Language Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

model.save('model2')