import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from threading import Thread
from playsound import playsound

engine = pyttsx3.init()
engine.setProperty('rate', 150)
is_voice_on = True

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# def say_text(text):
# 	if not is_voice_on:
# 		return
# 	while engine._inLoop:
# 		pass
# 	engine.say(text)
# 	engine.runAndWait()

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Best of luck', 1: 'I Love You', 2: 'Unite', 3: '', 4: 'Toilet please!', 5: 'zero', 6: 'one', 7: 'two', 8: 'three', 9: 'four', 10: 'five', 11: 'six', 12: 'seven', 13: 'eight', 14: 'nine', 15: 'Make a call', 16: 'I', 17: 'You', 18: 'Need water', 19: 'stop!', 20: 'Need food'}

count_same_frames = 0
predicted_character = ""
word = ""
old_text=""
audio = []

# def speak_call(word):
#     Thread(target=say_text, args=(word, )).start()
#     word=""
    


while True:
    old_predicted_character = predicted_character

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        if old_predicted_character == predicted_character:
            count_same_frames += 1
        else:
            count_same_frames = 0    
    
    if count_same_frames > 50:       
        # Thread(target=say_text, args=(predicted_character, )).start()
        # count_same_frames = 0

        if(predicted_character == 'stop!'):
            # speak_call(word)            
            for i in audio:
                playsound(i) 
            audio.clear()
            count_same_frames = 0
            word = ""
        # Create a black image
        image = np.zeros((400, 600, 3), dtype=np.uint8)

        # Put the text on the image
        text = predicted_character
        if(text != old_text and predicted_character!="stop!"):
            word = word + " " + predicted_character    
            audio.append("sound\\" + predicted_character + ".wav")          
        old_text = text

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White color in BGR
        text_position = (50, 200)

        cv2.putText(image, word, text_position, font, font_size, text_color, font_thickness)

        # Display the image in a separate dialog box
        cv2.imshow('Text Dialog Box', image)         
      
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
