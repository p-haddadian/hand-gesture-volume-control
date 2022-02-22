import cv2
import mediapipe as mp
import time

# note 1: mediapipe only accepts RGB images and not BGR (if you are using cv2, note that in mind)

capture = cv2.VideoCapture(1)

# call the Hands solution from medipipe (for hand tracking and detection)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
drawer = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

p_time = 0

while True:
    suc, img = capture.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    
    if results.multi_hand_landmarks != None:
        for hand_lm in results.multi_hand_landmarks:
            drawer.draw_landmarks(
                img,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.DrawingSpec(color=[0, 255, 0])
            )
    
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # showing the fps on the screen
    cv2.putText(img, 'FPS: {}'.format(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Webcam Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break