import cv2
import mediapipe as mp
import time


class HandDetector():
    """* Initializing the HandDetector object, it uses mediapipe as it ML pipeline (Introduced by Google) * """
    def __init__(self, mode = False, max_hands = 2, model_complexity = 1, detection_confidence = 0.5, track_confidence = 0.5):
        """
        --- Parameters:
        ---- mode = If set to False, the solution treats the input images as a video stream.
                    If set to true, hand detection runs on every input image, ideal for processing a batch of static, possibly unrelated, images.
                    Default to false. (boolean)
        ---- max_hands = Maximum number of hands to detect. Default to 2 (int)
        ---- model_complexity = Complexity of the hand landmark model: 0 or 1.
                                Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1. (0 or 1)
        ---- detection_confidence = Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5.
        ---- track_confidence = Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully,
                                or otherwise hand detection will be invoked automatically on the next input image. Default to 0.5
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            self.model_complexity,
            self.detection_confidence,
            self.track_confidence
        )
        self.drawer = mp.solutions.drawing_utils
        self.draw_style = mp.solutions.drawing_styles
    
    def find_hands(self, img, draw = True):
        '''
        Finds the 21 keyoints of the hand coordinates 
        '''
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        
        if self.results.multi_hand_landmarks != None:
            for hand_lm in self.results.multi_hand_landmarks:
                if draw:
                    self.drawer.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS, connection_drawing_spec= self.draw_style.DrawingSpec(color=(0, 255, 0)))
        
        return img
    
    def find_position(self, img, hand_num = 0):
        """_returns a list of positions for an specified hand number (Default to 0)

        Args:
            img (image): find the postions base on the shape of this image
            hand_num (int, optional): determine which hand to find its positions. Defaults to 0.

        Returns:
            list: containing the different positions for the specified hand.
        """
        lm_list = list()

        if hand_num > self.max_hands:
            raise IndexError('hand_num is more than the max_hands')

        if self.results.multi_hand_landmarks != None:
            cur_hand = self.results.multi_hand_landmarks[hand_num]
            
            for id, lm in enumerate(cur_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        
        return lm_list



def main():
    p_time = 0
    c_time = 0

    capture = cv2.VideoCapture(1)
    detector = HandDetector()

    while True:
        suc, img = capture.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        
        if len(lm_list) != 0:
            print(lm_list[4])
        img = cv2.flip(img, 1)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # showing the fps on the screen
        cv2.putText(img, 'FPS: {}'.format(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Webcam Screen", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    main()