import cv2

import mediapipe as mp

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
SHAPE=(800,600,3)

class Hands_detection():
   
    def __init__(self, model_path) -> None:
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        mp_hands = mp.solutions.hands
        self.detector = mp_hands.Hands()
        self.x=[]
        self.y=[]
        self.z=[]

    def detect(self,image):
            self.x=[]
            self.y=[]
            self.z=[]
            results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    x,y,z=[],[],[]
                    # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark, connections)
                    for landmark in hand_landmark.landmark:
                        x.append(landmark.x),y.append(landmark.y),z.append(landmark.z)
                    height, width, _ = image.shape
                    x = [int(landmark*width) for landmark in x]
                    y = [int(landmark*height) for landmark in y]
                    z = [int(landmark*width) for landmark in z]
                    self.x.append(x)
                    self.y.append(y)
                    self.z.append(z)