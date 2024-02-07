import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
SHAPE=(800,600,3)

class Hands_detection():
   
    def __init__(self, model_path) -> None:
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path),
                                                running_mode=VisionRunningMode.LIVE_STREAM,
                                                result_callback=self.get_results,
                                                num_hands=2,
                                                min_hand_detection_confidence=0.1,
                                                min_hand_presence_confidence=0.3,
                                                min_tracking_confidence=0.3)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.x=[]
        self.y=[]
        self.z=[]

    def get_results(self, detection_result, rgb_image, _):
        hand_landmarks_list = detection_result.hand_landmarks
        rgb_image = rgb_image.numpy_view()
        self.x=[]
        self.y=[]
        self.z=[]
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            # hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            # solutions.drawing_utils.draw_landmarks(
            #         self.annotated_image,
            #         hand_landmarks_proto,
            #         solutions.hands.HAND_CONNECTIONS,
            #         solutions.drawing_styles.get_default_hand_landmarks_style(),
            #         solutions.drawing_styles.get_default_hand_connections_style())

            height, width, _ = rgb_image.shape
            x = [int(landmark.x*width) for landmark in hand_landmarks]
            y = [int(landmark.y*height) for landmark in hand_landmarks]
            z = [int(landmark.z*width) for landmark in hand_landmarks]
            self.x.append(x)
            self.y.append(y)
            self.z.append(z)

    def detect(self,image):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timestamp = int(round(time.time()*1000))
            self.detector.detect_async(mp_image, timestamp)