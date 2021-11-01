import cv2
import mediapipe as mp

class Hand_Detector:
    def __init__(self, mode = False, max_hands = 2, detection_con = 0.5, track_con = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                        self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw = True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

        return frame

    
    def find_position(self, frame, hand_no = 0, draw = True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lnd in enumerate(my_hand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lnd.x*w), int(lnd.y*h)
                    lm_list.append([id, cx, cy])
        return lm_list
                    
                


def run_program():

    capture = cv2.VideoCapture(0)
    detector = Hand_Detector()

    while True:

        _, frame = capture.read()
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        if len(lm_list) != 0:
            print(lm_list[8])

        cv2.imshow("Capture",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_program()