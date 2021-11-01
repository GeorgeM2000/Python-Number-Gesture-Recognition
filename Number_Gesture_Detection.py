import mediapipe as mp
import cv2
import numpy as np
from Hand_Tracking import Hand_Detector
import tkinter as tk
import time



def main():
    capture = cv2.VideoCapture(0)

    cam_width = 640
    cam_height = 480
    capture.set(3, cam_width)
    capture.set(4, cam_height)

    detector = Hand_Detector(detection_con=0.75)
    tip_ids = [4, 8, 12, 16, 20]
    timer_initialized = False
    old_state_status = False
    old_state = None
    new_state = None
    time_passed = False
    start_time = 0.0

    while True:

        _, frame = capture.read()
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        if len(lm_list) != 0:
            fingers = []

            # Index, Middle, Ring and Pinky fingers detection
            for id in range(1,5):
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                    fingers.append(True)
                else:
                    fingers.append(False)

            # Thumb detection 
            if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0]-2][1]:
                fingers.append(False)
            else:
                fingers.append(True)

            # Old state
            if not old_state_status:
                old_state = sum(fingers)
                old_state_status = True

            # New state
            new_state = sum(fingers)

            if new_state == old_state:

                # If the timer has not been initialized
                if not timer_initialized:
                    start_time = time.time()
                    timer_initialized = True

                # If a second has passed since the timer initialization
                if (time.time() - start_time) > 1.0 and timer_initialized:
                    time_passed = True
                    timer_initialized = False
            else:
                start_time = 0.0
                time_passed = False
                timer_initialized = False
                old_state_status = False

            # If a second has passed
            if time_passed:
                cv2.putText(frame, str(sum(fingers)), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)


        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()