import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                max_num_hands=self.maxHands,
                                min_detection_confidence=self.detectionCon,
                                min_tracking_confidence=self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Use custom drawing style for better visibility
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                               self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2))
        return img

    def findPosition(self, img, handNo=0, draw=True, circle_radius=5, circle_color=(255, 0, 255), z_axis=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks): # Safety check
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    if not z_axis:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    else:
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                        lmList.append([id, cx, cy, cz])

                    if draw:
                        # Highlight important landmarks with different colors
                        if id in [4, 8, 12, 16, 20]: # Fingertips
                            cv2.circle(img, (cx, cy), circle_radius + 2, (0, 255, 0), cv2.FILLED)
                        elif id in [0, 5, 9, 13, 17]: # Base of fingers and wrist
                            cv2.circle(img, (cx, cy), circle_radius, (255, 0, 0), cv2.FILLED)
                        else:
                            cv2.circle(img, (cx, cy), circle_radius, circle_color, cv2.FILLED)

        return lmList

    def fingersUp(self, lmList):
        """
        Determine which fingers are up based on hand landmarks
        Returns a list of 5 values (0 or 1) for [thumb, index, middle, ring, pinky]
        """
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        
        if len(lmList) < 21:
            return [0, 0, 0, 0, 0]
        
        # Thumb (special case - check x coordinate)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check y coordinate)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def findDistance(self, p1, p2, img, lmList, draw=True):
        """
        Find distance between two landmarks
        """
        if len(lmList) > max(p1, p2):
            x1, y1 = lmList[p1][1], lmList[p1][2]
            x2, y2 = lmList[p2][1], lmList[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if draw:
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            return length, img, [x1, y1, x2, y2, cx, cy]
        
        return 0, img, []

if __name__ == "__main__":
    pTime = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    detector = handDetector(maxHands=1)
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera. Exiting.")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Show finger count
            fingers = detector.fingersUp(lmList)
            totalFingers = fingers.count(1)
            cv2.putText(img, f'Fingers: {totalFingers}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / ((cTime - pTime) + 1e-6) # Add a small epsilon to avoid ZeroDivisionError
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

