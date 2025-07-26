import sys
import cv2
import time
import math
import numpy as np
import HandTrackingModule_improved as htm
import pyautogui
import logging
from roboflow import Roboflow
from inference.core.interfaces.camera.entities import VideoFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dummy classes for volume control if pycaw is not available
class DummyVolume:
    def GetVolumeRange(self): return (-63.5, 0.0, 0.5)
    def SetMasterVolumeLevel(self, level, guid): pass
class DummyAudioUtilities:
    def GetSpeakers(self): return None

# Import pycaw only if on Windows
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    windows_os = True
except (ImportError, OSError):
    logging.warning("Pycaw not available. Volume control will be disabled.")
    windows_os = False
    AudioUtilities = DummyAudioUtilities()
    IAudioEndpointVolume = None


# Webcam dimensions
wCam, hCam = 640, 480

# Allow user to specify camera index (e.g., via command line argument or config file)
camera_index = 0 # Default camera index
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    logging.error(f"Error: Could not open camera at index {camera_index}.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0 # Previous time for FPS calculation

# Hand detector object (MediaPipe, still used for drawing landmarks)
detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# Roboflow Model ID and API Key - UPDATED WITH YOUR NEW PROJECT ID AND API KEY
ROBOFLOW_MODEL_ID = "my-first-project-zyx4y/4" # Updated to version 4
ROBOFLOW_API_KEY = "y12xyElzxsxb8Psc8zyo" # Your provided API key

# Initialize Roboflow
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_MODEL_ID.split('/')[0])
    model = project.version(ROBOFLOW_MODEL_ID.split('/')[1]).model
except Exception as e:
    logging.error(f"Failed to initialize Roboflow. Please check your API key and model ID. Error: {e}")
    exit()


# Volume control setup (Windows specific)
if windows_os:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()  # e.g., (-63.5, 0.0, 0.5)
    minVol = volRange[0]
    maxVol = volRange[1]
else:
    volume = DummyVolume()
    minVol = -63.5
    maxVol = 0.0

# Volume bar display variables
hmin = 50
hmax = 200
volBar = 400
volPer = 0
vol = 0
color = (0, 215, 255)

# Finger tip IDs from MediaPipe landmarks (still used for some calculations if needed)
tipIds = [4, 8, 12, 16, 20]

# Mode variables for different controls
mode = 'N' # 'N' for None, 'S' for Scroll, 'V' for Volume, 'C' for Cursor
active = 0 # To prevent continuous activation of a mode

# Variables for smoother cursor movement
prev_x, prev_y = 0, 0
alpha = 0.2 # Smoothing factor (0.0 to 1.0, lower is smoother)

# Variables for mode change visual feedback
mode_change_message = ""
mode_change_time = 0
MESSAGE_DISPLAY_DURATION = 1.5 # seconds

pyautogui.FAILSAFE = True # Re-enable failsafe for pyautogui

def putText(img, text, loc, color=(0, 255, 255)):
    cv2.putText(img, str(text), loc, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                3, color, 3)

# Modified get_gesture_status to use Roboflow predictions
def get_gesture_status(predictions):
    if not predictions:
        return 'none'
    highest_confidence_gesture = 'none'
    max_confidence = 0
    for p in predictions:
        if p["confidence"] > max_confidence:
            max_confidence = p["confidence"]
            highest_confidence_gesture = p["class"]
    return highest_confidence_gesture

# Function to handle scroll control
def handle_scroll_control(img, lmList, gesture):
    global active, mode
    putText(img, 'Scroll', (250, 450))
    cv2.rectangle(img, (200, 410), (245, 460), (255, 255, 255), cv2.FILLED)

    if gesture == 'Thumbs_up':
        putText(img, 'U', loc=(200, 455), color=(0, 255, 0))
        pyautogui.scroll(300)
        logging.info("Scrolling Up")
    elif gesture == 'Thumbs_down':
        putText(img, 'D', loc=(200, 455), color=(0, 0, 255))
        pyautogui.scroll(-300)
        logging.info("Scrolling Down")

# Function to handle volume control
def handle_volume_control(img, lmList, gesture):
    global active, mode, volBar, volPer, vol
    putText(img, 'Volume', (250, 450))
    if len(lmList) >= 9: # Ensure thumb and index finger landmarks are available
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), color, 3) # Corrected to draw line between fingertips
        cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [hmin, hmax], [minVol, maxVol])
        volBar = np.interp(vol, [minVol, maxVol], [400, 150])
        volPer = np.interp(vol, [minVol, maxVol], [0, 100])

        if windows_os:
            volume.SetMasterVolumeLevel(vol, None)
        logging.info(f"Setting Volume: {int(volPer)}%")

        # Dynamic volume bar color
        if volPer < 30:
            vol_color = (0, 255, 0) # Green for low volume
        elif volPer < 70:
            vol_color = (0, 255, 255) # Yellow for medium volume
        else:
            vol_color = (0, 0, 255) # Red for high volume

        if length < 50:
            cv2.circle(img, (cx, cy), 11, (0, 0, 255), cv2.FILLED)

        cv2.rectangle(img, (30, 150), (55, 400), (209, 206, 0), 3)
        cv2.rectangle(img, (30, int(volBar)), (55, 400), vol_color, cv2.FILLED) # Use dynamic color
        cv2.putText(img, f'{int(volPer)}%', (25, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (209, 206, 0), 3)

# Function to handle cursor control
def handle_cursor_control(img, lmList, gesture):
    global active, mode, prev_x, prev_y
    putText(img, 'Cursor', (250, 450))
    cv2.rectangle(img, (110, 20), (620, 350), (255, 255, 255), 3)

    if len(lmList) >= 9: # Ensure index finger and thumb landmarks are available
        x1, y1 = lmList[8][1], lmList[8][2]
        w, h = pyautogui.size()

        # Map index finger position to screen coordinates
        target_X = int(np.interp(x1, [110, 620], [0, w - 1]))
        target_Y = int(np.interp(y1, [20, 350], [0, h - 1]))

        # Apply smoothing filter
        if prev_x == 0 and prev_y == 0: # Initialize on first frame
            prev_x, prev_y = target_X, target_Y
        
        smoothed_X = int(prev_x * (1 - alpha) + target_X * alpha)
        smoothed_Y = int(prev_y * (1 - alpha) + target_Y * alpha)

        pyautogui.moveTo(smoothed_X, smoothed_Y)
        logging.info(f"Moving cursor to: ({smoothed_X}, {smoothed_Y})")

        prev_x, prev_y = smoothed_X, smoothed_Y # Update previous coordinates

        cv2.circle(img, (lmList[8][1], lmList[8][2]), 7, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (lmList[4][1], lmList[4][2]), 10, (0, 255, 0), cv2.FILLED)

        if gesture == 'ok':
            cv2.circle(img, (lmList[4][1], lmList[4][2]), 10, (0, 0, 255), cv2.FILLED)
            pyautogui.click()
            logging.info("Clicking")
            time.sleep(0.2)

while True:
    success, img = cap.read()
    if not success:
        logging.error("Failed to read from camera. Exiting.")
        break

    # Perform inference with Roboflow model
    results = model.predict(img, confidence=40, overlap=30).json()
    roboflow_predictions = results.get("predictions", [])

    img = detector.findHands(img) # MediaPipe for drawing landmarks
    lmList = detector.findPosition(img, draw=False)

    current_gesture = get_gesture_status(roboflow_predictions)
    putText(img, f'Gesture: {current_gesture}', (10, 30), color=(0, 255, 0))
    
    # Store previous mode for change detection
    previous_mode = mode

    # --- START OF THE FIX ---
    # PRIORITY 1: Check for the universal deactivation gesture ('fist')
    # This check now happens regardless of the current mode.
    if current_gesture == 'fist' and mode != 'N':
        active = 0
        mode = 'N'
        mode_change_message = "Neutral Mode Activated!"
        mode_change_time = time.time()
        logging.info(f"Deactivated by 'fist'. Returning to Neutral mode.")
        # Add a small delay to prevent immediate reactivation by another gesture
        time.sleep(0.5)
    
    # PRIORITY 2: If not deactivating, check for activation gestures (if in Neutral mode)
    elif active == 0: # Only activate a new mode if currently in Neutral state
        if current_gesture == 'point_finger':
            mode = 'S'
            active = 1
            mode_change_message = "Scroll Mode Activated!"
            mode_change_time = time.time()
            logging.info("Scroll mode activated by 'point_finger'")
        elif current_gesture == 'open_palm':
            mode = 'V'
            active = 1
            mode_change_message = "Volume Mode Activated!"
            mode_change_time = time.time()
            logging.info("Volume mode activated by 'open_palm'")
        elif current_gesture == 'L':
            mode = 'C'
            active = 1
            mode_change_message = "Cursor Mode Activated!"
            mode_change_time = time.time()
            logging.info("Cursor mode activated by 'L' gesture")
        elif current_gesture == 'Victory':
            pyautogui.screenshot("screenshot_" + time.strftime("%Y%m%d-%H%M%S") + ".png")
            logging.info("Screenshot taken by 'Victory' gesture")
            active = 1 # Briefly set active to 1 to prevent immediate re-triggering
            time.sleep(1) # Short delay to prevent multiple screenshots
            active = 0
        else:
            mode = 'N'

    # PRIORITY 3: If a mode is active, execute its control logic
    elif active == 1:
        if mode == 'S':
            handle_scroll_control(img, lmList, current_gesture)
        elif mode == 'V':
            handle_volume_control(img, lmList, current_gesture)
        elif mode == 'C':
            handle_cursor_control(img, lmList, current_gesture)
    # --- END OF THE FIX ---

    # Display current mode
    putText(img, f'Mode: {mode}', (10, 110), color=(255, 0, 0))

    # Display mode change message
    if time.time() - mode_change_time < MESSAGE_DISPLAY_DURATION:
        cv2.putText(img, mode_change_message, (wCam // 2 - 200, hCam // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

    # FPS calculation and display
    cTime = time.time()
    fps = 1 / ((cTime - pTime) + 1e-6)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (480, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv2.imshow('Hand LiveFeed', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


