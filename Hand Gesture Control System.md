# Hand Gesture Control System

This project implements a real-time hand gesture control system that allows users to interact with their computer using natural hand movements captured via a webcam. It leverages a combination of MediaPipe for precise hand landmark detection and a custom-trained Roboflow model for accurate gesture recognition. The system features a mode-based interface, enabling distinct control functionalities for mouse cursor manipulation, scrolling, and system volume adjustment, all managed through intuitive hand gestures.

The core idea is to provide an alternative, hands-free method of computer interaction, which can be particularly useful for accessibility, presentations, or simply for a novel user experience. The project is designed with modularity in mind, making it easy to understand, extend, and adapt to new gestures or functionalities.

## Features

*   **Real-time Hand Tracking:** Utilizes MediaPipe to detect and track hand landmarks with high accuracy.
*   **Custom Gesture Recognition:** Employs a Roboflow-trained object detection model to classify various hand gestures.
*   **Mode-Based Control:** Switches between different control modes (Cursor, Scroll, Volume) using specific activation gestures.
    *   **Cursor Mode:** Control mouse movement and perform clicks.
    *   **Scroll Mode:** Scroll up and down in applications.
    *   **Volume Mode:** Adjust system volume dynamically based on hand position, with fine-tuning options.
*   **Universal Deactivation:** A dedicated 'fist' gesture allows for immediate exit from any active control mode, returning to a neutral state.
*   **Screenshot Functionality:** A 'Victory' gesture can be used at any time to capture a screenshot of the desktop.
*   **Customizable:** Includes scripts (`data_collection.py`, `train_model.py`) to collect new gesture data and train a custom machine learning model, allowing users to define their own gestures and commands.
*   **Visual Feedback:** Displays detected gestures, active mode, and frames per second (FPS) directly on the live camera feed for an enhanced user experience and debugging.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+:** The project is developed using Python. You can download it from [python.org](https://www.python.org/downloads/).
*   **pip:** Python's package installer, usually comes with Python installation.
*   **Webcam:** A functional webcam is required for hand tracking and gesture recognition.
*   **Roboflow Account & Model (Optional):** The primary `Main_improved.py` script is configured to use a Roboflow model for gesture recognition. If you wish to use this online model, you will need a Roboflow account and a trained object detection model for hand gestures. Alternatively, you can train and use a local model as described in the "Lightweight System Option" section.

### Installation

1.  **Clone the repository (or download the files):**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your actual repo
    cd your-repo-name
    ```

    If you downloaded the files directly, navigate to the directory where you saved them.

2.  **Install dependencies:**

    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

    **`requirements.txt` content:**

    ```
    opencv-python
    mediapipe
    pyautogui
    roboflow
    numpy
    pandas
    scikit-learn
    comtypes # For Windows volume control
    pycaw # For Windows volume control
    ```

    *Note: `comtypes` and `pycaw` are specific to Windows for system volume control. If you are on Linux or macOS, these might not install or function, but the rest of the application will still work (volume control will be disabled with a warning).* 

3.  **Roboflow API Key and Model ID Configuration (if using Roboflow model):**

    If you intend to use the Roboflow-powered gesture recognition, open `Main_improved.py` and ensure the `ROBOFLOW_MODEL_ID` and `ROBOFLOW_API_KEY` variables are set correctly. The current values are:

    ```python
    ROBOFLOW_MODEL_ID = "my-first-project-zyx4y/4" # Your project ID and version
    ROBOFLOW_API_KEY = "y12xyElzxsxb8Psc8zyo" # Your Roboflow API Key
    ```

    **Important:** Replace `your-api-key` and `your-model-id/version` with your actual Roboflow credentials if you are using a different model. If you plan to use the lightweight local model, you can skip this step and proceed to the "Lightweight System Option" section.

## Usage

### Running the Main Application

To run the hand gesture control system, execute the `Main_improved.py` script:

```bash
python Main_improved.py
```

This will open a window displaying your webcam feed. As you perform the defined gestures, the system will interpret them and execute the corresponding computer commands.

### Modularity and Independent Operation

The project is designed with modularity in mind. The `Main_improved.py` script is the core application that orchestrates hand tracking, gesture recognition, and computer control. It relies on `HandTrackingModule_improved.py` for MediaPipe-based hand landmark detection. These two files can run together to provide the real-time gesture control functionality.

`data_collection.py` and `train_model.py` are separate utilities provided for training your own custom gesture recognition models. They are not required for the main application to run if you are using a pre-trained Roboflow model.

### Gesture to Task Mapping

The system operates in different modes, activated by specific gestures. A universal deactivation gesture (`fist`) allows you to return to the Neutral State from any active mode.

| Gesture (Roboflow Class) | Neutral State Action | Cursor Mode Action | Scroll Mode Action | Volume Mode Action |
| :----------------------- | :------------------- | :----------------- | :----------------- | :----------------- |
| `fist`                   | N/A                  | Deactivate Mode    | Deactivate Mode    | Deactivate Mode    |
| `L`                      | Activate Cursor Mode | N/A                | N/A                | N/A                |
| `point_finger`           | Activate Scroll Mode | N/A                | N/A                | N/A                |
| `open_palm`              | Activate Volume Mode | N/A                | N/A                | N/A                |
| `ok`                     | N/A                  | Mouse Click        | N/A                | N/A                |
| `Thumbs_up`              | N/A                  | N/A                | Scroll Up          | Increase Volume (Fine-tune) |
| `Thumbs_down`            | N/A                  | N/A                | Scroll Down        | Decrease Volume (Fine-tune) |
| `Victory`                | Take Screenshot      | Take Screenshot    | Take Screenshot    | Take Screenshot    |

*Note: In Volume Mode, the distance between your thumb and index finger dynamically controls the volume level. `Thumbs_up` and `Thumbs_down` provide fine-tuning adjustments.*

## Lightweight System Option: Training Your Own Custom Gestures Locally

If the Roboflow model is too heavy, or if you prefer an entirely offline and lightweight solution for gesture recognition, you can train your own custom machine learning model using MediaPipe landmarks and scikit-learn. This approach offers reduced latency and resource consumption, making it suitable for less powerful hardware or environments with limited internet connectivity.

This project provides a complete pipeline for this purpose:

### 1. Data Collection (`data_collection.py`)

This script helps you capture hand landmark data from your webcam and save it into a CSV file, along with the corresponding gesture labels.

1.  **Run the data collection script:**

    ```bash
    python data_collection.py
    ```

2.  **Follow the on-screen prompts:**

    *   The script will open your webcam feed.
    *   It will prompt you to enter a label for the gesture you are about to perform (e.g., `fist`, `open_hand`, `pointing_up`).
    *   Perform the gesture in front of the camera.
    *   Press `s` to save the current frame's hand landmark data with the entered label. Collect multiple samples for each gesture (e.g., 50-100 samples per gesture).
    *   To change the gesture label, simply type the new label when prompted.
    *   Press `q` to quit the data collection process.

    The collected data will be saved in `gesture_data/gesture_data.csv`. Each row will contain 42 normalized landmark coordinates (x, y for 21 landmarks) and the corresponding gesture label.

### 2. Model Training (`train_model.py`)

Once you have collected sufficient data, you can use the `train_model.py` script to train a RandomForestClassifier model that can recognize your custom gestures.

1.  **Ensure your data is ready:** The `train_model.py` script expects the data to be in a CSV file named `final_product.csv` in the same directory. If your collected data is in `gesture_data/gesture_data.csv`, you might need to copy or rename it to `final_product.csv` or modify the `data_file_path` variable in `train_model.py`.

2.  **Run the training script:**

    ```bash
    python train_model.py
    ```

3.  **Review the output:** The script will train a RandomForestClassifier, evaluate its performance (accuracy and classification report), and then save the trained model as `gesture_model.pkl`.

    This `gesture_model.pkl` file can then be loaded into your `Main_improved.py` script to replace the Roboflow model.

### 3. Integrating the Local Model into `Main_improved.py`

To switch from using the Roboflow model to your locally trained `gesture_model.pkl`:

1.  **Comment out Roboflow initialization** in `Main_improved.py`:

    ```python
    # rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    # project = rf.workspace().project(ROBOFLOW_MODEL_ID.split('/')[0])
    # model = project.version(ROBOFLOW_MODEL_ID.split('/')[1]).model
    ```

2.  **Load your local model** in `Main_improved.py` (add this near the other imports):

    ```python
    import pickle
    # ... other imports ...

    # Load the locally trained model
    try:
        with open('gesture_model.pkl', 'rb') as f:
            local_gesture_model = pickle.load(f)
        logging.info("Successfully loaded local gesture_model.pkl")
    except FileNotFoundError:
        logging.error("Error: gesture_model.pkl not found. Please train your model first.")
        exit()
    except Exception as e:
        logging.error(f"Error loading local model: {e}")
        exit()
    ```

3.  **Modify `get_gesture_status`** to use the local model for prediction. This function will now take `lmList` (MediaPipe landmarks) as input instead of `predictions` from Roboflow:

    ```python
    def get_gesture_status(lmList):
        if not lmList or len(lmList) < 21: # Ensure enough landmarks are detected
            return 'none'

        # Extract and normalize landmarks similar to data_collection.py
        # Assuming lmList is already in the format [[id, x, y], ...]
        # Need to convert it to the normalized flattened format expected by the model
        
        # Dummy wrist for normalization (using lmList[0] as wrist)
        wrist_x = lmList[0][1]
        wrist_y = lmList[0][2]

        normalized_landmarks = []
        for lm_id, x, y in lmList:
            normalized_landmarks.append(x - wrist_x)
            normalized_landmarks.append(y - wrist_y)
        
        # Reshape for model prediction
        input_data = np.array(normalized_landmarks).reshape(1, -1)

        try:
            prediction = local_gesture_model.predict(input_data)[0]
            # You might want to add a confidence threshold here if your model provides probabilities
            return prediction
        except Exception as e:
            logging.error(f"Error during local model prediction: {e}")
            return 'none'
    ```

4.  **Adjust the call to `get_gesture_status`** in the main loop of `Main_improved.py`:

    Change:
    `current_gesture = get_gesture_status(roboflow_predictions)`
    To:
    `current_gesture = get_gesture_status(lmList)`












