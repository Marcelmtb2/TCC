# Object Tracking and Detection System

This project implements an object tracking and detection system using background subtraction. The system is designed to track and detect the presence of an object in a video stream within a controlled environment. It evaluates if an object is stationary, centered in the scene, and not occluded by obstacles or the hands of a technician.

## Project Structure

- **BackgroundSubtraction.py**: Contains functions for initializing the background subtractor, preprocessing images, detecting objects, and identifying contours.
- **statemachine.py**: Implements a state machine for object tracking using the `transitions` library. It defines various states and transitions for tracking objects in the video stream.

## BackgroundSubtraction.py

This module provides the necessary functions to track and detect objects in a video stream.

### Functions

- **initialize_bg_sub(device=0)**: Initializes the background subtractor and video capture device.
- **config_object_capture()**: Configures the background subtractor with default values.
- **is_object_at_image(image, show_overlay=False)**: Checks if there are contours of an object in the image.
- **preprocess_image(image, show_overlay=False)**: Preprocesses the image by filtering noise and resizing.
- **find_foreground_object(image, learning_rate=0.0001)**: Identifies objects moving relative to the background.
- **identify_contours(object_found_binary_mask)**: Locates contours of the possible object in the binary mask.
- **locate_object(image, learning_rate=0.0001)**: Locates the object in the preprocessed image.

## statemachine.py

This module implements a state machine for object tracking using background subtraction. It defines various states and transitions for tracking objects in the video stream.

### States

- **Start**: Initial state.
- **Configuration**: Configures the system for capturing images.
- **Monitoring**: Captures images continuously and detects movement.
- **Detect_object**: Verifies if there are object contours in the workspace.
- **Workplace_blocked**: Waits for movement to remove the object from the workspace.
- **Tracking_objects**: Tracks if movement has stopped and checks for object contours.
- **Object_position**: Evaluates if the object is centered and away from image margins.
- **Validation_time**: Waits for 0.5 seconds to validate that the object is centered.
- **Take_image**: Captures the image of the object.
- **Object_extraction**: Waits for the removal of the object from the workspace.
- **Stop**: Terminates the state machine.

### Transitions

- **trigger_initialize**: Transition from Start to Configuration.
- **trigger_workplace_ready**: Transition to Monitoring state from various states.
- **trigger_object_blocking**: Transition to Workplace_blocked state from various states.
- **trigger_extraction_movement**: Transition to Detect_object state from various states.
- **trigger_workplace_free**: Transition to Configuration state from Detect_object.
- **trigger_movement_detected**: Transition to Tracking_objects state from various states.
- **trigger_object_stopped**: Transition to Object_position state from various states.
- **trigger_object_centered**: Transition to Validation_time state from Object_position.
- **trigger_stabilization_time**: Transition to Take_image state from Validation_time.
- **trigger_image_sent**: Transition to Object_extraction state from Take_image.
- **trigger_timeout**: Transition to Workplace_blocked state from various states.
- **trigger_terminate**: Transition to Stop state from any state.

## Usage

1. Clone the repository.
2. Ensure you have Miniconda installed. Create and activate the environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate cme
   ```
3. There is a private directory on OneDrive with the videos used to test and validate code. Copy the two videos and the directory "videosMock" into a folder named "video" at the same directory level as the python scripts.
4. Run the `statemachine.py` script to start the object tracking state machine.
   ```bash
   python statemachine.py
   ```

## CLI Usage

The `statemachine.py` script includes a simple command-line interface (CLI) to select the video source for object tracking. When you run the script, you will be prompted to choose a video file from the list or use the live camera feed.

1. After running the script, you will see a list of available video files.
2. Enter the number corresponding to the video file you want to use, or enter `0` to use the live camera feed.
3. The state machine will start and process the selected video or camera feed.

Example:
```bash
python statemachine.py
```
```
Video 1 - cabo luz off.mp4
Video 2 - caixa clara movimento maos luz on.mp4
...
Choose a number between 1 and 22. Choose 0 for live camera feed: 1
You chose: cabo luz off.mp4
Relative path: video\videosMock\cabo luz off.mp4
```
