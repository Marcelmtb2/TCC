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

## V2 Files

The v2 version of the system implements a similar state machine but is designed to work in scenarios where frames are provided externally, rather than being captured directly from a camera or video file. This allows integration with external applications that handle frame acquisition and preprocessing.

### Project Structure (V2)

- **BackgroundSubtraction_v2.py**: Contains functions for processing externally provided frames, detecting objects, and identifying contours.
- **statemachine_v2.py**: Implements a state machine for object tracking using the `transitions` library. It defines states and transitions similar to the original version but expects frames to be passed in from an external source.

### BackgroundSubtraction_v2.py

This module provides the necessary functions to process externally provided frames for object tracking.

#### Functions

- **initialize_bg_sub_v2()**: Initializes the background subtractor for processing external frames.
- **config_object_capture_v2()**: Configures the background subtractor with default values for external frame processing.
- **is_object_at_image_v2(frame, show_overlay=False)**: Checks if there are contours of an object in the provided frame.
- **preprocess_image_v2(frame, show_overlay=False)**: Preprocesses the frame by filtering noise and resizing.
- **find_foreground_object_v2(frame, learning_rate=0.0001)**: Identifies objects moving relative to the background in the provided frame.
- **identify_contours_v2(object_found_binary_mask)**: Locates contours of the possible object in the binary mask.
- **locate_object_v2(frame, learning_rate=0.0001)**: Locates the object in the preprocessed frame.

### statemachine_v2.py

This module implements a state machine for object tracking using externally provided frames. It defines states and transitions similar to the original version but is adapted for frame-based input.

#### States

The states in `statemachine_v2.py` are identical to those in the original `statemachine.py` but operate on externally provided frames instead of directly controlling the camera.

#### Transitions

The transitions in `statemachine_v2.py` are also identical to those in the original `statemachine.py`, with the key difference being that they rely on frames passed in from an external source.

### Usage (V2)

1. Clone the repository.
2. Ensure you have Miniconda installed. Create and activate the environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate cme
   ```
3. Run the `statemachine_v2.py` script to start the object tracking state machine for external frame input.
   ```bash
   python statemachine_v2.py
   ```

### External Frame Input

The `statemachine_v2.py` script expects frames to be provided by an external application. The external application should send frames to the state machine in real-time or batch mode. The state machine processes each frame and performs object tracking and detection.

Example workflow:
1. The external application captures or generates frames.
2. Frames are passed to the `statemachine_v2.py` script via a predefined interface (e.g., function calls, sockets, or shared memory).
3. The state machine processes each frame and transitions between states based on the analysis of the frame.

This design allows for greater flexibility and integration with other systems that handle frame acquisition and preprocessing.