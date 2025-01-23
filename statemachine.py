# state machine
from transitions import Machine
import cv2
import BackgroundSubtraction as bgsub


class ObjectTracking(object):
    """
    A state machine for object tracking using background subtraction.

    Attributes:
        device (int or str): The video capture device or file path.
        subtractor_bg: The background subtractor object.
        nxt_transition (str): The next transition trigger.
        terminate_flag (bool): Flag to terminate the state machine.
        output_image: The current image selected by the state machine.
        previous_output_image: The backup image selected by the state machine.
        image_available_flag (bool): Flag indicating if an image is available.
        workspace_blocked (bool): Flag indicating if the workspace is blocked.
        show_debug (bool): Flag to show debugging images.
        frame_count (int): Frame counter for debugging images.
        machine: The state machine object.
    """

    # Defining states, following the image acquisition requirements
    states = [
        "Start",
        "Configuration",
        "Monitoring",
        "Detect_object",
        "Workplace_blocked",
        "Tracking_objects",
        "Object_position",
        "Validation_time",
        "Take_image",
        "Object_extraction",
        "Stop",
    ]

    def __init__(self, device=0, show_debug=False):
        """
        Initialize the ObjectTracking state machine.

        Args:
            device (int or str): The video capture device or file path.
            show_debug (bool): Flag to show debugging images.
        """
        self.device = device  # By default, look for an installed camera
        # on the PC

        self.subtractor_bg = None  # Handler for BackgroundSubtractor

        self.nxt_transition = "trigger_initialize"  # First transition name

        self.terminate_flag = False  # Flag to terminate the execution of
        # this state machine

        self.output_image = None  # Current image selected by the state machine

        self.previous_output_image = None  # Backup image

        self.image_available_flag = False  # Flag for image availability

        self.workspace_blocked = False  # Flag for workspace blocked

        self.show_debug = show_debug  # Flag to turn on all debugging figures

        self.frame_count = 0  # Frame counter for showing debugging images

        # Initialize the state machine with a pseudo-state.
        # Convention will be states starting with a capital letter, and
        # transitions will have the trigger_ prefix
        self.machine = Machine(
            model=self,
            states=ObjectTracking.states,
            initial="Start"
        )

        # Defining the transitions:
        # Unconditional transition at Start
        self.machine.add_transition(
            trigger="trigger_initialize",
            source="Start",
            dest="Configuration"
        )

        # At Configuration state, configure the system for
        # capturing images. When configuration ends, it checks if there is
        # any object at the workplace. If not, transition to Monitoring
        # state. Also, this trigger transitions to Monitoring state from
        # the following states: Object_extraction, Tracking_objects, and
        # Monitoring.
        self.machine.add_transition(
            trigger="trigger_workplace_ready",
            source=["Configuration",
                    "Monitoring",
                    "Tracking_objects",
                    "Object_extraction"],
            dest="Monitoring",
        )

        # Else, if there is any object in the workplace, transition to
        # state Workplace_blocked. Also, the trigger_object_blocking
        # transitions to Workplace_blocked state from the following
        # states: Detect_object and Workplace_blocked.
        self.machine.add_transition(
            trigger="trigger_object_blocking",
            source=["Configuration",
                    "Detect_object",
                    "Workplace_blocked"],
            dest="Workplace_blocked",
        )

        # At Workplace_blocked state, wait until there is any movement detected
        # that may extract the object from the workplace. If detected,
        # transition to state Detect_object
        self.machine.add_transition(
            trigger="trigger_extraction_movement",
            source=["Workplace_blocked",
                    "Detect_object"],
            dest="Detect_object",
        )

        # At Detect_object state, verify if there is any contour detected in
        # the workplace. If any contour is detected, trigger_object_blocking.
        # If not, transition to Configuration state to reset system
        # configuration settings.
        self.machine.add_transition(
            trigger="trigger_workplace_free",
            source="Detect_object",
            dest="Configuration",
        )

        # At the "Monitoring" state, it will capture images continuously, and
        # change to state "Detect_object" if any object enters the scene,
        # and keeps moving.
        self.machine.add_transition(
            trigger="trigger_movement_detected",
            source=[
                "Monitoring",
                "Object_position",
                "Validation_time",
                "Tracking_objects",
            ],
            dest="Tracking_objects",
        )

        # At the Tracking_objects state, it will analyze if the movement in
        # the scene stops, if there is an object contour identified
        # and its bounding box has no intersection with the image border region
        self.machine.add_transition(
            trigger="trigger_object_stopped",
            source=["Tracking_objects",
                    "Object_position"],
            dest="Object_position",
        )

        # At the Object_position state, any object detected near the image
        # borders (2% of image width or height in pixels) will halt the
        # object detection. Only when no objects are at the image borders,
        # the trigger_object_centered is enabled and transitions to the
        # state Validation_time. If any movement is detected while in
        # this state, transition to the Tracking_objects state.
        self.machine.add_transition(
            trigger="trigger_object_centered",
            source="Object_position",
            dest="Validation_time",
        )

        # At the Validation_time state, the system waits for 0.5 seconds of
        # no movement at the image borders, in order to validate that the
        # detected object is centered and at rest. All the conditions needed
        # for acquiring an image sample are satisfied, and the transition
        # trigger_stabilization_time will take another image of the same object
        # at the same position
        self.machine.add_transition(
            trigger="trigger_stabilization_time",
            source="Validation_time",
            dest="Take_image",
        )

        # At the Take_image state, the system captures the image as valid and
        # proceeds with its transmission to the Central Server for image
        # recognition service
        self.machine.add_transition(
            trigger="trigger_image_sent",
            source="Take_image",
            dest="Object_extraction"
        )

        # At the Object_extraction state, the system waits for the removal of
        # the object from the workplace. If the image is sent, transitions to
        # the Monitoring state using trigger_workplace_ready.
        # Else, if the object remains in the workplace and is not removed
        # after a long time, the Background Subtraction algorithm starts
        # to integrate the object as a background image. In this condition,
        # after the object is removed, it remains a "shadow" in the foreground
        # image mask at the last location of the object. To correct this error,
        # it is best to reuse the state Workplace_blocked which already solves
        # a similar problem. The same situation may occur at the state
        # Object_position, and the solution will be the same
        self.machine.add_transition(
            trigger="trigger_timeout",
            source=["Object_extraction", "Object_position"],
            dest="Workplace_blocked",
        )

        # At any state, it will be possible to terminate the state machine and
        # end the execution of the program. It needs the trigger_terminate
        # transition trigger from any state to state Stop, which ends the
        # execution of the script
        self.machine.add_transition(
            trigger="trigger_terminate",
            source="*",
            dest="Stop"
        )

    # =================================================
    # State callbacks
    # =================================================
    # Defining callbacks for functions as the state machine enters each state

    def on_enter_Configuration(self):
        """
        Callback for entering the Configuration state.
        Initializes image capture and checks for objects in the workspace.
        """
        # Initialize image capture for background subtraction
        self.cap, self.subtractor_bg = bgsub.initialize_bg_sub(self.device)

        # Optional code to visualize results during development
        # Get the frame rate (FPS) of the video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # The Configuration state detects if there is an object in the
        # workspace before starting monitoring. Capture the first frame.
        ret, frame = self.cap.read()
        if not ret:
            print("Camera did not send images. Check the equipment.")

        # Preprocess the image by filtering noise and resizing.
        preproc_image = bgsub.preprocess_image(frame)

        # Check if the first frame contains contours of an object.
        # output[0] = True/False
        self.blocking_object = bgsub.is_object_at_image(preproc_image,
                                                        self.show_debug)[0]

        # Determine if the workspace is free or not and trigger the
        # appropriate transition
        if self.blocking_object:
            self.nxt_transition = "trigger_object_blocking"
        else:
            self.nxt_transition = "trigger_workplace_ready"

    def on_enter_Workplace_blocked(self):
        """
        Callback for entering the Workplace_blocked state.
        Waits for movement to remove the object from the workspace.
        """
        # The video stream started with an object in the workspace
        # Or there was a Timeout in the Object position or Object extraction
        # states. Look for movement in the video to remove the object
        # Remember that the learning rate at this stage needs to be
        # set to 0.1 to quickly forget the shadow of the removed object.
        workplace_activity = False
        while workplace_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            learning_rate = 0.1  # There is an object present in the image,
            # setting the learningRate to a high value, greater than 0.01.
            # Using parentheses in unpacking does not create a tuple!
            (valid_boxes,
             border_boxes,
             final_object_box) = bgsub.locate_object(frame, learning_rate)

            if valid_boxes or border_boxes:
                # The workplace has movement, transition to Detect_object
                workplace_activity = True
                self.nxt_transition = "trigger_extraction_movement"
            else:
                # No movement in the workspace, will not exit the loop
                # Keep the next transition as object blocking in case something
                # in pytransitions causes it to exit this state
                workplace_activity = False
                self.nxt_transition = "trigger_object_blocking"

    def on_enter_Detect_object(self):
        """
        Callback for entering the Detect_object state.
        Waits for movement to stop and checks for object contours.
        """
        # The transition to here occurs only if movement was detected. Now
        # it will wait for the movement to stop and check if there are object
        # contours in the workspace. If there are, it returns to
        # Workplace_blocked. If not, it proceeds to Configuration
        workplace_activity = True
        while workplace_activity is True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            learning_rate = 0.1  # There is an object present in the image,
            # setting the learningRate to a high value, greater than 0.01.
            # Using parentheses in unpacking does not create a tuple!
            (valid_boxes,
             border_boxes,
             final_object_box) = bgsub.locate_object(frame, learning_rate)

            # Wait for the movement to stop in the image, either because the
            # object is no longer in the scene, or because a Timeout error
            # occurs
            if not valid_boxes or not border_boxes:
                # Workspace without movement (pixel in the foreground mask),
                # check if there is an object by the contour in the image
                preproc_img = bgsub.preprocess_image(frame)
                self.blocking_object = bgsub.is_object_at_image(preproc_img)[0]

                if self.blocking_object is True:
                    # No movement in the workplace and the object remains.
                    # Return to Workplace_blocked state
                    workplace_activity = False
                    self.nxt_transition = "trigger_object_blocking"
                else:
                    # No object contour. Proceed to the Configuration state
                    workplace_activity = False
                    self.nxt_transition = "trigger_workplace_free"

            else:
                # There is movement in the workspace, will not exit the loop
                # Keep the transition as trigger_extraction_movement in case
                # something in pytransitions causes it to exit this state
                workplace_activity = True
                self.nxt_transition = "trigger_extraction_movement"

    # Note: the same frame detected in Monitoring state must be used up to the
    # state Object_extraction? No need, as the same background subtractor
    # object is used in all states, and this object maintains the frame history
    # to calculate pixel differences

    def on_enter_Monitoring(self):
        """
        Callback for entering the Monitoring state.
        Waits for movement in the workspace.
        """
        # Workspace without an object. Wait until there is movement
        workplace_activity = False
        while workplace_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            # Using parentheses in unpacking does not create a tuple!
            (valid_boxes, border_boxes, _) = bgsub.locate_object(frame)

            if valid_boxes or border_boxes:
                # The workplace has movement, transition to Tracking_objects
                workplace_activity = True
                self.nxt_transition = "trigger_movement_detected"
            else:
                # No movement in the workspace, will not exit the loop
                # Keep the next transition as object blocking in case something
                # in pytransitions causes it to exit this state
                workplace_activity = False
                self.nxt_transition = "trigger_workplace_ready"

    def on_enter_Tracking_objects(self):
        """
        Callback for entering the Tracking_objects state.
        Tracks if movement has stopped and checks for object contours.
        """
        # Track if the movement in the scene has stopped.
        # Look for any object contours. If there are, proceed to
        # Object_position state. If not, return to Monitoring state.
        # Does not differentiate between objects at the edges or in the center
        # of the frames.
        workplace_has_activity = True
        first_frame = True
        # Must have a minimum loop to compare at least two frames and
        # assert if the object found is still in the scene.
        while workplace_has_activity is True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            (valid_boxes, border_boxes, _) = bgsub.locate_object(frame)
            # Default learning rate (0.0001)

            # Movement occurs and tracks if the movement stops. Then
            # check if the movement has stopped by looking for contours

            # If it is the first execution of the movement search
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # Never ever back here again...
                # Cannot leave this state yet. It does not mean the
                # state is left then reentered again...

            else:
                # Combine all found boxes to facilitate the comparison
                # between masks by concatenating the lists.
                # It is expected that union_boxes is never empty at this point
                # in the code, because it is testing the recent end of a
                # movement detected by the Background Subtractor.
                union_boxes = valid_boxes + border_boxes
                intersect = cv2.bitwise_and(past_all_boxes, union_boxes)
                iou = (cv2.countNonZero(intersect) /
                       cv2.countNonZero(union_boxes))

                if iou > 0.9:  # Intersection over Union
                    # More than 90% overlap between the boxes to allow for
                    # noise in the foreground masks
                    print("Object stopped or no object")

                    # Confirm if there is an object in the workplace by
                    # looking forobject contours in the frame.
                    # Preprocess the image.
                    preproc = bgsub.preprocess_image(frame)
                    object_ok = bgsub.is_object_at_image(preproc)[0]
                    if object_ok is True:
                        # There is an object in the workplace, without
                        # movements
                        self.nxt_transition = "trigger_object_stopped"
                        self.blocking_object = True
                        workplace_has_activity = False
                    else:
                        # No object in the workplace, return to Monitoring
                        self.nxt_transition = "trigger_workplace_ready"
                        workplace_has_activity = False
                        self.blocking_object = False

                else:
                    print("Movement detected")
                    workplace_has_activity = True
                    self.nxt_transition = "trigger_movement_detected"

    def on_enter_Object_position(self):
        """
        Callback for entering the Object_position state.
        Evaluates if the object is centered and away from image margins.
        """
        # Evaluates if the object is centered and away from the image margins
        # If the object is not centered, it will wait for movement activity
        # to transition back to the Tracking_objects state. If it takes too
        # long, and the object persists in the scene and is considered as
        # background by the MOG2 algorithm, and a Timeout event occurs
        # In this case, the transition will be to the Workplace_blocked state.
        # Finally, if the object is centered, it will trigger the transition
        # to the Validation_time state.
        workplace_move_activity = False
        first_frame = True

        while workplace_move_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            (valid_boxes, border_boxes, _) = bgsub.locate_object(frame)
            # Default learning rate (0.0001)
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # Never ever back here again...

            if valid_boxes and not border_boxes:  # Trigger transition to
                # Validation_time? Confirm if the object is centered by
                # finding contours in the frame, and not touching margins.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # Find in image for contours of objects
                mask_height, mask_width = frame.shape
                size_border_factor = 0.01  # Borders of 10 pixels (1%)
                margin_top_bot = int(mask_height * size_border_factor)
                margin_left_right = int(mask_width * size_border_factor)
                # Make this border as a percentage, 1% of the H or W dimension

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if any contour is at the image borders
                    top_left = ((x > margin_left_right)
                                and
                                (y > margin_top_bot))

                    top_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 (y > margin_top_bot))

                    bot_left = ((x > margin_left_right)
                                and
                                ((y + h) < (mask_height - margin_top_bot)))

                    bot_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 ((y + h) < (mask_height - margin_top_bot)))

                    image_outside_borders = (
                        top_left and top_right and bot_left and bot_right
                    )
                    # False if any corner inside borders

                    if image_outside_borders is True:
                        # All found contours are valid ones
                        # No timeout event triggers this branch
                        self.nxt_transition = "trigger_object_centered"
                        workplace_move_activity = True
                    else:
                        # Any found contours are border ones
                        # Timeout event triggers this branch.
                        # When timeout occurs, the object mask fades away
                        # gradually, and the border contours may disappear
                        # before the valid ones.
                        self.nxt_transition = "trigger_timeout"
                        workplace_move_activity = True
                        break

            elif not valid_boxes and not border_boxes:  # There was an object
                # Reaches here only if Timeout. There is another state to
                # track movement (else clause), to avoid reaching here if
                # the object is repositioned.

                self.nxt_transition = "trigger_timeout"

            else:  # Border_boxes found, considered as movement for removing
                # object from the workplace. May have valid_boxes too, but
                # they are not considered in this state. The object is not
                # centered yet. It will be hard to differentiate any part of
                # the object inside image margins from moving objects. This
                # is simplified as considering the presence of any
                # border_boxes as movement.
                self.nxt_transition = "trigger_movement_detected"
                workplace_move_activity = True

    def on_enter_Validation_time(self):
        """
        Callback for entering the Validation_time state.
        Waits for 0.5 seconds to validate that the object is centered.
        """
        # Wait for 0.5 seconds to validate that the object is centered
        frames_to_wait = int(self.fps / 2)  # FPS times 0.5 seconds

        while frames_to_wait > 0:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            frames_to_wait -= 1
            (_, border_boxes, _) = bgsub.locate_object(frame)

            if border_boxes:
                # Movement detected, return to Tracking_objects state
                self.nxt_transition = "trigger_movement_detected"
                break
            # The contours at this stage are only valid ones. If any
            # border_box is detected, the transition will be to the
            # Tracking_objects state. Otherwise, the transition will be to
            # Take_image.

        # Validation time is over, the object is centered and immobile and no
        # occlusion is detected. Transition to Take_image state
        self.nxt_transition = "trigger_stabilization_time"
        self.output_image = frame

    def on_enter_Take_image(self):
        """
        Callback for entering the Take_image state.
        Captures the image of the object and sets the image available flag.
        """
        # Capture the image of the object and send it to the central server
        ret, frame = self.cap.read()
        if not ret:
            print("Camera did not send images. Check the equipment.")
        else:
            self.output_image = frame
            self.image_available_flag = True
            self.nxt_transition = "trigger_image_sent"

    def on_enter_Object_extraction(self):
        """
        Callback for entering the Object_extraction state.
        Waits for the removal of the object from the workspace.
        """
        # Wait for the removal of the object from the workspace. If the object
        # is removed, transition to Monitoring. If the object is not removed
        # after a long time, the Background Subtraction algorithm starts
        # to integrate the object as a background image. In this case, after
        # the object is removed, a "shadow" remains in the foreground mask
        # at the last location of the object. To correct this error, it is best
        # to reuse the Workplace_blocked state which already solves a similar
        # problem.

        workplace_move_activity = False
        first_frame = True

        while workplace_move_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                break

            (valid_boxes, border_boxes, _) = bgsub.locate_object(frame)
            # Default learning rate (0.0001)
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # Never ever back here again...

            if valid_boxes and not border_boxes:  # Trigger transition to
                # Validation_time? Confirm if the object is centered by
                # finding contours in the frame, and not touching margins.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # Find in image for contours of objects
                mask_height, mask_width = frame.shape
                size_border_factor = 0.01  # Borders of 10 pixels (1%)
                margin_top_bot = int(mask_height * size_border_factor)
                margin_left_right = int(mask_width * size_border_factor)
                # Make this border as a percentage, 1% of the H or W dimension

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if any contour is at the image borders
                    top_left = ((x > margin_left_right)
                                and
                                (y > margin_top_bot))

                    top_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 (y > margin_top_bot))

                    bot_left = ((x > margin_left_right)
                                and
                                ((y + h) < (mask_height - margin_top_bot)))

                    bot_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 ((y + h) < (mask_height - margin_top_bot)))

                    image_outside_borders = (
                        top_left and top_right and bot_left and bot_right
                    )
                    # False if any corner inside borders

                    if image_outside_borders is True:
                        # All found contours are valid ones
                        # No timeout event triggers this branch
                        self.nxt_transition = "trigger_object_centered"
                        workplace_move_activity = True
                    else:
                        # Any found contours are border ones
                        # Timeout event triggers this branch.
                        # When timeout occurs, the object mask fades away
                        # gradually, and the border contours may disappear
                        # before the valid ones.
                        self.nxt_transition = "trigger_timeout"
                        workplace_move_activity = True
                        break

    def start_object_tracking(self):
        """
        Starts the object tracking state machine.
        Cycles through states and triggers transitions dynamically.
        """
        # This function will cycle through the states. Each state entry
        # triggers an on_enter_<<state>> callback as soon as the transition
        # ends. Use the .trigger('next_state') method, where the next state is
        # defined dynamically
        # Outputs are the captured image flag and the selected image.
        # If the image is unavailable, the flag is false
        # If the image is available, the flag is true
        # If the image is read, reset the captured image flag
        while self.terminate_flag is not True:
            print(f"{self.state}")
            # How to listen to any event that may trigger the terminate flag?
            self.trigger(self.nxt_transition)

    def get_image(self):
        """
        Returns the captured image if available.

        Returns:
            The captured image or None if no image is available.
        """
        # Returns the captured image, if available
        if self.image_available_flag:
            # Reset the captured image flag
            self.image_available_flag = False

            # Store the output image
            output = self.output_image
            # Store a backup of the image
            self.previous_output_image = self.output_image.copy()

            return output
        else:
            # No image available
            return None


if __name__ == "__main__":
    # Original videos recorded in 4k. Reduce resolution for the copy used
    # to calculate contours? Re-encoded videos to Full HD to test
    # the algorithms, then use the original 4k videos

    folder = "video\\videosMock\\"

    # Complete videos for final module testing
    # Simple video
    # cap = cv2.VideoCapture(r"video\teste.mp4")

    # Complex video with various adverse conditions
    # cap = cv2.VideoCapture(r"video\video.mp4")

    # Generic objects
    object1 = "cabo luz off.mp4"
    object2 = "cabo movimento maos luz on.mp4"
    object3 = "caixa clara movimento maos luz on.mp4"
    object4 = "caixa desde inicio luz on.mp4"
    object5 = "caixa luz off.mp4"
    object6 = "caixa mudanca iluminacao.mp4"
    object7 = "Paquimetro luz off.mp4"
    object8 = "Paquimetro mao presente luz off.mp4"
    object9 = "Paquimetro para caixa luz off.mp4"
    object10 = "Regua luz off.mp4"
    object11 = "regua refletiva luz on.mp4"

    # Mock instrument
    object12 = "BaixaIluminacao100luxSombraForte.mp4"
    object13 = "TrocaObjetosAutofocoAtrapalha.mp4"
    object14 = "Iluminacao800_560lux.mp4"
    object15 = "Objeto15segs.mp4"
    object16 = "Objeto15segSubstituido.mp4"
    object17 = "objeto3segs.mp4"
    object18 = "ObjetoInicio.mp4"
    object19 = "ObjetoOrtogonalDiagonal.mp4"
    object20 = "ObjetoReposicionado.mp4"
    object21 = "OclusãoMão.mp4"
    object22 = "OclusãoTempMão.mp4"

    # - To generate HSV values of the background
    object23 = "TemperaturaCor3k_9k.mp4"

    # - HSV values of the background
    object24 = "ContrasteTemperaturaCor3k_9k.mp4"

    # Select the object video
    object = object17

    # Consider in the final version that the device should be a camera
    # Initialize image capture
    device = folder + object
    # If using the main camera, comment the previous line and uncomment
    # the following line
    # device = 0

    supervisor = ObjectTracking(device, True)

    print(supervisor.state)  # To follow what is the current state
    supervisor.start_object_tracking()
    # supervisor.trigger_initialize
    # print(supervisor.state)
