import cv2
import numpy as np
import BackgroundSubtractionV2 as bgsub
from transitions import Machine
from transitions.extensions import HierarchicalMachine
from transitions.extensions.nesting import NestedState

# NestedState.separator = '►'  # Alt+16 parent►child
# Caracteres unicode não pode estar em nome de identificadores python!

class SurgicalInstrumentTrackDetect(object):
    """
    A state machine for object tracking using background subtraction.

    Attributes:
        # device (int or str): The video capture device or file path.
        subtractor_bg: The background subtractor object.
        nxt_transition (str): The next transition trigger.
        terminate_flag (bool): Flag to terminate the state machine.
        output_image: The current image selected by the state machine.
        # previous_output_image: The backup image selected by the state machine.
        image_available_flag (bool): Flag indicating if an image is available.
        workspace_blocked (bool): Flag indicating if the workspace is blocked.
        show_debug (bool): Flag to show debugging images.
        frame_count (int): Frame counter for debugging images.
        machine: The state machine object.
    """

    # Defining states, following the image acquisition requirements
    # using the hierarchical state model
    states = ['stop', 'start',
              {'name': 'operational',
              'initial': 'configuration',
              'children':[
                         'configuration',
                         'error',
                         'takeImage',
                         {'name': 'monitoring',
                          'initial': 'workplaceFree',
                          'children':[
                              'workplaceFree',
                              'tracking',
                              'centered'
                          ]
                         }
                        ] 
              }
             ]
    
    # Defining transitions
    transitions = [
        # Transitions Hierarchical level 0

        # Unconditional initialize, from start to operational.
        # Start is a pseudo-state, waiting for the signal of
        # trigger_initialize, after the image is ready for 
        # analysis
        ['trigger_initialize', 'start', 'operational'],

        # Unconditional terminate, from any state/substate
        ['trigger_terminate', 'operational', 'stop'],

        # Transitions Hierarchical level 1

        # If any object is in the workplace, transition
        # to Error state, and reconfigure system
        ['trigger_workplaceBlocked',
         'operational_configuration', 'operational_error'],

        # Reflexive transition while error is not cleared
        ['reflexive_error',
         'operational_error', '='],

        # If no contour is detected in the workplace, transition to
        # configuration state to reset system settings.
        ['trigger_emptyWorkplace',
         'operational_error', 'operational_configuration'],

        # When configuration ends, if no object is at the workplace,
        # transition to Monitoring state
        ['trigger_workplaceReady',
         'operational_configuration', 'operational_monitoring'],

        # If an object remains in the workplace after a long time,
        # the Background Subtraction algorithm integrate it as background.
        # In this condition, after the object is removed, it remains a 
        # "shadow" in the image mask at the last location of the object.
        ['trigger_timeout',
         'operational_monitoring', 'operational_error'],

        # At the takeImage state, the system captures and make the image
        # available for transmission to the image recognition Server.
        ['trigger_imageSent',
         'operational_takeImage', 'operational_monitoring'],

        # Transitions Hierarchical level 2
        # Reflexive transition
        ['reflexive_workplaceFree',
         'operational_monitoring_workplaceFree', '='],

        # At the monitoring state, it will capture images, and change to the
        # state tracking if any object enters the scene, and keeps moving.
        ['trigger_movement_detected',
         'operational_monitoring_workplaceFree',
         'operational_monitoring_tracking'],
        
        # At the tracking state, if no object is in the scene, fallback to
        # the workplaceFree state.
        ['trigger_noObject',
         'operational_monitoring_tracking',
         'operational_monitoring_workplaceFree'],

        # If the object is present, check if it is away from borders and if
        # it has no movement in the scene at the centered state.
        ['trigger_objectStopped',
         'operational_monitoring_tracking',
         'operational_monitoring_centered'],

        # If the object is reaching the margin region near the image borders,
        # it is not considered at rest, going back to the tracking state.
        ['trigger_readjustPosition',
         'operational_monitoring_centered',
         'operational_monitoring_tracking'],

        # The object is centered and immobile, so take the current frame and
        # make it available to the image recognition service.
        ['trigger_imageOk',
         'operational_monitoring_centered',
         'operational_takeImage'],
    ]

    def __init__(self, show_debug=False):
        """
        Initialize the ObjectTracking state machine. Define attributes and
        state transitions

        Args:
            show_debug (bool): Flag to show debugging images.
        """

        self.show_debug = show_debug  # Flag to turn on all debugging figures

        self.subtractor_bg = None  # Handler for BackgroundSubtractor

        self.nxt_transition = None # placeholder for transition name

        self.terminate_flag = False  # Flag to terminate the execution of
        # this state machine

        self.output_image = None  # selected image output by the state machine

        self.received_image = None  # last image input for analysis

        self.image_available_flag = False  # Flag for image availability

        self.workplace_activity = False  # Flag for movement in the workplace

        # Initialize the state machine with a pseudo-state.
        # Convention will be states starting with a capital letter, and
        # transitions will have the trigger_ prefix
        self.machine = HierarchicalMachine(
            model=self,
            states=SurgicalInstrumentTrackDetect.states,
            transitions=SurgicalInstrumentTrackDetect.transitions,
            initial="start",
            ignore_invalid_triggers=True,
            auto_transitions=False
        )

    def is_frame_error(self):
        if self.received_image is None:
            print("Camera did not send images. Check the equipment.")
            # Terminate at Stop state
            self.nxt_transition = "trigger_terminate"
            return True
        return False
    

    def show_debug_frame(self, frame):
        if self.show_debug is True:
            debugframe = cv2.resize(frame, None, fx=0.3, fy=0.3,
                                    interpolation=cv2.INTER_NEAREST)
            cv2.imshow('debug', debugframe)
            print('Frame correto')

    # =================================================
    # State callbacks
    # =================================================
    # Defining callbacks for functions as the state machine enters each state

    # The initial state, "start", does not trigger the on_enter_<state>
    # method!

    def on_enter_operational_configuration(self):
        """
        Callback for entering the configuration state.
        Initializes resources.
        """
        # Initialize image capture for background subtraction 
        self.subtractor_bg = bgsub.initialize_bg_sub()

        # The Configuration state detects if there is an object in the
        # workspace before starting monitoring.
        frame = self.received_image
        # if frame is None:
        #     print("Camera did not send images. Check the equipment.")
        #     # Terminate at Stop state
        #     self.nxt_transition = "trigger_terminate"
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)

            # Preprocess the image by filtering noise and resizing.
            preproc_image = bgsub.preprocess_image(frame)

            # Check if the first frame contains contours of an object.
            self.blocking_object = bgsub.is_object_at_image(preproc_image,
                                                            self.show_debug)[0]

            # Determine if the workspace is free or not and trigger the
            # appropriate transition
            if self.blocking_object:
                self.nxt_transition = "trigger_workplaceBlocked"
                print("Object detected in the workspace")
            else:
                self.nxt_transition = "trigger_workplaceReady"
                print("Workspace is free")

            # Guarantee the output at every state
            self.image_available_flag = False
            self.output_image = None

    def on_enter_operational_error(self):
        """
        Callback for entering the error state.
        Waits for removal of the object from the workspace.
        """
        # REFACTOR: Coalesces old Workspace_Blocked and Detect_Object states

        # The input frames started with an object in the workspace,
        # or there was a Timeout event in the Monitoring superstate.
        # Look for movement in the video to remove the object
        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)

        learning_rate = 0.1  # Set the learning rate at this stage needs to be
        # set to 0.1 to quickly forget the shadow of the removed object.
        # Aprox. 10 frames to clear the background model

        # Using parentheses in unpacking does not create a tuple!
        # Check if there is an object by the contour in the image
        preproc_img = bgsub.preprocess_image(frame)

        (valid_boxes,
         border_boxes,
         _) = bgsub.locate_object(self.subtractor_bg, frame, learning_rate)
        
        if self.workplace_activity is False:
            # preferir forma if len(valid_boxes) > 0 or len(border_boxes) > 0:?
            if valid_boxes or border_boxes:
                # The workplace has movement, look for no object next iteration
                self.workplace_activity = True
            else:
                # No movement in the workspace.
                self.workplace_activity = False
            self.nxt_transition = "reflexive_error"  # For firing this method again
        else:
        # Check if there are object contours in the workspace. If yes, it returns to
        # workplace_activity = False. If not, it proceeds to Configuration
        # This section works after a reflexive_error transition is called
            if not valid_boxes or not border_boxes:
                # Workspace without movement (pixel in the foreground mask),
                # check if there is an object by the contour in the image
                preproc_img = bgsub.preprocess_image(frame)
                self.blocking_object = bgsub.is_object_at_image(preproc_img)[0]

                if self.blocking_object is True:
                    # No movement in the workplace and the object remains.
                    # Return to workplace_activity = False
                    self.workplace_activity = False
                    self.nxt_transition = "reflexive_error"
                else:
                    # No object contour. Proceed to the Configuration state
                    self.workplace_activity = False
                    self.nxt_transition = "trigger_emptyWorkplace"

            else:
                # There is movement in the workspace, will not transition yet.
                self.workplace_activity = True
                self.nxt_transition = "reflexive_error"

        # Guarantee the output at every state
        self.image_available_flag = False
        self.output_image = None

    def on_enter_operational_monitoring_workplaceFree(self):
        """
        Callback for entering the workplaceFree state.
        Waits for movement in the workspace.
        """
        # Workspace without an object. Wait until there is movement
        #workplace_activity = False
        # while workplace_activity is not True:
        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)

            # Using parentheses in unpacking does not create a tuple!
            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)

            if valid_boxes or border_boxes:
                # The workplace has movement, transition to tracking state
                self.nxt_transition = "trigger_movement_detected"
            else:
                # No movement in the workspace, will not exit this state
                self.nxt_transition = "reflexive_workplaceFree"
        # Guarantee the output at every state
        self.image_available_flag = False
        self.output_image = None

    def on_enter_operational_monitoring_tracking(self):
        """
        Callback for entering the tracking state.
        Tracks if movement has stopped and checks for object contours.
        """
        # Track if the movement in the scene has stopped.
        # Look for any object contours. If there is any one, proceed to
        # centered state. If not, return to workspaceFree state.
        # Does not differentiate between objects at the edges or in the center
        # of the frames.
        self.workplace_activity = True
        first_frame = True

        frame_count_no_change = 0
        # Must have a minimum loop to compare at least three (3) frames and
        # assert if the object found is still in the scene.
        #while workplace_has_activity is True:

        # TODO: Adicionar contador como atributo de quantos frames idênticos
        # com objeto identificado pelo MOG2
        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)

            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)
            # Default learning rate (0.0001)

            # Movement occurs and tracks if the movement stops. Then
            # start looking for contours

            # If it is the first execution of the movement search
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes

                last_x, last_y, last_w, last_h = 0, 0, 0, 0
                first_frame = False  # Never ever back here again...
                # Cannot leave this state yet. It does not mean the
                # state is left then reentered again...

            else:
                # Combine all found boxes to facilitate the comparison
                # between masks by concatenating the lists.
                # It is expected that union_boxes is never empty at this point
                # in the code, because it is testing the recent end of a
                # movement detected by the Background Subtractor.

                # Define the size of the mask, as a cropped frame, for the
                # objects moving in the workplace.
                mask_MOG2 = np.zeros_like(bgsub.preprocess_image(frame))

                # Mask for the objects in the frame. If moving, the blur will
                # diminish the size of the object comparing to the MOG2 mask.
                mask_object = np.zeros_like(bgsub.preprocess_image(frame))

                # Find the contours of the objects in the frame, 2nd position
                # of the return tuple
                instant_contours = bgsub.is_object_at_image(
                                        bgsub.preprocess_image(frame)
                                   )[1]

                # Compare the position of all combined boxes in the scene
                # from the locate_object() and the mask_object. If the
                # intersection over union is greater than 0.9, the object
                # is considered stopped. If not, the object is still moving.

                if instant_contours:
                    x, y, w, h = cv2.boundingRect(np.vstack(instant_contours))
                    cv2.rectangle(mask_object, (x, y), (x + w, y + h), 255,
                                  thickness=cv2.FILLED)

                    for box in past_all_boxes:
                        cv2.rectangle(mask_MOG2, (box[0], box[1]),
                                      (box[0] + box[2], box[1] + box[3]),
                                      255, thickness=cv2.FILLED)

                    intersect = cv2.bitwise_and(mask_MOG2, mask_object)
                    union = cv2.bitwise_or(mask_MOG2, mask_object)

                    intersection_area = np.sum(intersect > 0)
                    union_area = np.sum(union > 0)
                    iou = intersection_area / union_area

                    # mask_MOG2_3ch = cv2.cvtColor(mask_MOG2,
                    # cv2.COLOR_GRAY2BGR)
                    # mask_object_3ch = cv2.cvtColor(mask_object,
                    # cv2.COLOR_GRAY2BGR)

                    # cv2.imshow("MOG2 mask", mask_MOG2_3ch)
                    # cv2.imshow("Object mask", mask_object_3ch)
                    #  cv2.waitKey(0)

                    past_all_boxes = valid_boxes + border_boxes

                    if iou > 0.9:  # Intersection over Union

                        # It is needed to count frames, 5 frames without
                        # modification in the size of the bounding rectangle
                        # coordinates.
                        # The pixel count may oscillate due to noise in the
                        # foreground mask. The object is considered stopped
                        # if the pixel number of the MOG2 mask is within 95%
                        # of the total pixel number of the previous rectangle.
                        if (last_x, last_y, last_w, last_h) == (x, y, w, h):
                            frame_count_no_change += 1  # Wait for 5 frames
                        else:
                            frame_count_no_change = 0  # object is moving
                            last_x, last_y, last_w, last_h = x, y, w, h

                        if frame_count_no_change >= 5:
                            # The object is considered stopped
                            # Evaluate iou is meaningful now
                            print("Object stopped or no object")

                            # Confirm if there is an object in the workplace by
                            # looking forobject contours in the frame.
                            # Preprocess the image.
                            preproc = bgsub.preprocess_image(frame)
                            object_ok = bgsub.is_object_at_image(preproc)[0]

                            if object_ok is True:
                                # There is an object in the workplace, without
                                # movements. Transition to Object_position
                                self.nxt_transition = "trigger_object_stopped"
                                self.blocking_object = True
                                workplace_has_activity = False
                            else:
                                # No object in the workplace, return to
                                # Monitoring
                                self.nxt_transition = "trigger_workplace_ready"
                                workplace_has_activity = False
                                self.blocking_object = False

                    else:
                        print("Movement detected")
                        workplace_has_activity = False
                        self.nxt_transition = "trigger_movement_detected"
        # Guarantee the output at every state
        self.image_available_flag = False
        self.output_image = None

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
        # first_frame = True

        while workplace_move_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                # Keep locked at Stop state
                self.nxt_transition = "trigger_terminate"
                break

            if self.show_debug is True:
                debugframe = cv2.resize(frame, None, fx=0.3, fy=0.3,
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imshow('debug', debugframe)

            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)
            # Default learning rate (0.0001)
            # if first_frame is True:
            #     past_all_boxes = valid_boxes + border_boxes
            #     first_frame = False  # Never ever back here again...

            if valid_boxes and not border_boxes:  # Trigger transition to
                # Validation_time? Confirm if the object is centered by
                # finding contours in the frame, and not touching margins.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # Find in image for contours of objects
                mask_height, mask_width = preproc.shape
                size_border_factor = 0.01  # Borders of ~10 pixels (1%)
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
                # Keep locked at Stop state
                self.nxt_transition = "trigger_terminate"
                break

            if self.show_debug is True:
                debugframe = cv2.resize(frame, None, fx=0.3, fy=0.3,
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imshow('debug', debugframe)

            frames_to_wait -= 1
            (_,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)

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
            # Keep locked at Stop state
            self.nxt_transition = "trigger_terminate"
        else:
            self.output_image = frame
            self.image_available_flag = True
            self.nxt_transition = "trigger_image_sent"
            print("Image captured")
            resized = cv2.resize(frame, (640, 480))
            cv2.imshow("captured", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
        # first_frame = True
        movement_detected_flag = False

        while workplace_move_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera did not send images. Check the equipment.")
                # Create a way to handle this type of error
                # Interrupt the state machine execution with a terminate
                # transition to the stop state.
                # Keep locked at Stop state
                self.nxt_transition = "trigger_terminate"
                break

            if self.show_debug is True:
                debugframe = cv2.resize(frame, None, fx=0.3, fy=0.3,
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imshow('debug', debugframe)

            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)
            # Default learning rate (0.0001)

            # It is still possible to have only valid_boxes at this stage.
            # The time from the object stabilization to the beginning of
            # this state will be less than the Timeout.

            if border_boxes and (movement_detected_flag is False):
                # movement detected
                movement_detected_flag = True
                last_valid_boxes = valid_boxes
                # now identify if the
                # movement ceases and there is no object in the scene.
                # the object may be repositioned and touch the borders
                # of the image. The object is not considered as removed
                # from the workplace yet.
            else:  # no more movement detected due to lack of activity
                # at the image borders. Now, the instrument has changed
                # its position, or it was completely removed from the scene.
                # it must be compared the current valid_boxes with the
                # previous ones. If there is any change, consider the object
                # as repositioned, and trigger the transition to the
                # Tracking_object state.
                # If there is no valid_boxes, test the Timeout event, too.

                # Confirm if the object is removed by
                # not finding contours in the frame.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # if the movement_detected_flag is False and the object_ok
                # is true, there is a Timeout situation event. The object
                # was not removed from the workplace, and the Background
                # Subtraction algorithm starts to integrate the object as
                # a background image.
                if not movement_detected_flag and object_ok:
                    # Timeout event triggers this branch
                    self.nxt_transition = "trigger_timeout"
                    workplace_move_activity = True

                elif object_ok and movement_detected_flag:
                    # Compare the current valid_boxes with the previous ones
                    # to check if the object was repositioned.
                    if last_valid_boxes == valid_boxes:
                        # Movement did not change the position of the object.
                        # It won't be considered as a new object in the scene.
                        # Trigger the transition to the Object_extraction state
                        # again, to wait for the object to be removed.
                        workplace_move_activity = False
                        self.nxt_transition = "trigger_object_blocking"
                    else:
                        # The object was repositioned. Trigger the transition
                        # to the Tracking_objects state.
                        workplace_move_activity = True
                        self.nxt_transition = "trigger_movement_detected"
                else:
                    # if object_ok is False and movement_detected_flag is True
                    # The object was removed from the workplace. Transition to
                    # Monitoring state.
                    workplace_move_activity = True
                    self.nxt_transition = "trigger_workplace_ready"

    def on_enter_stop(self):
        """
        Callback for entering the Stop state.
        Terminates the state machine.
        """
        # Terminate the state machine
        self.terminate_flag = True  # Flag to terminate the state machine
        print("State machine terminated")
        # Keep locked at Stop state
        self.nxt_transition = "trigger_terminate"

    def object_tracking(self, picture):
        """
        Starts the object tracking state machine at every image input.
        Cycles through states and triggers transitions dynamically.

        Returns:
            list: A list containing the video captured frame or None and
            a flag for the resulting object tracking and detection .
        """
        # This function will cycle through the states. Each state entry
        # triggers an on_enter_<<state>> callback as soon as the transition
        # ends. Start the sequence of state transitions with an unconditional
        # trigger_initialize after the input image was received
        # Outputs are the captured image flag and the selected image.
        # If the image is unavailable, the flag is false
        # If the image is available, the flag is true
        # If the image is read, reset the captured image flag
        if self.terminate_flag is not True:
            try:
                image_data = np.frombuffer(picture, dtype=np.uint8)
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                self.received_image = frame  # Saving the received image
            except Exception as e:
                print(f'Ocorreu erro ao receber imagem:/n{e}')
                self.image_available_flag = False
                self.output_image = None
                return self.image_available_flag, self.get_image()
            # Displaying which is the state machine current state during debug
            if self.show_debug:
                print(f"Initial state: {self.state}")

            # As the input picture is available, start the state machine
            if self.state == "start":
                self.trigger_initialize()
                print(f"Initial state: {self.state}")
            else:
            # When the trigger is called, it executes the on_enter_<state>?
                self.trigger(self.nxt_transition)
            
            if self.show_debug:
                print(f"Final state: {self.state}") 

            if self.state == "stop":
                # decomission the object
                pass
                # TODO: Finish this!
            return self.image_available_flag, self.get_image()
#             if self.state == "Stop":
#                 output = self.get_image()

#                 if output is not None:
#                     print("Image available")
#                     cv2.imshow("Output image", output)
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()
#                 else:
#                     print("No image available")

#                 # break
#             elif self.state == "Object_extraction":
#                 output = self.get_image()

#                 if output is not None:
#                     print("Image available. Click the image and press any key \
# to continue")
#                     cv2.imshow("Output image", output)
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()
#                 else:
#                     print("No image available")

#                 print("Object extraction state reached and image captured.")
#                 print("Analyze the video for an another object detection?")
#                 next_round = input('(Y/N):')
#                 if next_round.upper() == 'Y':
#                     self.trigger("trigger_image_sent")
#                 else:
#                     self.trigger("trigger_terminate")

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
            #self.previous_output_image = self.output_image.copy()

            self.output_image = None  # Reset the output image

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

    videosamples = [  # Generic objects
                    "cabo luz off.mp4",
                    "caixa clara movimento maos luz on.mp4",
                    "caixa desde inicio luz on.mp4",
                    "caixa luz off.mp4",
                    "caixa mudanca iluminacao.mp4",
                    "Paquimetro luz off.mp4",
                    "Paquimetro mao presente luz off.mp4",
                    "Paquimetro para caixa luz off.mp4",
                    "Regua luz off.mp4",
                    "regua refletiva luz on.mp4",
                    # Mock instrument
                    "BaixaIluminacao100luxSombraForte.mp4",
                    "TrocaObjetosAutofocoAtrapalha.mp4",
                    "Iluminacao800_560lux.mp4",
                    "Objeto15segs.mp4",
                    "Objeto15segSubstituido.mp4",
                    "objeto3segs.mp4",
                    "ObjetoInicio.mp4",
                    "ObjetoOrtogonalDiagonal.mp4",
                    "ObjetoReposicionado.mp4",
                    "OclusãoMão.mp4",
                    "OclusãoTempMão.mp4"
                    ]

#     # - To generate HSV values of the background
#     # object23 = "TemperaturaCor3k_9k.mp4"

#     # - HSV values of the background
#     # object24 = "ContrasteTemperaturaCor3k_9k.mp4"

#     # Select the object video
#     # object = object17

#     # Consider in the final version that the device should be a camera
#     # Initialize image capture
#     # device = folder + videosamples
#     # If using the main camera, comment the previous line and uncomment
#     # the following line
#     # device = 0
    user_choosing = True
    while user_choosing is True:
        try:
            # Ask user for a number corresponding to a video and 0 to camera
            for index, video in enumerate(videosamples, start=1):
                print(f'Video {index} - {video}')
            escolha = int(input(f"Choose a number between 1 \
and {len(videosamples)}. Choose 0 for live camera feed: "))

            # Verifica se o número está dentro do intervalo válido
            if 1 <= escolha <= len(videosamples):
                # Obtém o nome do arquivo correspondente
                arquivo_escolhido = videosamples[escolha - 1]
                caminho_completo = folder + arquivo_escolhido

                # Exibe o nome do arquivo e o caminho completo
                print(f"You chose: {arquivo_escolhido}")
                print(f"Relative path: {caminho_completo}")
                break
            else:
                print("Number outside range. Try again.")
        except ValueError:
            print("Invalid input. Please, insert an integer number.")

    device = caminho_completo
    camera = cv2.VideoCapture(device)
    supervisor = SurgicalInstrumentTrackDetect(True)

    while supervisor.terminate_flag is not True:  # Enquanto a statemachine não encerra
        ret, frame = camera.read()
        if not ret:
            print("Camera did not send images. Check the equipment.")
            # Terminate the state machine
            supervisor.trigger_terminate()
        # imagem corretamente capturada, e deve ser convertida para um arquivo tipo .jpeg, mas
        # sem armazenar em disco
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            # Converte o buffer para bytes, se necessário
            jpeg_bytes = buffer.tobytes()
            print("Frame convertido para JPEG em memória.")
        else:
            print("Falha na codificação JPEG.")

        # Chamar a rotina da máquina de estados que tem o jpeg_bytes como input e
        # uma lista [flag, imagem\None]

        flag, image = supervisor.object_tracking(jpeg_bytes)
#     print('Test program for object detection state machine.')

#     print('Press any key to start the state machine.')
#     _ = input()
#     print('State machine started.')
#     print('Current state: ', supervisor.state)

#     supervisor.start_object_tracking()
#     # supervisor.trigger_initialize
#     # print(supervisor.state)

    # Segunda versão do CLI de teste
    # Os frames do vídeo serão lidos externamente e enviados individualmente para
    # a máquina de estado.
    # O método que invoca a máquina deve receber a imagem e uma flag de debug
    # Se a flag de debug estiver ligada, mostra o estado atual da máquina
