import cv2
import numpy as np
import BackgroundSubtractionV2 as bgsub
from transitions.extensions import HierarchicalMachine

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
        ['trigger_movementDetected',
         'operational_monitoring_workplaceFree',
         'operational_monitoring_tracking'],
        
        # Reflexive transition
        ['reflexive_tracking',
         'operational_monitoring_tracking', '='],

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

        self.show_debug = show_debug  # Flag bool to turn on all debugging figures

        self.subtractor_bg = None  # Handler for BackgroundSubtractor

        self.nxt_transition = None # placeholder for transition name

        self.terminate_flag = False  # Flag to terminate the execution of
        # this state machine

        self.old_mask = None  # Placeholder to the binary mask for detecting movement

        self.counter = 0  # Generic counter for state

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
            ignore_invalid_triggers=False,
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
            cv2.waitKey(10)
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
            preproc_image = bgsub.preprocess_image(frame, self.show_debug)

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
                self.nxt_transition = "trigger_movementDetected"
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
        # Supposing there is only one object. If found stopped objects, proceed to
        # centered state. If not, return to workspaceFree state.
        # Does not differentiate between objects at the edges or in the center
        # of the frames.

        # self.counter = 0
        # Must have a minimum loop to compare at least three (3) frames and
        # assert if the object found is still in the scene.

        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)
            
            preproc_image = bgsub.preprocess_image(frame, True)
            clean_mask = bgsub.find_foreground_object(self.subtractor_bg, preproc_image)
            # Default learning rate (0.0001)
            # Movement occurs and tracks if the movement stops. Then
            # start looking for contours
            if self.old_mask is None:
                self.old_mask = clean_mask
                # keep this state, in a reflexive transition
                self.nxt_transition = "reflexive_tracking"
            else:  # better to calculate Intersect over Union of sequential masks
                intersect = cv2.bitwise_and(self.old_mask, clean_mask)
                union = cv2.bitwise_or(self.old_mask, clean_mask)
                intersection_area = np.sum(intersect > 0)
                union_area = np.sum(union > 0)
                if union_area == 0:
                    # clean union mask means no object
                    self.nxt_transition = "trigger_noObject"
                else:
                    # looking for timeout event, that is a contour in the preproc_image
                    # that is not present at the clean_mask

                    # If the object is darker than the green background, in grayscale
                    thresh_dark = cv2.threshold(preproc_image, 90, 255, cv2.THRESH_BINARY_INV)[1]
                    cv2.imshow("thresh_dark", thresh_dark)
                    cv2.waitKey(10)
                    # If the object is lighter than the green background, in grayscale
                    thresh_light = cv2.threshold(preproc_image, 200, 255, cv2.THRESH_BINARY)[1]
                    cv2.imshow("thresh_light", thresh_light)
                    cv2.waitKey(10)
                    # The final mask is the bitwise OR combination of the two masks. Hysteresis
                    # tries to avoid shadows on the green sheet.
                    thresh_object = cv2.bitwise_or(thresh_dark, thresh_light)

                    # comparing with the clean mask, if the IoU
                    matching_figure = (np.sum(cv2.bitwise_and(thresh_object, clean_mask) > 0) /
                                       np.sum(thresh_object > 0))
                    
                    if matching_figure < 0.2:
                        # The mask is considering the object as background, causing a timeout
                        # event
                        self.nxt_transition = "trigger_timeout"
                        self.old_mask = None
                    else:
                        # Now, can divide by a non-zero value
                        iou_move = intersection_area / union_area
                        
                        if iou_move > 0.9:
                            self.counter += 1
                            if self.counter > 2:
                                # at least 3 masks are almost coincidental. It would be
                                # a very slow movement or a standstill object.
                                self.nxt_transition = "trigger_objectStopped"
                                self.old_mask = None
                                self.counter = 0
                            else:
                                self.nxt_transition = "reflexive_tracking"
                                self.old_mask = clean_mask
                        else:
                            # Non-coincidental masks mean relative movement between them
                            self.counter = 0
                            self.nxt_transition = "reflexive_tracking"
                            # Updating the old_mask for the next comparison
                            self.old_mask = clean_mask

        # Guarantee the output at every state
        self.image_available_flag = False
        self.output_image = None

    def on_enter_operational_monitoring_centered(self):
        """
        Callback for entering the centered state.
        Evaluates if the object is centered and away from image margins.
        """
        # Evaluates if the object is away from the image margins
        # If the object is not centered, it will transition back to the
        # tracking state. If it takes too long, a Timeout event occurs,
        # transitioning to the error state.
        # Finally, if the object is centered, it will trigger the transition
        # imageOk.

        # workplace_move_activity = False
        # first_frame = True

        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)

            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(self.subtractor_bg, frame)
            # Default learning rate (0.0001)

            if valid_boxes and not border_boxes:
                self.nxt_transition = "trigger_imageOk"
            else:
                # Any found contours are border ones
                # Timeout event triggers are not expected in this state.
                # When timeout occurs, the object mask fades away
                # gradually, and the border contours may disappear
                # before the valid ones.
                self.nxt_transition = "trigger_readjustPosition"

        # Guarantee the output at every state
        self.image_available_flag = False
        self.output_image = None

    def on_enter_operational_takeImage(self):
        """
        Callback for entering the Take_image state.
        Captures the image of the object and sets the image available flag.
        """
        # Capture the image of the object and send it to the central server
        frame = self.received_image
        if self.is_frame_error():
            return
        else:
            self.show_debug_frame(frame)
            
            self.output_image = frame
            self.image_available_flag = True
            self.nxt_transition = "trigger_image_sent"
            print("Image captured")
            resized = cv2.resize(frame, (640, 480))
            cv2.imshow("captured", resized)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

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

#================================
# Auxiliary methods
#================================

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
            # When the trigger is called, it executes the on_enter_<state>
                self.trigger(self.nxt_transition)
            
            if self.show_debug:
                print(f"Final state: {self.state}") 

            if self.state == "stop":
                # decomission the object
                pass
                # TODO: Finish this!
            return self.image_available_flag, self.get_image()

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

##============================================================
## Implementation testing
##
## Reproducing the conditions expected from CME_VISION_API calls
## The requests just send a frame captured by a camera
## There are no specific commands from server to control the state
## machine.
## Main test must convert the images to .jpeg format before calling
## the FSM with the image.
## Only the sequence of images must change the FSM State

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
        else:
            print("Number outside range. Try again.")
    except ValueError:
        print("Invalid input. Please, insert an integer number.")

## ========================================
## Simulating the CME_VISION_API
## Camera externally defined and controlled
    device = caminho_completo
    camera = cv2.VideoCapture(device)

## ========================================
## Simulating the CME_VISION_API
## Instantiating the FSM Controller Object
    supervisor = SurgicalInstrumentTrackDetect(True)


## ========================================
## Simulating the CME_VISION_API
## Calling the FSM methods
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

## ========================================
## Simulating the CME_VISION_API
## FSM calls only with the .jpeg image
## Output must be a tuple with a boolean flag and a None or Image output)

        # Chamar a rotina da máquina de estados que tem o jpeg_bytes como input e
        # uma lista [flag, imagem\None]

        flag, image = supervisor.object_tracking(jpeg_bytes)


    # Segunda versão do CLI de teste
    # Os frames do vídeo serão lidos externamente e enviados individualmente para
    # a máquina de estado.
    # O método que invoca a máquina deve receber a imagem e uma flag de debug
    # Se a flag de debug estiver ligada, mostra o estado atual da máquina
        print(f'{flag}')
        if image is not None:
            cv2.imshow('outputframe', image)
