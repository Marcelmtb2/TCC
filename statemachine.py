# máquina de estados
from transitions import Machine
import cv2
import BackgroundSubtraction as bgsub


class ObjectTracking(object):

    # Defining states, following the image acquisition requirements
    states = ['Start', 'Configuration', 'Monitoring',
              'Detect_object', 'Workplace_blocked',
              'Tracking_objects', 'Object_position',
              'Validation_time', 'Take_image',
              'Object_extraction', 'Stop']

    def __init__(self, device=0, show_debug=False):

        self.device = device  # As default, look for a installed camera at PC

        self.subtractor_bg = None  # Handler for BackgroundSubtractor

        self.nxt_transition = 'trigger_initialize'  # first transition name

        self.terminate_flag = False  # flag for terminate the execution of this
        # state machine

        self.workspace_blocked = False  # flag for workspace blocked

        self.show_debug = show_debug  # flag to turn on all debugging images

        self.frame_count = 0  # frame counter for showing debugging images



        # Initialize the state machine, with a pseudo-state. Transitions
        # are added later
        # Convention will be states starting with capital letter, and
        # transitions will have trigger_ prefix
        self.machine = Machine(model=self, states=ObjectTracking.states,
                               initial='Start')

        # Defining the transitions:
        # Unconditional transition at Start
        self.machine.add_transition(trigger='trigger_initialize',
                                    source='Start',
                                    dest='Configuration')

        # At Configuration state, configure system for
        # capturing images. When configuration ends, it check if there is
        # any object at the workplace. If not, transition to Monitoring
        # state. Also, this trigger transitions to Monitoring state from
        # the following states: Object_extraction, Tracking_objects and
        # Monitoring.
        self.machine.add_transition(trigger='trigger_workplace_ready',
                                    source=['Configuration',
                                            'Monitoring',
                                            'Object_extraction'],
                                    dest='Monitoring')

        # Else, if there is any object in the workplace, transition to
        # state Workplace_blocked. Also, the trigger_object_blocking
        # transitions to Workplace_blocked state from the following
        # states: Detect_object and Workplace_blocked.
        self.machine.add_transition(trigger='trigger_object_blocking',
                                    source=['Configuration',
                                            'Detect_object',
                                            'Workplace_blocked'],
                                    dest='Workplace_blocked')

        # At Workplace_blocked state, wait until there is any movement detected
        # that may extract the object from the workplace. If detected,
        # transition to state Detect_object
        self.machine.add_transition(trigger='trigger_extraction_movement',
                                    source=['Workplace_blocked',
                                            'Detect_object'],
                                    dest='Detect_object')

        # At Detect_object state, verify if there is any contour detected in
        # the workplace. If any contour detected, trigger_object_blocking. If
        # not, transition to Configuration state to reset system
        # configuration settings.
        self.machine.add_transition(trigger='trigger_workplace_free',
                                    source='Detect_object',
                                    dest='Configuration')

        # At the "Monitoring" state, it will capture images continuously, and
        # change to state "Detect_object" if any object enters the scene,
        # and keeps moving.
        self.machine.add_transition(trigger='trigger_movement_detected',
                                    source=['Monitoring',
                                            'Object_position',
                                            'Validation_time',
                                            'Tracking_objects'],
                                    dest='Tracking_objects')

        # At the Tracking_objects state, it will analyze if the movement in
        # scene stops, if there is an object contour identified
        # and its bounding box has no intersection with image border region
        self.machine.add_transition(trigger='trigger_object_stopped',
                                    source=['Tracking_objects',
                                            'Object_position'],
                                    dest='Object_position')

        # At the Object_position state, any object detected near the image
        # borders (2% of image width or height in pixels) will halt the
        # object detection. Only when no objects are at the image borders,
        # the trigger_object_centered is enabled and transitions to the
        # state Validation_time. If any movement is detected while in
        # this state, transition to the Tracking_objects state.
        self.machine.add_transition(trigger='trigger_object_centered',
                                    source='Object_position',
                                    dest='Validation_time')

        # At the Validation_time state, the system waits for 0,5 seconds of
        # no movement at the image borders, in order to validate that the
        # detected object is centered and at rest. All the conditions needed
        # for acquiring an image sample are satisfied, and the transition
        # trigger_stabilization_timenot take another image of the same object
        # at the same position
        self.machine.add_transition(trigger='trigger_stabilization_time',
                                    source='Validation_time',
                                    dest='Take_image')

        # At the Take_image state, the system captures the image as valid and
        # proceed its transmission for the Central Server for image recognition
        # service
        self.machine.add_transition(trigger='trigger_image_sent',
                                    source='Take_image',
                                    dest='Object_extraction')

        # At the Object_extraction state, the system waits for the removal of
        # the object from the workplace. If the image is sent, transitions to
        # the Monitoring state using trigger_workplace_ready.
        # Else, if the object remains in the workplace and is not removed
        # after a long time, the Background Subtraction algorithm start
        # to integrate the object as background image. In this condition, after
        # the object is removed, it remains a "shadow" in the foreground image
        # mask at the last location of the object. For correcting this error,
        # it is best to reuse the state Workplace_blocked which already solves
        # a similar problem. The same situation may occur at the state
        # Object_position, and the solution will be the same
        self.machine.add_transition(trigger='trigger_timeout',
                                    source=['Object_extraction',
                                            'Object_position'],
                                    dest='Workplace_blocked')

        # At any state, it will be possible to terminate the state machine and
        # end the execution of the program. It need the trigger_terminate
        # transition trigger from any state to state Stop, which ends the
        # execution of the script
        self.machine.add_transition(trigger='trigger_terminate',
                                    source='*',
                                    dest='Stop')
    # =================================================
    # State callbacks
    # =================================================
    # defining callbacks for functions as the state machine enters each state

    def on_enter_Configuration(self):
        # Inicializar captura de imagens para subtração de background
        self.cap, self.subtractor_bg = bgsub.initialize_bg_sub(self.device)

        # Código opcional para visualizar resultados durante o desenvolvimento
        # Obter a taxa de quadros (FPS) do vídeo
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # O estado Configurar detecta se há objeto no espaço de trabalho
        # antes de iniciar o monitoramento. Capturar o primeiro frame.
        ret, frame = self.cap.read()
        if not ret:
            print('Câmera não enviou imagens. Conferir o equipamento.')

        # Pré-processar a imagem, filtrando ruídos e redimensionando.
        preproc_image = bgsub.preprocess_image(frame)

        # Verificar se o primeiro frame contém contornos de um objeto.
        # saída[0] =  True/False
        self.blocking_object = bgsub.is_object_at_image(preproc_image,
                                                        self.show_debug)[0]

        # Julgar se o espaço de trabalho está livre ou não e disparar a
        # transição apropriada
        if self.blocking_object:
            self.nxt_transition = 'trigger_object_blocking'
        else:
            self.nxt_transition = 'trigger_workplace_ready'

    def on_enter_Workplace_blocked(self):
        # O video stream começou com objeto no espaço de trabalho
        # Ou houve Timeout nos estados Object position ou Object extraction
        # Buscar se há movimento no vídeo para remover o objeto
        # lembrar que o learning rate nesta etapa precisa ser
        # configurado em 0.1, para esquecer rapidamente a sombra
        # do objeto removido.
        workplace_activity = False
        while workplace_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            learning_rate = 0.1  # há objeto presente na imagem, configurando
            # o learningRate para valor alto, maior que 0.01.
            # Usar parênteses no desempacotamento não cria tupla!
            (valid_boxes,
             border_boxes,
             final_object_box) = bgsub.locate_object(frame, learning_rate)

            if valid_boxes or border_boxes:
                # workplace tem movimento, transição para Detect_object
                workplace_activity = True
                self.nxt_transition = 'trigger_extraction_movement'
            else:
                # Sem movimentação no espaço de trabalho, não vai sair do loop
                # manter próxima transição como object blocking caso alguma
                # coisa no pytransitions faça sair deste estado
                workplace_activity = False
                self.nxt_transition = 'trigger_object_blocking'

    def on_enter_Detect_object(self):
        # A transição para cá ocorre apenas se foi detectado movimento. Agora
        # vai aguardar o movimento cessar e conferir se há contorno de objeto
        # no espaço de trabalho. Se houver, volta para Workplace_blocked. Se
        # não, segue para Configuration
        workplace_activity = True
        while workplace_activity is True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            learning_rate = 0.1  # há objeto presente na imagem, configurando
            # o learningRate para valor alto, maior que 0.01.
            # Usar parênteses no desempacotamento não cria tupla!
            (valid_boxes,
             border_boxes,
             final_object_box) = bgsub.locate_object(frame, learning_rate)

            # Aguarda acabar a movimentação na imagem, ou por não existir
            # mais o objeto na cena, ou por acontecer o erro de Timeout
            if not valid_boxes or not border_boxes:
                # espaço de trabalho sem movimento (pixel na foreground mask),
                # verificar se há objeto pelo contorno na imagem
                preproc_img = bgsub.preprocess_image(frame)
                self.blocking_object = bgsub.is_object_at_image(preproc_img)[0]

                if self.blocking_object is True:
                    # Não há movimentação no workplace e objeto permanece.
                    # voltar para estado Workplace_blocked
                    workplace_activity = False
                    self.nxt_transition = 'trigger_object_blocking'
                else:
                    # Não há contorno de objeto. Seguir para o estado
                    # Configuration
                    workplace_activity = False
                    self.nxt_transition = 'trigger_workplace_free'

            else:
                # Há movimentação no espaço de trabalho, não vai sair do loop
                # manter transição como trigger_extraction_movement caso alguma
                # coisa no pytransitions faça sair deste estado
                workplace_activity = True
                self.nxt_transition = 'trigger_extraction_movement'

    # Note: the same frame detected in Monitoring state must be used up to the
    # state Object_extraction?Não precisa, pois o mesmo objeto do subtrator de
    # background é usado em todos os estados, e esse objeto mantém o histórico
    # de frames para calcular as diferenças entre pixels

    def on_enter_Monitoring(self):
        # workplace sem objeto. Aguardar até que haja movimento
        workplace_activity = False
        while workplace_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            # Usar parênteses no desempacotamento não cria tupla!
            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(frame)

            if valid_boxes or border_boxes:
                # workplace tem movimento, transição para Tracking_objects
                workplace_activity = True
                self.nxt_transition = 'trigger_movement_detected'
            else:
                # Sem movimentação no espaço de trabalho, não vai sair do loop
                # manter próxima transição como object blocking caso alguma
                # coisa no pytransitions faça sair deste estado
                workplace_activity = False
                self.nxt_transition = 'trigger_workplace_ready'

    def on_enter_Tracking_objects(self):
        # Rastrear se acabou o movimento na cena.
        # Buscar se há algum contorno de objeto. Se houver, segue para
        # estado Object_position. Se não, retorna para estado Monitoring.
        # Não diferencia entre objetos nas bordas ou no centro dos frames.
        workplace_activity = True
        last_all_boxes = []
        # last_final_object_box = []
        first_frame = True
        while workplace_activity is True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            (valid_boxes,
             border_boxes,
             _) = bgsub.locate_object(frame)  # learning rate default (0.0001)

            # Acontece movimento e rastreia se o movimento encerra. em
            # seguida verificar se cessou o movimento buscando contorno

            # se for a primeira execução da busca por movimento
            if first_frame is True:
                last_all_boxes = valid_boxes + border_boxes
                first_frame = False  # never ever back here again...
                # Cannot leave this state yet. It does not mean the
                # state is left then reentered again...

            else:
                # Juntar todos os boxes encontrados para facilitar a comparação
                # entre máscaras, concatenando as listas
                union_boxes = valid_boxes + border_boxes

                intersect = cv2.bitwise_and(last_all_boxes, union_boxes)
                iou = cv2.countNonZero(intersect)/cv2.countNonZero(union_boxes)

                if iou > 0.9:  # Intersection over Union
                    # Mais de 90% de sobreposição entre os boxes para margem
                    # à ruido de imagem nas máscaras de foreground de 10% de
                    # variações de pixels da máscara.
                    print("Objeto parado ou sem objeto")

                    # Julgar se há objeto no workplace ao buscar por contornos
                    # de objetos no frame.
                    self.blocking_object = bgsub.is_object_at_image(frame)[0]

                    if self.blocking_object is True:
                        # Há objeto no workplace, depois de cessar movimento
                        # Seguir para estado Object_position
                        self.nxt_transition = 'trigger_object_stopped'
                        workplace_activity = False
                    else:
                        # Não há objeto no workplace, voltar para Monitoring
                        self.nxt_transition = 'trigger_movement_detected'
                        workplace_activity = False
                else:
                    print('objeto movimentando')
                    workplace_activity = True
                    # Não precisa mudar a transição, continuar

    def on_enter_Object_position(self):
        pass

    def on_enter_Validation_time(self):
        pass

    def on_enter_Take_image(self):
        pass

    def on_enter_Object_extraction(self):
        pass

    def start_object_tracking(self):
        # Esta função vai ciclar os estados. Cada entrada de estado dispara
        # um callback on_enter_<<estado>> assim que termina a transição.
        # Usar método .trigger('proximo_estado'), em que o próximo estado é
        # definido dinamicamente

        while self.terminate_flag is not True:
            print(f'{self.state}')
            # How to listen any event that may trigger the terminate flag?
            self.trigger(self.nxt_transition)


if __name__ == "__main__":
    # Vídeos originais gravados em 4k. Reduzir resolução para a cópia usada
    # para calcular os contornos? Reencodados videos para Full HD para testar
    # os algoritmos, depois usar os vídeos originais em 4k

    pasta = "video\\videosMock\\"

    # videos completos, para teste final do módulo
    # video simples
    # cap = cv2.VideoCapture(r"video\teste.mp4")

    # video complexo, com várias condições adversas
    # cap = cv2.VideoCapture(r"video\video.mp4")

    # objetos genéricos
    objeto1 = "cabo luz off.mp4"
    objeto2 = "cabo movimento maos luz on.mp4"
    objeto3 = "caixa clara movimento maos luz on.mp4"
    objeto4 = "caixa desde inicio luz on.mp4"
    objeto5 = "caixa luz off.mp4"
    objeto6 = "caixa mudanca iluminacao.mp4"
    objeto7 = "Paquimetro luz off.mp4"
    objeto8 = "Paquimetro mao presente luz off.mp4"
    objeto9 = "Paquimetro para caixa luz off.mp4"
    objeto10 = "Regua luz off.mp4"
    objeto11 = "regua refletiva luz on.mp4"

    # mock de instrumento
    objeto12 = "BaixaIluminacao100luxSombraForte.mp4"
    objeto13 = "TrocaObjetosAutofocoAtrapalha.mp4"
    objeto14 = "Iluminacao800_560lux.mp4"
    objeto15 = "Objeto15segs.mp4"
    objeto16 = "Objeto15segSubstituido.mp4"
    objeto17 = "objeto3segs.mp4"
    objeto18 = "ObjetoInicio.mp4"
    objeto19 = "ObjetoOrtogonalDiagonal.mp4"
    objeto20 = "ObjetoReposicionado.mp4"
    objeto21 = "OclusãoMão.mp4"
    objeto22 = "OclusãoTempMão.mp4"

    # - para gerar valores HSV do background
    objeto23 = "TemperaturaCor3k_9k.mp4"

    # - valores HSV do background
    objeto24 = "ContrasteTemperaturaCor3k_9k.mp4"

    # selecionar o video do objeto
    objeto = objeto17

    # Considerar na versão final que o dispositivo deve ser uma câmera
    # Inicializar captura de imagens
    device = pasta + objeto
    # se usada a câmera principal, comentar linha anterior e descomentar
    # a linha seguinte
    # device = 0

    supervisor = ObjectTracking(device, True)

    print(supervisor.state)  # to follow what is the current state
    supervisor.start_object_tracking()
    #supervisor.trigger_initialize
    #print(supervisor.state)
