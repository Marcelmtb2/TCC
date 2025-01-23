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
# al´m de retornar imagem, retornar status de objeto no workspace
    def __init__(self, device=0, show_debug=False):

        self.device = device  # As default, look for a installed camera at PC

        self.subtractor_bg = None  # Handler for BackgroundSubtractor

        self.nxt_transition = 'trigger_initialize'  # first transition name

        self.terminate_flag = False  # flag for terminate the execution of this
        # state machine

        self.output_image = None  # current image selected by the state machine

        self.previous_output_image = None  # backup image selected by the state machine

        self.image_available_flag = False  # flag for image available

        self.workspace_blocked = False  # flag for workspace blocked

        self.show_debug = show_debug  # flag to turn on all debugging figures

        self.frame_count = 0  # frame counter for showing debugging images

        # Initialize the state machine, with a pseudo-state. 
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
                                            'Tracking_objects',
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
        workplace_has_activity = True
        first_frame = True
        # must have a minimum loop for compare at least two frames and
        # assert if the object found is still in the scene.
        while workplace_has_activity is True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            (valid_boxes,
             border_boxes, _) = bgsub.locate_object(frame)
            # default learning rate  (0.0001)

            # Acontece movimento e rastreia se o movimento encerra. em
            # seguida verificar se cessou o movimento buscando contorno

            # se for a primeira execução da busca por movimento
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # never ever back here again...
                # Cannot leave this state yet. It does not mean the
                # state is left then reentered again...

            else:
                # Juntar todos os boxes encontrados para facilitar a comparação
                # entre máscaras, concatenando as listas.
                # Se espera que union_boxes nunca seja vazia neste ponto do
                # código, porque está testando o recente fim de um movimento
                # detectado pelo Subtrator de Background.
                union_boxes = valid_boxes + border_boxes
                intersect = cv2.bitwise_and(past_all_boxes, union_boxes)
                iou = (cv2.countNonZero(intersect) /
                       cv2.countNonZero(union_boxes))

                if iou > 0.9:  # Intersection over Union
                    #  90% de sobreposição entre os boxes para dar margem
                    # à ruido nas máscaras de foreground
                    print("Objeto parado ou não há objeto")

                    # Confirmar se há objeto no workplace ao buscar por
                    # contornos de objetos no frame.
                    # Pré-processar a imagem.
                    preproc = bgsub.preprocess_image(frame)
                    object_ok = bgsub.is_object_at_image(preproc)[0]
                    if object_ok is True:
                        # Há objeto no workplace, sem movimentos
                        self.nxt_transition = 'trigger_object_stopped'
                        self.blocking_object = True
                        workplace_has_activity = False
                    else:
                        # Não há objeto no workplace, voltar para Monitoring
                        self.nxt_transition = 'trigger_workplace_ready'
                        workplace_has_activity = False
                        self.blocking_object = False

                else:
                    print('Movimento detectado')
                    workplace_has_activity = True
                    self.nxt_transition = 'trigger_movement_detected'

    def on_enter_Object_position(self):
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
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            (valid_boxes,
             border_boxes, _) = bgsub.locate_object(frame)
            # default learning rate (0.0001)
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # never ever back here again...

            if valid_boxes and not border_boxes:  # Trigger transition to
                # Validation_time? Confirm if the object is centered by
                # finding contours in frame, and not touching margins.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # Find in image for contours of objects
                mask_height, mask_width = frame.shape
                size_border_factor = 0.01  # Bordas de 10 pixels (1%)
                margin_top_bot = int(mask_height * size_border_factor)
                margin_left_right = int(mask_width * size_border_factor)
                # fazer essa borda como percentual, 1% da dimensão H ou W

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Verificar se algum contorno está nas bordas da imagem
                    top_left = ((x > margin_left_right) and
                                (y > margin_top_bot))

                    top_right = (((x + w) < (mask_width - margin_left_right))
                                 and (y > margin_top_bot))

                    bot_left = ((x > margin_left_right) and
                                ((y + h) < (mask_height - margin_top_bot)))

                    bot_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 ((y + h) < (mask_height - margin_top_bot)))

                    image_outside_borders = (top_left and top_right and
                                             bot_left and bot_right)
                    # False if any corner inside borders

                    if image_outside_borders is True:
                        # all found contours are valid ones
                        # no timeout event trigger this branch
                        self.nxt_transition = 'trigger_object_centered'
                        workplace_move_activity = True
                    else:
                        # any found countours is border one
                        # timeout event trigger this branch.
                        # When timeout occurs, the object mask fades away
                        # gradually, and the border contours may disappear
                        # before the valid ones.
                        self.nxt_transition = 'trigger_timeout'
                        workplace_move_activity = True
                        break

            elif not valid_boxes and not border_boxes:  # there was an object
                # Reaches here only if Timeout. There is another state to
                # track movement (else clause), to avoid reaching here if
                # the object is repositioned.

                self.nxt_transition = 'trigger_timeout'

            else:  # border_boxes found, considered as movement for removing
                # object from workplace. May have valid_boxes too, but they are
                # not considered in this state. The object is not centered yet.
                # It will be hard to differentiate any part of object inside
                # image margins from moving objects. This is simplified as
                # considering the presence of any border_boxes as movement.
                self.nxt_transition = 'trigger_movement_detected'
                workplace_move_activity = True

    def on_enter_Validation_time(self):
        # Wait for 0.5 seconds to validate that the object is centered
        frames_to_wait = int(self.fps / 2)  # FPS times 0.5 seconds

        while frames_to_wait > 0:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            frames_to_wait -= 1
            (_, border_boxes, _) = bgsub.locate_object(frame)

            if border_boxes:
                # Movement detected, return to Tracking_objects state
                self.nxt_transition = 'trigger_movement_detected'
                break
            # os contornos nesta etapa são os válidos apenas. Se qualquer
            # border_box for detectado, a transição será para o estado
            # Tracking_objects. Senão, a transição será para Take_image.
        # Validation time is over, object is centered and immobile and no
        # occlusion is detected. Transition to Take_image state
        self.nxt_transition = 'trigger_stabilization_time'
        self.output_image = frame

    def on_enter_Take_image(self):
        # Capturar a imagem do objeto e enviar para o servidor central
        ret, frame = self.cap.read()
        if not ret:
            print('Câmera não enviou imagens. Conferir o equipamento.')
        else:
            self.output_image = frame
            self.image_available_flag = True
            self.nxt_transition = 'trigger_image_sent'


    def on_enter_Object_extraction(self):
        # Aguardar a retirada do objeto do espaço de trabalho. Se o objeto
        # for retirado, transição para Monitoring. Se o objeto não for retirado
        # após um longo tempo, o algoritmo de Subtração de Background começa
        # a integrar o objeto como imagem de background. Neste caso, após a
        # retirada do objeto, permanece uma "sombra" na máscara de foreground
        # na última localização do objeto. Para corrigir este erro, é melhor
        # reutilizar o estado Workplace_blocked que já resolve um problema
        # semelhante.

        workplace_move_activity = False
        first_frame = True

        while workplace_move_activity is not True:
            ret, frame = self.cap.read()
            if not ret:
                print('Câmera não enviou imagens. Conferir o equipamento.')
                # criar alguma maneira de lidar com esse tipo de erro
                break

            (valid_boxes,
             border_boxes, _) = bgsub.locate_object(frame)
            # default learning rate (0.0001)
            if first_frame is True:
                past_all_boxes = valid_boxes + border_boxes
                first_frame = False  # never ever back here again...

            if valid_boxes and not border_boxes:  # Trigger transition to
                # Validation_time? Confirm if the object is centered by
                # finding contours in frame, and not touching margins.
                # Avoids timeout gradual fading contours situation.
                preproc = bgsub.preprocess_image(frame)
                object_ok, contours = bgsub.is_object_at_image(preproc)

                # Find in image for contours of objects
                mask_height, mask_width = frame.shape
                size_border_factor = 0.01  # Bordas de 10 pixels (1%)
                margin_top_bot = int(mask_height * size_border_factor)
                margin_left_right = int(mask_width * size_border_factor)
                # fazer essa borda como percentual, 1% da dimensão H ou W

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Verificar se algum contorno está nas bordas da imagem
                    top_left = ((x > margin_left_right) and
                                (y > margin_top_bot))

                    top_right = (((x + w) < (mask_width - margin_left_right))
                                 and (y > margin_top_bot))

                    bot_left = ((x > margin_left_right) and
                                ((y + h) < (mask_height - margin_top_bot)))

                    bot_right = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 ((y + h) < (mask_height - margin_top_bot)))

                    image_outside_borders = (top_left and top_right and
                                             bot_left and bot_right)
                    # False if any corner inside borders

                    if image_outside_borders is True:
                        # all found contours are valid ones
                        # no timeout event trigger this branch
                        self.nxt_transition = 'trigger_object_centered'
                        workplace_move_activity = True
                    else:
                        # any found countours is border one
                        # timeout event trigger this branch.
                        # When timeout occurs, the object mask fades away
                        # gradually, and the border contours may disappear
                        # before the valid ones.
                        self.nxt_transition = 'trigger_timeout'
                        workplace_move_activity = True
                        break

    def start_object_tracking(self, terminate=False, interrogate):
        # Esta função vai ciclar os estados. Cada entrada de estado dispara
        # um callback on_enter_<<estado>> assim que termina a transição.
        # Usar método .trigger('proximo_estado'), em que o próximo estado é
        # definido dinamicamente
        #Saídas são flag de imagem capturada e imagem selecionada.
        # se imagem indisponível, flag falsa
        # se imagem disponível, flag verdadeira e imagem disponível
        # se imagem lida, zerar flag de imagem capturada
        while self.terminate_flag is not True:
            print(f'{self.state}')
            # How to listen any event that may trigger the terminate flag?
            self.trigger(self.nxt_transition)

    def get_image(self):
        # Retorna a imagem capturada, se houver
        if self.image_available_flag:
            # Zerar flag de imagem capturada
            self.image_available_flag = False

            # Armazena imagem de saída
            output = self.output_image
            # Armazena backup da imagem
            self.previous_output_image = self.output_image.copy()


            return output
        else:
            # Nenhuma imagem disponível
            return None

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
