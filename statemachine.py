# máquina de estados
from transitions import Machine
import cv2
import BackgroundSubtraction as bgsub


class ObjectTracking(object):

    # Defining states, following the image acquisition requirements
    states = ['configurar', 'monitorar', 'detectar_objeto',
              'objeto_coberto', 'objeto_cortado', 'obter_imagem',
              'aguarda_novo_objeto']

    def __init__(self):

        # Initialize the state machine
        self.machine = Machine(model=self, states=ObjectTracking.states,
                               initial='configurar', after='initialization')

        # At "configurar" state, camera will configure the system for
        # capturing images. When configuration ends, it will change
        # Unconditionally to "monitorar" state.
        self.machine.add_transition(trigger='iniciar_monitoramento',
                                    source='configurar',
                                    dest='monitorar',
                                    )

        # At the "monitorar" state, it will capture images continuously, and
        # change to state "detectar_objeto" if any object enters the scene,
        # and keeps moving
        self.machine.add_transition(trigger='objeto_movimentando',
                                    source='monitorar',
                                    dest='detectar_objeto')

        # At the "detectar_objeto" state, it will analyze if the movement in
        # scene stops, if there is an object contour identified
        # and its bounding box has no intersection with image border region
        self.machine.add_transition(trigger='objeto_centrado',
                                    source='detectar_objeto',
                                    dest='obter_imagem')

        # At the "obter_imagem" state, all necessary conditions for
        # capturing the image are satisfied. The image is sent to the image
        # recognition server/service, and the system must wait for reposition
        # or removal of the current object
        self.machine.add_transition(trigger='imagem_enviada',
                                    source='obter_imagem',
                                    dest='aguarda_novo_objeto')

        # At the "aguarda_novo_objeto", the system waits for movement at the
        # image borders, in order to not take another image of the same object
        # at the same position
        self.machine.add_transition(trigger='movimento_nas_bordas_da_imagem',
                                    source='aguarda_novo_objeto',
                                    dest='monitorar')

    def initialization(self):
        # Inicializar captura de imagens
        device = 0  # pasta + objeto

        # deve ser criado um objeto "Supervisor" para armazenar atributos
        # de inicialização do sistema?
        cap = cv2.VideoCapture(device)

        # Treating adverse image conditions
        # Image not stopping
        # Image occlusions
        # Service not recognizing the image (needs object repositioning)
        # 


if __name__ == "__main__":
    supervisor = ObjectTracking("MaquinaEstadosCamera")
    print(supervisor.state)
    supervisor.iniciar_monitoramento()
    print(supervisor.state)
    supervisor.nap()
    print(supervisor.state)
    try:
        supervisor.clean_up()
    except Exception as e:
        # Captura e exibe o tipo e a mensagem da exceção
        print(f"Ocorreu uma exceção: {type(e).__name__} - {e}")
    supervisor.wake_up()
    supervisor.work_out()
    print(supervisor.state)
#'hungry'

# supervisor still hasn't done anything useful...
print(supervisor.kittens_rescued)#0

# We now take you live to the scene of a horrific kitten entreement...
supervisor.distress_call()
#'Beauty, eh?'
print(supervisor.state)
#'saving the world'

# Back to the crib.
supervisor.complete_mission()
print(supervisor.state)
#'sweaty'

supervisor.clean_up()
print(supervisor.state)
# 'asleep'   # Too tired to shower!

# Another productive day, Alfred.
supervisor.kittens_rescued
