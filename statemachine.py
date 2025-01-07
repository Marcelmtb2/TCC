# máquina de estados
from transitions import Machine
import random


class ObjectTracking(object):

    # Defining states, following the image acquisition requirements
    states = ['configurar', 'monitorar', 'obter_imagem', 'detectar_objeto',
              'objeto_coberto', 'objeto_cortado', ]

    def __init__(self):

        # Initialize the state machine
        self.machine = Machine(model=self, states=ObjectTracking.states,
                               initial='configurar')

        # At "configurar" state, camera will configure the system for
        # capturing images. When configuration ends, it will change
        # Unconditionally to "monitorar" state.
        self.machine.add_transition(trigger='iniciar_monitoramento',
                                    source='configurar',
                                    dest='monitorar')

        # At the "monitorar" state, it will capture images continuously, and
        # change to state "detectar_objeto" if any object enters the scene
        self.machine.add_transition(trigger='objeto_movimentando',
                                    source='monitorar',
                                    dest='detectar_objeto')

        # At the "detectar_objeto" state, it will analyze if the centroid
        # of the detected contour stops moving in the image, and its
        # bounding box has no intersection with image borders
        self.machine.add_transition(trigger='objeto_parado',
                                    source='detectar_objeto',
                                    dest='obter_imagem')

        # At the "obter_imagem" state, all necessary conditions for
        # capturing the image are satisfied. The image is sent to the image
        # recognition server/service, and the system must return to the
        # "monitorar" state to get ready for new image acquisitions
        self.machine.add_transition(trigger='imagem_enviada',
                                    source='obter_imagem',
                                    dest='monitorar')

        # Treating adverse image conditions
        # Image not stopping
        # Image occlusions
        # Service not recognizing the image (needs object repositioning)
        # 


if __name__ == "__main__":
    batman = ObjectTracking("Camera")
    print(batman.state)
    batman.wake_up()
    print(batman.state)
    batman.nap()
    print(batman.state)
    try:
        batman.clean_up()
    except Exception as e:
        # Captura e exibe o tipo e a mensagem da exceção
        print(f"Ocorreu uma exceção: {type(e).__name__} - {e}")
    batman.wake_up()
    batman.work_out()
    print(batman.state)
#'hungry'

# Batman still hasn't done anything useful...
print(batman.kittens_rescued)#0

# We now take you live to the scene of a horrific kitten entreement...
batman.distress_call()
#'Beauty, eh?'
print(batman.state)
#'saving the world'

# Back to the crib.
batman.complete_mission()
print(batman.state)
#'sweaty'

batman.clean_up()
print(batman.state)
# 'asleep'   # Too tired to shower!

# Another productive day, Alfred.
batman.kittens_rescued
