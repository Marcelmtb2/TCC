import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Object Tracking and Detection Module

This module implements the necessary tasks to track and detect
the presence of one object in a video stream, in a controlled enviroment.

The module has the tools to track the movement of objects and evaluate if
an object brought into the scene is stand still, if it is centered in the
scene by not reaching the image limits, and if it is not occluded by
obstacles or by the hands of an technician operating with the system.

Returns:
    _type_: _description_
"""


# inicializar sistema de captura de objetos
def config_object_capture():

    # Inicializar o subtrator de background com valores default
    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)

    # All default values were confirmed, as testing history and varThreshold
    # do not altered much the final result
    return fgbgMOG2


def find_object_at_start(image, object_at_start_flag=False):
    # Verificar se o primeiro frame contém contornos de um objeto.
    # Configurar o método de subtração de background com uma taxa
    # de aprendizado muito alta, para adaptar rapidamente o plano
    # de fundo médio até que o objeto seja removido.
    # Para detectar bordas, usar threshold em imagens convertidas para escala
    # de cinza ou aplicar filtro Canny para evidenciar bordas de contornos
    # Recebe imagem pré-processada

    # Se o objeto mais escuro que o plano de fundo verde, em escala de cinza
    thresh_dark = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("mascara dark", thresh_dark)
    # Se o objeto mais claro que o plano de fundo verde, em escala de cinza
    thresh_light = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("mascara light", thresh_light)
    # Máscara final é a combinação bitwise or das duas máscaras

    thresh = cv2.bitwise_or(thresh_dark, thresh_light)
    cv2.imshow("mascara lightORdark", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # se algum contorno for identificado, retornar true
    if len(contours):
        width_img_gray, height_img_gray = image.shape[0:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Problema: Se houver sombra nítida, será confundido com
            # um objeto. Usar "margem de histerese" ao definir os thresholds
            # das máscaras de objetos claros e objetos escuros
            if w >= 0.9 * width_img_gray:  # se um contorno de sombra atravessa
                # a imagem de um lado a outro
                return False
            cv2.rectangle(image, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
        # visualize the binary image
        cv2.imshow('Binary image', image)
        cv2.waitKey(0)
        return True
    return False
    # find contours of objects before background subtraction. If there are any
    # contours, max out the learningRate parameter to 0.1 until no object is
    # present in the image


def preprocess_image(image):
    # filtrar ruído de digitalização da câmera.
    # Testados filtros para os casos com baixa iluminação

    # image está no formato 16:9
    height, width = image.shape[0:2]
    # imagem da câmera BRIO vem em 4k (2160 x 3840)
    # Cortar imagem para remover os pés do suporte de câmera
    cropleft = int(0.21875 * width)
    cropright = int(0.77 * width)  # para cortar ponta do suporte.
    # 0.78125 foi o valor originalmente pensado
    # Remove completamente com valor 0.77

    frame = image[0:height, cropleft:cropright]

    # Redimensionar para acelerar processamento
    reduced = cv2.resize(frame, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)

    # Filtrando o ruído de digitalização da câmera
    f_bilateral = cv2.bilateralFilter(reduced, d=9, sigmaColor=75,
                                      sigmaSpace=125)

    # Conversão para tons de cinza
    gray_frame = cv2.cvtColor(f_bilateral, cv2.COLOR_BGR2GRAY)

    # Debug - ver a imagem original, antes de filtrar
    cv2.imshow("Imagem cortada e redimensionada", reduced)

    return gray_frame


def find_foreground_object(image, learning_rate=0.0001):
    # Identifica objetos que se movem em relação ao background
    # O parâmetro learningRate controla em quanto tempo um objeto
    # em movimento que para na cena será considerado parte do background
    # Com learningRate muito pequeno, 0.0001, para preservar objeto
    # por algum tempo após parar o movimento. Calibrar para que o objeto
    # seja considerado background após 10 segundos, na inicialização.
    fgmaskMOG2 = fgbgMOG2.apply(image, learningRate=learning_rate)
    thresh_mask = cv2.threshold(fgmaskMOG2, 210, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.dilate(thresh_mask, kernel, iterations=2)
    return clean_mask


def identify_contours(object_found_binary_mask):
    # Define contornos do possível objeto
    # Geralmente, não será possível identificar perfeitamente o objeto apenas
    # com a subtração de background
    # Se houver atividade da subtração de background, classificar se os pixels
    # modificados formam objetos.
    # Descartar contornos que estejam a uma margem de 5% da dimensão das bordas
    # Agrupar os contornos válidos em um bounding_box maior
    # Confirmar a máscara calculada com o subtrator de background com o
    # contorno de objeto a partir da imagem original.
    # Retorna True ou False para objeto detectado ou não

    # Detecção de contornos na máscara calculada
    contours = cv2.findContours(object_found_binary_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    mask_height, mask_width = object_found_binary_mask.shape
    image_border = 10  # Distância mínima das bordas de 10 pixels
    valid_contours = []
    border_contours = []
    for contour in contours:
        x_0, y_0, x_1, y_1 = cv2.boundingRect(contour)
        # Verificar se o contorno está longe das bordas
        # X0,y0 e x1,y1 são coordenadas da diagonal do contorno
        bool_top_corner = x_0 > image_border and y_0 > image_border
        bool_bot_corner1 = (x_0 + x_1) < mask_width - image_border
        bool_bot_corner2 = (y_0 + y_1) < mask_height - image_border

        if bool_bot_corner1 and bool_bot_corner2 and bool_top_corner:
            valid_contours.append(contour)
        else:
            border_contours.append(contour)
    # Criar uma lista de bounding boxes, se existirem
    if len(contours):
        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        # Agrupar bounding boxes próximas
        x_min = min([x for (x, y, w_box, h_box) in bounding_boxes])
        y_min = min([y for (x, y, w_box, h_box) in bounding_boxes])
        x_max = max([x + w_box for (x, y, w_box, h_box) in bounding_boxes])
        y_max = max([y + h_box for (x, y, w_box, h_box) in bounding_boxes])

        # Caixa circundante final
        final_bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    # identificando quais contornos estão em contato com a borda
    # agrupar apenas os contornos longe da borda
    # se algum contorno em contato com a borda se aproximar do
    # objeto, considerar como sendo a mão do operador.
    
    

    print(f'foram encontrados {len(contours)} contornos.')
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area > MIN_CONTOUR_AREA:
        # Obter as coordenadas do retângulo delimitador
        x, y, w, h = cv2.boundingRect(contour)

        # Desenhar contorno válido e exibir mensagem
        # x, y, w, h = cv2.boundingRect(valid_contour)
        cv2.rectangle(preproc_image, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)


def analyze_hsv(frame):
    # # plotar o histograma RGB e HSV do frame atual

    # colors = ('b', 'g', 'r')
    # total_pixels = frame.shape[0] * frame.shape[1]  # Total de pixels
    # no frame

    # # Inicializar janela gráfica, se necessário
    # if not hasattr(analyze_hsv, 'initialized'):
    #     plt.ion()
    #     analyze_hsv.fig, (ax_hist) = plt.subplots(1, 1, figsize=(6, 6))

    #     ax_hist = analyze_hsv.ax_hist = analyze_hsv.fig.add_subplot(111)
    #     ax_hist.set_title("Histograma Acumulado")
    #     ax_hist.set_xlabel("Intensidade de Pixels")
    #     ax_hist.set_ylabel("Percentual (%)")  # ("Número de Pixels")
    #     ax_hist.set_xlim([0, 256])
    #     ax_hist.set_ylim([0, 10])  # Limite do eixo Y consistente
    #     analyze_hsv.initialized = True

    # # Limpar o gráfico anterior
    # ax_hist = analyze_hsv.ax_hist
    # ax_hist.clear()
    # ax_hist.set_title("Histograma Acumulado")
    # ax_hist.set_xlabel("Intensidade de Pixels")
    # ax_hist.set_ylabel("Percentual (%)")  # ("Número de Pixels")
    # ax_hist.set_xlim([0, 256])
    # ax_hist.set_ylim([0, 10])  # Limite do eixo Y consistente

    # # Calcular e acumular histogramas
    # for i, color in enumerate(colors):
    #     hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
    #     # accumulated_histograms[color] += hist.flatten()
    #     hist_percent = (hist.flatten() / total_pixels) * 100
    # # Converter para percentual
    #     accumulated_histograms[color] = hist_percent
    # # Acumular em percentual
    #     ax_hist.plot(accumulated_histograms[color], color=color)

    # # Atualizar gráficos
    # plt.pause(0.001)

    # Converter para HSV (mais eficaz para diferenciar cores)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir faixa para o plano de fundo verde
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Criar uma máscara para o plano de fundo
    bg_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Criar máscara para o objeto (inverter máscara do plano de fundo)
    object_mask = cv2.bitwise_not(bg_mask)

    # Calcular histogramas
    bg_hist = cv2.calcHist([hsv], [0, 1], bg_mask, [180, 256],
                           [0, 180, 0, 256])
    obj_hist = cv2.calcHist([hsv], [0, 1], object_mask,
                            [180, 256], [0, 180, 0, 256])

    # Normalizar histogramas
    bg_hist = cv2.normalize(bg_hist, bg_hist).flatten()
    obj_hist = cv2.normalize(obj_hist, obj_hist).flatten()

    # Subtrair histogramas
    hist_difference = bg_hist - obj_hist

    # Plotar histogramas
    plt.figure()
    plt.title("Histograma do Objeto vs. Plano de Fundo")
    plt.plot(obj_hist, label="Objeto", color="red")
    plt.plot(bg_hist, label="Plano de Fundo", color="green")
    plt.plot(hist_difference, label="Diferença", color="blue")
    plt.legend()
    plt.show()


def locate_object(image):
    # Recebe imagem preprocessada, retorna máscara e bounding box do objeto
    #
    pass


def check_capture_conditions():
    # tem que usar vários frames para fazer essa verificação
    # Usar a máquina de estados para isso
    pass


def send_capture():
    # Estágio final, para enviar o frame original depois de
    # satisfeitas todas os requisitos para o envio da imagem
    pass


def reajust_background():
    # recalcula o learningRate de acordo com a condição de cena
    # 1 - mudanças bruscas de iluminação e se algum contorno foi
    # identificado na borda sem haver contorno encontrado longe
    # das bordas == Acelerar o learningRate(>0.01)
    # 2 - Se acionada a mudança de learningRate, aguardar 5 frames
    # para retornar o learningRate para o padrão 0.001 e manter
    # a persistência do contorno identificado
    pass


def check_illumination_change(fg_mask,):
    # rotina para checar mudanças bruscas de iluminação, para
    # evitar a formação de artefatos
    # Caso haja mudança brusca de iluminação, comandar o
    # reinício da captura de background com as novas condições
    # de iluminação.

    # Definir o limiar para detectar mudanças bruscas de iluminação
    threshold_ratio = 0.7  # 70% dos pixels

    # Contar os pixels não nulos na máscara
    non_zero_pixels = cv2.countNonZero(fg_mask)
    total_pixels = fg_mask.size
    non_zero_ratio = non_zero_pixels / total_pixels

    # Verificar se a proporção ultrapassa o limiar
    if non_zero_ratio > threshold_ratio:
        print("Mudança brusca de iluminação detectada!")
        cv2.putText(frame, "Lighting Change Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # return True
    # return False
    # DEBUG - Mostrar a máscara e o quadro original
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Video Feed', frame)


if __name__ == '__main__':

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
    objeto = objeto11

    # ====================================
    # Adicionar marcações para a máquina de estados!
    # Descobrir como chamar as funções de transição da máquina
    # de estados.
    # Fazer o programa de forma que as funções de processamento
    # de imagem sejam chamadas pelos estados correspondentes da
    # máquina de estados.
    # ====================================

    # ====================================
    # Estado 0 - Inicializar
    # Configura o sistema para condição mínima de funcionamento.
    # ====================================

    # Considerar na versão final que o dispositivo deve ser uma câmera
    # Inicializar captura de imagens
    device = pasta + objeto

    # se usada a câmera principal, comentar linha anterior e descomentar esta
    # device = 0

    cap = cv2.VideoCapture(device)

    # Inicializar o subtrator de background com valores default.
    fgbgMOG2 = config_object_capture()

    # Inicializar a checagem de presença de objeto no início do vídeo.
    # Caso o primeiro frame do vídeo tenha contornos de objeto, manter
    # flag object_at_start_flag em True
    object_at_start_flag = True

    # Flag para debugar estado do sistema
    system_status = "None"

    # _____________
    # Separando código para visualizar resultados durante o desenvolvimento
    # Obter a taxa de quadros (FPS) do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # deve incluir aqui o código para checar se há objeto no início do
    # funcionamento do programa? Se houver objeto, não pode executar a
    # tarefa corretamente até que o objeto seja removido
    # ===================================
    # Fim do estado Inicializar
    # ===================================

    # ===================================
    # Inicio do estado Monitorar
    # Ação contínua, por isso deve incluir o loop
    # ===================================

    # Loop para checar presença de objetos frame a frame
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print('Câmera não enviou imagens. Conferir o equipamento.')
            system_status = "errorCamera"
            break

        # Pré-processar a imagem, filtrando ruídos e redimensionando.
        preproc_image = preprocess_image(frame)

        # verificar se há objeto desde o início será a ultima atividade!!
        # # Verificar se o primeiro frame contém contornos de um objeto.
        # if object_at_start_flag:
        #     object_at_start_flag = find_object_at_start(preproc_image)
        #     learning_rate = 0.1
        #     # No primeiro loop do programa, considera que há objeto
        #     # presente na imagem, configurando o learningRate para valor alto,
        #     # maior que 0.01. Em seguida, testa se não existe contorno de
        #     # algum objeto na primeira imagem recebida.
        #     # Caso exista o objeto, manter a taxa alta.
        # else:
        learning_rate = 0.0001
        # Se retornar False, manter a taxa de aprendizado baixa, para
        # melhorar a persistência do objeto na máscara calculada

        clean_mask = find_foreground_object(preproc_image, learning_rate)

        # ---------------------------------------------- Transformar em função
        mask_height, mask_width = clean_mask.shape  # Dimensões da imagem
        image_border = 10  # Distância mínima das bordas de 10 pixels
        # fazer essa borda como percentual, entre 2 a 5% da dimensão H ou W
        valid_contours = []
        border_contours = []
        # Detecção de contornos na máscara calculada
        contours = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Verificar se o contorno está longe das bordas
            top_left_corner = x > image_border and y > image_border
            top_right_corner = (x+w) < (mask_width - w) and (y+w) < (mask_height - w)
            bot_left_corner = (x_0 + x_1) < mask_width - image_border
            bot_right_corner = (y_0 + y_1) < mask_height - image_border

            if bool_bot_corner1 and bool_bot_corner2 and bool_top_corner:
                valid_contours.append(contour)
            else:
                border_contours.append(contour)

        # Criar uma lista de bounding boxes, se existirem
        if len(contours):
            bounding_boxes = [cv2.boundingRect(c) for c in contours]

            # Agrupar bounding boxes próximas
            x_min = min([x for (x, y, w_box, h_box) in bounding_boxes])
            y_min = min([y for (x, y, w_box, h_box) in bounding_boxes])
            x_max = max([x + w_box for (x, y, w_box, h_box) in bounding_boxes])
            y_max = max([y + h_box for (x, y, w_box, h_box) in bounding_boxes])

            # Caixa circundante final
            final_bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)

            # identificando quais contornos estão em contato com a borda
            # agrupar apenas os contornos longe da borda
            # se algum contorno em contato com a borda se aproximar do
            # objeto, considerar como sendo a mão do operador.
        
        

        print(f'foram encontrados {len(contours)} contornos.')
        for contour in contours:
            area = cv2.contourArea(contour)
            # if area > MIN_CONTOUR_AREA:
            # Obter as coordenadas do retângulo delimitador
            x, y, w, h = cv2.boundingRect(contour)

            # Desenhar contorno válido e exibir mensagem
            # x, y, w, h = cv2.boundingRect(valid_contour)
            cv2.rectangle(preproc_image, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        # Mostrar o reduced para depuração
        # cv2.imshow("Frame", reduced[50:400, 350:600])
        # cv2.imshow("Mascara", clean_mask[50:400, 350:600])
        clean_mask_3ch = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        preproc_image_3ch = cv2.cvtColor(preproc_image, cv2.COLOR_GRAY2BGR)

        # Calcular o tempo decorrido em segundos
        time_elapsed = frame_count / fps
        minutes = int(time_elapsed // 60)
        seconds = int(time_elapsed % 60)

        # Adicionar o tempo como overlay no frame
        overlay_text = f"Tempo: {minutes:02}:{seconds:02}"
        cv2.putText(preproc_image_3ch, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        if len(contours):
            cv2.rectangle(preproc_image_3ch,
                          final_bounding_box,
                        #   (x_min, y_min),
                        #   (x_min + (x_max - x_min), y_min + (y_max - y_min)),
                          (0, 255, 0), 2)
        combined_view = cv2.hconcat([preproc_image_3ch, clean_mask_3ch])
        # clean_mask_3ch

        # _____________________
        # Separando código para visualizar resultados durante o desenvolvimento
        
        # Debug - visualizar imagem filtrada em escala de cinza
        cv2.imshow("img", preproc_image)

        cv2.imshow("Preprocessed image (Left) vs Object Mask (Right)",
                   combined_view)

        frame_count += 1
        # Finalizar com tecla 'q'
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    # Decifrar as causas de erros
    if system_status == "errorCamera":
        print('erro na câmera')

    # print(frame.shape)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
