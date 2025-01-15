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
def initialize_bg_sub(device=0):
    # Recebe o endereço do dispositivo, se arquivo de video ou stream da
    # câmera do computador.
    # Retorna os handlers do objeto da câmera e o da instância do objeto
    # backgroundsubtractor
    foregroundbackgroundMOG2 = config_object_capture()
    camera = cv2.VideoCapture(device)
    return [camera, foregroundbackgroundMOG2]


def config_object_capture():

    # Inicializar o subtrator de background com valores default
    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)

    # All default values were confirmed, as testing history and varThreshold
    # do not altered much the final result
    return fgbgMOG2


def is_object_at_image(image, show_overlay=False):
    # Verificar se o primeiro frame contém contornos de um objeto.
    # Configurar o método de subtração de background com uma taxa
    # de aprendizado muito alta, para adaptar rapidamente o plano
    # de fundo médio até que o objeto seja removido.
    # Para detectar bordas, usar threshold em imagens convertidas para escala
    # de cinza ou aplicar filtro Canny para evidenciar bordas de contornos
    # Recebe imagem pré-processada

    # Descobrir como fazer o threshold dinâmico! Para contornar condições
    # de baixa iluminação

    # Se o objeto mais escuro que o plano de fundo verde, em escala de cinza
    thresh_dark = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

    # Se o objeto mais claro que o plano de fundo verde, em escala de cinza
    thresh_light = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]

    # Máscara final é a combinação bitwise or das duas máscaras. Histerese
    # tenta escapar de sombras sobre o lençol verde.
    thresh = cv2.bitwise_or(thresh_dark, thresh_light)

    # se algum contorno for identificado, retornar true
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours):
        if show_overlay:
            # Copy image to avoid modifying it
            output_image = image.copy()

            # Show binary masks
            cv2.imshow("mascara dark", thresh_dark)
            cv2.imshow("mascara light", thresh_light)
            cv2.imshow("mascara lightORdark", thresh)
            width_img_gray, height_img_gray = output_image.shape[0:2]

            # Draw all contours found
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Problema: Se houver sombra nítida, será confundido com
                # um objeto. Usar "margem de histerese" ao definir os thresholds
                # das máscaras de objetos claros e objetos escuros
                if w >= 0.9 * width_img_gray:  # se um contorno de sombra atravessa
                    # a imagem de um lado a outro
                    return [False, contours]
                cv2.rectangle(output_image, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
            # visualize the binary image
            cv2.imshow('Binary image', output_image)
            # cv2.waitKey(0)
        return [True, contours]
    return [False, contours]
    # find contours of objects before background subtraction. If there are any
    # contours, max out the learningRate parameter to 0.1 until no object is
    # present in the image


def preprocess_image(image, show_overlay=False):
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
    
    f_bilateral = cv2.GaussianBlur()

    # Conversão para tons de cinza
    gray_frame = cv2.cvtColor(f_bilateral, cv2.COLOR_BGR2GRAY)

    # Debug - ver a imagem original, antes de filtrar
    if show_overlay:
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
    clean_mask = cv2.dilate(thresh_mask, kernel, iterations=4)
    return clean_mask


def identify_contours(object_found_binary_mask):
    # Localiza contornos do possível objeto
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
    mask_height, mask_width = object_found_binary_mask.shape  # Dimensões da imagem
    size_border_factor = 0.01  # Distância mínima das bordas de 10 pixels
    margin_top_bot = int(mask_height * size_border_factor)
    margin_left_right = int(mask_width * size_border_factor)
    # fazer essa borda como percentual, entre 2 a 5% da dimensão H ou W?
    valid_contours = []
    border_contours = []
    final_object_box = ()
    # Detecção de contornos na máscara calculada
    contours = cv2.findContours(object_found_binary_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Verificar se cada contorno está longe das bordas da imagem
        top_left_away_border = ((x > margin_left_right) and
                                (y > margin_top_bot))

        top_right_away_border = (((x + w) < (mask_width - margin_left_right))
                                 and (y > margin_top_bot))

        bot_left_away_border = ((x > margin_left_right) and
                                ((y + h) < (mask_height - margin_top_bot)))

        bot_right_away_border = (((x + w) < (mask_width - margin_left_right)) and
                                 ((y + h) < (mask_height - margin_top_bot)))

        image_away_borders = (top_left_away_border and
                              top_right_away_border and
                              bot_left_away_border and
                              bot_right_away_border)  # True if away

        if image_away_borders:
            valid_contours.append((x, y, w, h))
        else:
            border_contours.append((x, y, w, h))

    # Criar uma lista de bounding boxes, se existirem contornos
    if len(valid_contours):
        # Agrupar bounding boxes próximas
        x_min = min([x for (x, y, w_box, h_box) in valid_contours])
        y_min = min([y for (x, y, w_box, h_box) in valid_contours])
        x_max = max([x + w_box for (x, y, w_box, h_box) in valid_contours])
        y_max = max([y + h_box for (x, y, w_box, h_box) in valid_contours])

        # Caixa circundante final do objeto estimado
        final_object_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    return valid_contours, border_contours, final_object_box


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


def locate_object(image, learning_rate=0.0001):
    # Recebe imagem preprocessada, retorna máscara e bounding box do objeto
    preproc_image = preprocess_image(image)
    clean_mask = find_foreground_object(preproc_image, learning_rate)
    (valid_boxes,
     border_boxes,
     final_object_box) = identify_contours(clean_mask)

    return valid_boxes, border_boxes, final_object_box


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
    objeto = objeto16

    # Considerar na versão final que o dispositivo deve ser uma câmera
    # Inicializar captura de imagens
    device = pasta + objeto
    # se usada a câmera principal, comentar linha anterior e descomentar
    # a linha seguinte
    # device = 0

    cap, fgbgMOG2 = initialize_bg_sub(device)

    # Inicializar a checagem de presença de objeto no início do vídeo.
    # Caso o primeiro frame do vídeo tenha contornos de objeto, manter
    # flag object_at_start_flag em True
    object_at_start_flag = True

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
            break

        # Pré-processar a imagem, filtrando ruídos e redimensionando.
        preproc_image = preprocess_image(frame)

        # verificar se há objeto desde o início será a ultima atividade!!
        # Verificar se o primeiro frame contém contornos de um objeto.
        if object_at_start_flag:
            object_at_start_flag = is_object_at_image(preproc_image)[0]
            learning_rate = 0.1
            # No primeiro loop do programa, considera que há objeto
            # presente na imagem, configurando o learningRate para valor alto,
            # maior que 0.01. Em seguida, testa se não existe contorno de
            # algum objeto na primeira imagem recebida.
            # Caso exista o objeto, manter a taxa alta.
        else:
            learning_rate = 0.0001
        # Se retornar False, manter a taxa de aprendizado baixa, para
        # aumentar a persistência do objeto na máscara calculada

        clean_mask = find_foreground_object(preproc_image, learning_rate)

        temp_a, temp_b, temp_c = identify_contours(clean_mask)

        valid_boxes, border_boxes, final_object_box = temp_a, temp_b, temp_c
        # Redundância necessária!
        # Como os objetos válidos, às vezes, não são reconhecidos com um
        # contorno único. Foi necessário combinar as coordenadas de
        # todos os bounding boxes para encontrar a região esperada para o
        # Objeto.
        # Dessa forma, quando for encontrada um final_object_box, passar
        # para a identificação de contorno na imagem original
        if valid_boxes and not border_boxes:
            mask_MOG2 = np.zeros_like(clean_mask)
            x, y, w, h = final_object_box
            cv2.rectangle(mask_MOG2, (x, y), (x + w, y + h), 255,
                          thickness=cv2.FILLED)
            #cv2.imshow("maskMOG2", mask_MOG2)
            mask_object = np.zeros_like(clean_mask)
            instant_contours = is_object_at_image(preproc_image)[1]
            # Recebe a saída contours
            if instant_contours:  # Pode existir um caso em que
                # há valid_boxes com apenas um contorno e nenhum
                # contorno identificado na imagem original
                # O restante do processo só existe se houverem
                # as duas máscaras
                x, y, w, h = cv2.boundingRect(np.vstack(instant_contours))
                cv2.rectangle(mask_object, (x, y), (x + w, y + h), 255,
                              thickness=cv2.FILLED)
                #cv2.imshow("objectContour", mask_object)
                intersection = cv2.bitwise_and(mask_MOG2, mask_object)
                union = cv2.bitwise_or(mask_MOG2, mask_object)

                intersection_area = np.sum(intersection > 0)
                union_area = np.sum(union > 0)

                iou = intersection_area / union_area
                # Calcule o índice de Jaccard (IoU)
                # Defina um limite para considerar uma boa coincidência
                threshold = 0.8
                print(iou)
                if iou > threshold:
                    print("Contorno e máscara coincidem!")
                else:
                    print("Diferença significativa entre contorno e máscara.")

# _____________________________
# Drawing is not part of final version

            # Draw valid contours in purple, final object box in green and
            # border contours in red

        # Visualização não será parte da versão final

        clean_mask_3ch = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        preproc_image_3ch = cv2.cvtColor(preproc_image, cv2.COLOR_GRAY2BGR)
        if len(valid_boxes):
            cv2.rectangle(preproc_image_3ch,
                          final_object_box,
                          (0, 255, 0), 5)
            cv2.rectangle(clean_mask_3ch,
                          final_object_box,
                          (0, 255, 0), 5)
            for ok_contour in valid_boxes:
                x, y, w, h = ok_contour
                cv2.rectangle(preproc_image_3ch, (x, y), (x + w, y + h),
                              (255, 0, 255), 2)
                cv2.rectangle(clean_mask_3ch, (x, y), (x + w, y + h),
                              (255, 0, 255), 2)
        if len(border_boxes):
            for bad_contour in border_boxes:
                x, y, w, h = bad_contour
                cv2.rectangle(preproc_image_3ch, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
                cv2.rectangle(clean_mask_3ch, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)

#         print(f'foram encontrados {len(valid_contours) + len(border_contours)}\
# contornos.')
#         print(f'foram encontrados {len(valid_contours)} contornos válidos.')
#         print(f'foram encontrados {len(border_contours)} contornos na borda.')
#         print()

        # Calcular o tempo decorrido em segundos
        time_elapsed = frame_count / fps
        minutes = int(time_elapsed // 60)
        seconds = int(time_elapsed % 60)

        # Adicionar o tempo como overlay no frame
        overlay_text = f"Tempo: {minutes:02}:{seconds:02}"
        #cv2.putText(preproc_image_3ch, overlay_text, (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            1, (0, 255, 255), 2)

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

    # print(frame.shape)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
