import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: Descrever os objetivos diários também:
# segunda-feira 09/12 - terminar o statemachine e o subtrator de background
# terça-feira 10/12 - Adicionar identificação por histograma e revisar
# a máquina de estados, para permitir a tomada de novas imagens depois do
# objeto ser rotacionado no campo visual, sem ser removido da imagem
# quarta-feira 11/12- modificar a máquina de estados para identificar quando
# o operador pega o objeto por meio das bounding boxes em contato com as
# bordas da imagem. Localizar bounding boxes com centroides variando no tempo
# com e sem contato com as bordas da imagem.
# quinta-feira 19/12- Fazer vídeos com o mock do instrumento, replicando cada
# um dos desafios de detecção listados no readme do projeto
# Sexta-feira 20/12 - integrar tudo nos moldes OOP. Mandar o resumo a Jeff!

# TODO: O serviço de identificação foi treinado com imagens quadradas. Ideia:
# Utilizar o centro da bounding box para extrair um ROI quadrado, que
# contenha completamente o objeto adicionando uma margem de 20 ou mais pixels.

# Resolvido problemas de ruído de cor. Descartar imagens com iluminação
# baixa. Padronizar iluminação resolve parcialmente. Iluminação usada segue
# o padronizado pelo ministério do trabalho, pela NR-17, e ABNT, pela NBR5413.
# Videos com o instrumento mock padronizaram o nível mínimo de iluminação para
# 800 lux, acima dos 750 lux exigidos para um ambiente de trabalho como o do
# CME. Foi usado um aplicativo de celular para fazer a medição. A margem de
# 50 lux foi para contornar algum problema de calibração da câmera de celular
# usada para a medição do nível de luminosidade.

# Filtragem adicional usada foi a Bilateral. Filtrou ruídos tipo "shot noise"
# e suavizou o "temporal noise", característicos de sensores tipo CCD ou CMOS.
# Testado pré-processamento antes de aplicar o subtrator, como equalização
# de histograma grayscale para aumentar o contraste, mas não funciona, pois
# adicionou muito ruído de imagem.
# Para processar a máscara, usar operações morfológicas para melhorar a
# qualidade da máscara calculada. Testado dilatação, evitou quebras de
# contiguidade do objeto em cena. Funcionou melhor do que usar o Fechamento,
# pois evitou a divisão do contorno de objetos presentes na cena devido a
# variações pequenas de cor.
# Usar Fechamento causou quebras de contorno e partiu o contorno do objeto
# em algumas situações de sombreamento em objetos no campo de visão.


# TODO: Melhorar a identificação depois do contour detection, aplicando
# métodos de cor e histograma,
# Contour Detection encontra os bounding boxes, mas devem ser filtrados.
# Podem ser usados os critérios de área mínima em pixels e proximidade dos
# contornos às bordas da imagem.
# Métodos de cor e Histograma obtém perfil do background no início do
# funcionamento do sistema, e identificam facilmente a cor dos objetos
# inseridos na imagem.

# Comparado o subtrator tanto o MOG2 com o KNN, com as conclusões de
# não usar KNN por gerar mais ruído de pixels. MoG2 será configurado para
# ter taxa de aprendizado (learningRate) variando, de acordo com a condição
# do ambiente.
# Primeiro caso a resolver é do de alteração brusca de iluminação, que cria
# vários artefatos de identificação pequenos. O segundo é quando o objeto
# está presente na cena desde o início, atrapalhando a aquisição do frame
# de referência. Tornar o learningRate muito alto (0.9~1.0), menos frames
# são usados para calcular o background, e o algoritmo se adapta mais
# rápido às novas condições.
# Ideia de implementação é iniciar o programa com taxa muito alta e baixar
# a taxa para muito lenta quando se perceber alguma movimentação de objeto.

# TODO: Determinar a região de interesse (ROI) como o bounding box esperado
# para o instrumento, como o maior presente na tela. Também adicionar a
# restrição de que ele estará próximo ao centro da imagem, ao definir
# margem de 10% da dimensão horizontal ou vertical em relação às bordas, para
# simplificar a aplicação dos algoritmos ignorando as regiões laterais

# Gravados novos vídeos com o mock de instrumental de laparoscopia
# Não foi possível reajustar a altura da câmera para não pegar o suporte
# Ver como se faz para calibrar a câmera para corrigir distorções
# tipo "pincushion" que produzem distorção esférica em torno do centro
# da imagem
# Refazer os vídeos com as seguintes condições:
# - Entrada e saída simples de objeto do campo de visão da câmera
# - Entrada, e rotação do objeto antes da saída
# - Entrada e saída de 2x do mesmo objeto, para identificar 2 objetos
# diferentes
# - Mudanças de iluminação
# - Baixa iluminação

# Resumo dos parâmetros do subtrator tipo MixOfGaussians2
#
# Parâmetro history: Define número de frames para calcular o background.
# Aumentar o valor torna o sistema mais robusto contra mudanças
# rápidas, mas com resposta mais lenta a objetos que se tornam estáticos.
# Padrão = 500
#
# Parâmetro varThreshold: Controla a sensibilidade para decidir se um pixel
# pertence ao foreground. Aumente para reduzir ruídos de iluminação.
# Limiar para decisão de "objeto em movimento" com base na variância das
# Gaussianas do modelo.
# Padrão = 16.0
# Valores menores: Mais sensível a pequenas variações, como ruídos e
# leves mudanças no fundo.
#
# Parâmetro detectShadows: Ativa (Default = True) para lidar com sombras. Isso
# classifica pixels de sombra como "quase foreground" com um tom mais escuro.

# fgbgKNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
# Resumo dos parâmetros do subtrator tipo KNN
# dist2Threshold: Define a distância para considerar um pixel como parte do
# foreground. Valores maiores tornam o método mais tolerante a variações.
# Parâmetro history: Funciona identicamente ao MOG2.

# Desenhar o código com princípios de Orientação a Objeto
# Tarefas do programa:
# - Inicializar e checar sistema
# - pré-processar o frame
# - identificar contorno
# - verificar se o contorno está imóvel
# - verificar se o contorno não tem contato com a borda da imagem (braço/mão)
# - Se as duas verificações forem verdadeiras, capturar uma imagem centrada
# em torno do centróide do bounding box da detecção do objeto (o maior objeto
# encontrado deve ser o instrumento), gerando uma imagem quadrada, similar ao
# formato usado como input da rede neural de identificação
# - Se houver a identificação de novos contornos depois das duas verificações,
# retornar ao estado de pré-processar frame

# Vídeos originais gravados em 4k. Reduzir resolução para a cópia usada
# para calcular os contornos? Reencodados videos para Full HD para testar
# os algoritmos, depois usar os vídeos originais

pasta = "video\\videosMock\\"

# cap = cv2.VideoCapture(r"video\video.mp4")
# cap = cv2.VideoCapture(r"video\teste.mp4")

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
objeto23 = "TemperaturaCor3k_9k.mp4"  # - para gerar valores HSV do background
objeto24 = "ContrasteTemperaturaCor3k_9k.mp4"  # - valores HSV do background

objeto = objeto23

cap = cv2.VideoCapture(pasta + objeto)


def config_capture():
    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=1500, varThreshold=50, detectShadows=True)
    return fgbgMOG2


def find_object_at_start(image):
    pass
    # find contours of objects before background subtraction. If there are any
    # contours, max out the learningRate parameter to 1 until no object is
    # present in the image


def preprocess_image(image):
    # retorna imagem filtrada
    # filtrar contra ruído 'sal e pimenta' para os casos com  baixa iluminação

    # verifica se image está no formato 16:9
    height, width = image.shape[0:2]
    # imagem da câmera BRIO vem em 4k (2160 x 3840)
    # ratio 9x16 = 2160/3840 = 0.56225
    ratio3x4 = 0.75
    cropleft = 0.21875
    cropright = 0.78125

    if (height/width) < ratio3x4:
        frame = image[0:height, int(cropleft * width): int(cropright * width)]
    else:
        frame = image  # Não cropar se a resolução não for 9x16
    # Pré processamento
    # Redimensionar para acelerar processamento
    reduced = cv2.resize(frame, None, fx=0.25, fy=0.25,
                         interpolation=cv2.INTER_AREA)

    # Filtrando o ruído "sal e pimenta" de ruído da câmera
    # Adicionar as descrições dos parâmetros e variar os valores
    f_bilateral = cv2.bilateralFilter(reduced, d=9, sigmaColor=75,
                                      sigmaSpace=75)
    # O filtro bilateral distorceu menos os contornos do que o filtro
    # mediana cv2.medianBlur(reduced, 3)

    # Conversão para tons de cinza
    gray_frame = cv2.cvtColor(f_bilateral, cv2.COLOR_BGR2GRAY)

    # Equalizar o histograma da imagem com uma função de opencv do tipo
    # equalizeHist() realça muitos pontos de ruído de imagem, mesmo após
    # a aplicação de filtro bilateral
    # gray_frame = cv2.equalizeHist(gray_frame)
    # Suavização para reduzir ruído - Usar apenas um nível de filtro
    # Segunda passagem quebra a definição do contorno do objeto, ainda
    # mais se for preto refletindo luz forte
    # b_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    # b_frame = cv2.GaussianBlur(b_frame, (5, 5), 0)

    return gray_frame


def analyze_hsv(frame):
    # # plotar o histograma RGB e HSV do frame atual

    # colors = ('b', 'g', 'r')
    # total_pixels = frame.shape[0] * frame.shape[1]  # Total de pixels no frame

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
    #     hist_percent = (hist.flatten() / total_pixels) * 100  # Converter para percentual
    #     accumulated_histograms[color] = hist_percent  # Acumular em percentual
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
    bg_hist = cv2.calcHist([hsv], [0, 1], bg_mask, [180, 256], [0, 180, 0, 256])
    obj_hist = cv2.calcHist([hsv], [0, 1], object_mask, [180, 256], [0, 180, 0, 256])

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
    # Recebe imagem, retorna máscara e bounding box do objeto
    #
    pass


def check_capture_conditions():
    # tem que usar vários frames para fazer essa verificação
    # Usar a máquina de estados para isso
    pass


def send_capture():
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


def check_illumination():
    pass


if __name__ == '__main__':
    fgbgMOG2 = config_capture()

    # Inicializar acumuladores de histogramas
    accumulated_histograms = {
        'b': np.zeros((256,), dtype=np.float32),
        'g': np.zeros((256,), dtype=np.float32),
        'r': np.zeros((256,), dtype=np.float32)
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Determinar um frame de referência, mas atualizá-lo a cada
        # 30 frames (1x por segundo a 30fps)?
        # analyze_hsv(frame)
        preproc_image = preprocess_image(frame)

        # b_frame = g_frame
        # Aplicar subtração de fundo
        fgmaskMOG2 = fgbgMOG2.apply(preproc_image, learningRate=0.0001)
        # Taxa de aprendizado muito reduzida = 0.0001 -> tempo muito longo
        # para recalcular background
        # Taxa = 0.001 = Recalcula background a cada 14 segundos
        # A taxa é a velocidade em que o algoritmo atualiza o background
        # Se for rápida, qualquer objeto em movimento que parar de se mover
        # na imagem será considerado background rapidamente
        # Com learningRate=0.0001, há bastante tempo para identificar uma
        # bounding box

        # Pós-Processamento
        thresh = cv2.threshold(fgmaskMOG2, 210, 255, cv2.THRESH_BINARY)[1]
        # Elimina sombras (valores mais baixos que 200)

        # Operações morfológicas para limpar ruído - mesmo testando
        # kernel 3x3, ao invés do 5x5, cortou alguns dos objetos e
        # algumas das pontas dos instrumentais
        # Ruído de imagem foi drasticamente reduzido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
        # clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.dilate(thresh, kernel, iterations=2)
        # mask = cv2.erode(mask, kernel, iterations=2)

        # Detecção de contornos na máscara calculada (justificar os parâmetros)
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
        mask_height, mask_width = clean_mask.shape  # Dimensões da imagem
        image_border = 10  # Distância mínima das bordas
        valid_contours = []
        border_contours = []
        for contour in contours:
            x_0, y_0, x_1, y_1 = cv2.boundingRect(contour)
            # Verificar se o contorno está longe das bordas
            bool_top_corner = x_0 > image_border and y_0 > image_border
            bool_bot_corner1 = (x_0 + x_1) < mask_width - image_border
            bool_bot_corner2 = (y_0 + y_1) < mask_height - image_border

            if bool_bot_corner1 and bool_bot_corner2 and bool_top_corner:
                valid_contours.append(contour)
            else:
                border_contours.append(contour)

        # TODO>> aparecem dezenas de pequenos contornos quando a mão traz um
        # novo objeto. Como filtrar os menores contornos?
        # Depois, identificar o maior contorno pois o esperado é que apenas um
        # instrumento esteja no centro da imagem. Usar o bounding box para delimitar
        # uma região de interesse e salvar apenas uma cópia dessa ROI.
        # Enquanto houver um objeto na mesa, comparar o objeto com a cópia da ROI
        #

        print(f'foram encontrados {len(contours)} contornos.')
        for contour in contours:
            area = cv2.contourArea(contour)
            # if area > MIN_CONTOUR_AREA:
            # Obter as coordenadas do retângulo delimitador
            x, y, w, h = cv2.boundingRect(contour)

            # Desenhar contorno válido e exibir mensagem
            # x, y, w, h = cv2.boundingRect(valid_contour)
            cv2.rectangle(preproc_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar o reduced para depuração
        # cv2.imshow("Frame", reduced[50:400, 350:600])
        # cv2.imshow("Mascara", clean_mask[50:400, 350:600])
        clean_mask_3ch = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        preproc_image_3ch = cv2.cvtColor(preproc_image, cv2.COLOR_GRAY2BGR)
        if len(contours):
            cv2.rectangle(preproc_image_3ch, (x_min, y_min), (x_min + (x_max - x_min), y_min + (y_max - y_min)), (0, 255, 0), 2)
        combined_view = cv2.hconcat([preproc_image_3ch, clean_mask_3ch])  # clean_mask_3ch

        cv2.imshow("Preprocessed image (Left) vs Object Mask (Right)", combined_view)

        # Finalizar com tecla 'q'
        if cv2.waitKey(130) & 0xFF == ord("q"):
            break

    # Finalizar e fechar janela gráfica
    cap.release()
    plt.ioff()
    plt.show()

    # print(frame.shape)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# =========================
# Código de teste de comparação entre MOG2 e KNN
#
# arquivo de referência -- "cabo luz off.mp4"
#
# Houve mudança significativa de desempenho entre as
# duas técnicas. Em geral, MOG2 é cerca de 15%~20% mais lento para
# calcular a diferença entre foreground e background
#
# Qualitativamente, a MOG2 tem imunidade a ruído melhor que a
# versão KNN em todos os testes. No KNN, o ruído está sempre
# presente, mas o objeto fica destacado por mais tempo depois
# de parar em cena. Foi testado em baixa iluminação
#
# A persistência do objeto parado no método MOG2 pode ser compensado
# com um parâmetro learningRate igual a 0.001, mas esse parâmetro
# aplicado ao modelo KNN piora muito a condição de identificação
# nos testes de objeto em cena desde o começo e de mudança na iluminação,
# com grande impacto no desempenho (58% mais lento que o MOG)
#
# O parâmetro learningRate=0.01 no método MOG2 é rápido demais. Antes da
# mão do operador se afastar do objeto, o MOG considera o objeto como
# background.
#

# import time

# mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
# knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

# mog2_time, knn_time = 0, 0
# frame_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.GaussianBlur(frame, (5, 5), 0)
#     frame_count += 1

#     # MOG2
#     start = time.time()
#     mog2_mask = mog2.apply(frame, learningRate=0.0001)
#     mog2_mask = cv2.threshold(mog2_mask, 200, 255, cv2.THRESH_BINARY)[1]
#     # Ruído de imagem foi drasticamente reduzido
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
#     mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_CLOSE, kernel)
#     mog2_time += time.time() - start

#     # KNN
#     start = time.time()
#     knn_mask = knn.apply(frame, learningRate=0.01)
#     knn_mask = cv2.threshold(knn_mask, 200, 255, cv2.THRESH_BINARY)[1]
#     # Ruído de imagem foi drasticamente reduzido
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
#     knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_CLOSE, kernel)
#     knn_time += time.time() - start

#     # Exibir comparativo
#     combined_view = cv2.vconcat([mog2_mask, knn_mask])
#     cv2.imshow("MOG2 (Left) vs KNN (Right)", combined_view)

#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# print(f"MOG2 Average Time per Frame: {mog2_time / frame_count:.5f} seconds")
# print(f"KNN Average Time per Frame: {knn_time / frame_count:.5f} seconds")
# print(f"Percentual difference of MOG2 in relation to KNN: {(mog2_time/knn_time - 1) * 100}%")

# --------------------------------
# Videos de condições de imagem
# Colocar um objeto, aguardar 3 segundos e recolher - v
# Colocar um objeto, aguardar 15 segundos e recolher - v
# Colocar um objeto, manter a mão sobre o objeto e recolher - v
# Colocar um objeto, passar a mão sobre o objeto e recolher - v
# Colocar um objeto, reposicionar e recolher - v
# Colocar um objeto, substituir por outro - v
# Colocar um objeto, passar uma sombra forte e recolher
# Iniciar a captura com um objeto na cena, remover e recolocar o objeto - v
# Objeto em posições ortogonais e diagonais - v
# Alterar bruscamente a iluminação - v

# CME é ambiente com iluminação mínima controlada. Mas ainda podem haver
# alterações de iluminação durante o dia por causa de pequena janelas
# perto da área de trabalho.
# Encontrada uma fonte que referencia as normas técnicas de iluminação
# em ambientes de trabalho. Iluminação para ambientes de visualização
# crítica deve ter entre 750 e 1000 lux de iluminação

# Setup de simulação usou uma fonte de iluminação WRGB LED, uma câmera
# Logitech BRIO, um suporte estável e um lençol verde cedido pelo CME
# para a simulação do ambiente de trabalho.

# Foi utilizado aplicativo de celular para ajustar a intensidade de luz
# para a faixa de 850 lux, como margem à provavel erro de calibração do
# aplicativo para a câmera de celular utilizada.

# O teste de variação brusca de iluminação será entre 800 lux a 560 lux,
# de 25% para 15% da potência máxima de iluminação


# Pasta com os arquivos de vídeo
# import os
# main_path = r"video\videosMock"
# files = os.listdir(main_path)
# for file in files:
#     print(file)