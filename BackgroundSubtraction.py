import cv2

# Configurações do sistema - Definir tamanho do frametest com o AT
FRAME_WIDTH = 480
FRAME_HEIGHT = 640
CENTER_TOLERANCE = 100  # Tolerância para centralizar o objeto(em pixels)
# MIN_CONTOUR_AREA = 10  # Área mínima para um contorno ser objeto válido?
MOTION_THRESHOLD = 1  # Limite para considerar movimento. Filtra ruído de
# detecção de borda

# TODO: O serviço de identificação foi treinado com imagens quadradas. Ideia:
# Utilizar o centro da bounding box para extrair um ROI quadrado, que
# contenha completamente o objeto adicionando uma margem de 20 ou mais pixels.

# TODO: Resolver problemas encontrados.
# Em imagens com iluminação diminuída, é visível
# o aparecimento de ruídos de imagem, como ruído
# "tipo chuvisco de TV analógica",
# perturbando os valores de cor dos pixels.
# Quando a iluminação spot foi acionada, a relação sinal-ruído é alta e
# minimiza o aparecimento do ruído na imagem. Mas aparece sombra intensa
# da estrutura de suporte sobre o lençol

# TODO: Utilizar como forma de melhorar a identificação o contour detection,
# métodos de cor e histograma, e o filtro de kalman para operar sobre a
# máscara extraída pelo subtrator de background.
# Contour Detection para encontrar os maiores bounding boxes.
# Métodos de cor e Histograma para obter perfil do background no início do
# funcionamento do sistema, e identificar facilmente a cor dos objetos
# inseridos na imagem, para usar essa informação em conjunto com os bounding
# boxes encontrados para localizar o objeto na imagem.
# O filtro de Kalman pode ser usado para prever movimento de objetos ou
# das mãos, com o desafio de identificar os movimentos mais rápidos de
# retirada das mãos do campo de imagem.

# TODO: Configurar o subtrator tanto o MOG2 quanto o KNN para ter taxa de
# aprendizado (learningRate) variando, de acordo com a condição do ambiente.
# Primeiro caso a resolver é do de alteração brusca de iluminação, que cria
# vários artefatos de identificação pequenos. O segundo é quando o objeto
# está presente na cena desde o início, atrapalhando a aquisição do frametest
# de referência. Tornar o learningRate muito alto (0.9~1.0), menos frames
# são usados para calcular o background, e o algoritmo se adapta mais
# rápido às novas condições.

# TODO: Testar pré-processamento antes de aplicar o subtrator. Testar
# filtros passa-baixas (atrapalha definir contorno dos objetos) e equalização
# de histograma para aumentar o contraste.

# TODO: Para processar a máscara, usar operações morfológicas para melhorar a
# qualidade da máscara calculada

# TODO: Comparar os dois métodos de subtratores

# TODO: Determinar a região de interesse (ROI) no centro da imagem, ao definir
# margem de 10% da dimensão horizontal ou vertical em relação às bordas, para
# simplificar a aplicação dos algoritmos ignorando as regiões laterais

# TODO: Como os objetos geralmente são maiores ou de tamanho similar ao das
# mãos, selecionar os maiores bounding boxes encontrados, analisar se não é
# background pelo histograma e excluir os outros bounding boxes.

# TODO: Avaliar a combinação de learning rates mais curtos com filtros
# temporais

# TODO: 

# Inicializar a captura de vídeo e o subtrator de fundo

# Vídeos originais gravados em 4k. Reduzir resolução para a cópia usada
# para calcular os contornos? Reencodados videos para Full HD para testar
# os algoritmos, depois usar os vídeos originais

pasta = "video\\videosMock\\"

# cap = cv2.VideoCapture(r"video\video.mp4")
# cap = cv2.VideoCapture(r"video\teste.mp4")
cap = cv2.VideoCapture(pasta + "cabo luz off.mp4")
# cap = cv2.VideoCapture(pasta + "cabo movimento maos luz on.mp4")
# cap = cv2.VideoCapture(pasta + "caixa clara movimento maos luz on.mp4")
# cap = cv2.VideoCapture(pasta + "caixa desde inicio luz on.mp4")
# cap = cv2.VideoCapture(pasta + "caixa luz off.mp4")
# cap = cv2.VideoCapture(pasta + "caixa mudanca iluminacao.mp4")
# cap = cv2.VideoCapture(pasta + "Paquimetro luz off.mp4")
# cap = cv2.VideoCapture(pasta + "Paquimetro mao presente luz off.mp4")
# cap = cv2.VideoCapture(pasta + "Paquimetro para caixa luz off.mp4")
# cap = cv2.VideoCapture(pasta + "Regua luz off.mp4")
# cap = cv2.VideoCapture(pasta + "regua refletiva luz on.mp4")

# fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(
#             history=500, varThreshold=10, detectShadows=True)

# # Resumo dos parâmetros do subtrator tipo MixOfGaussians2
# #
# # Parâmetro history: Define número de frames para calcular o background.
# # Aumentar o valor torna o sistema mais robusto contra mudanças
# # rápidas, mas com resposta mais lenta a objetos que se tornam estáticos.
# # Padrão = 500
# #
# # Parâmetro varThreshold: Controla a sensibilidade para decidir se um pixel
# # pertence ao foreground. Aumente para reduzir ruídos de iluminação.
# # Limiar para decisão de "objeto em movimento" com base na variância das
# # Gaussianas do modelo.
# # Padrão = 16.0
# # Valores menores: Mais sensível a pequenas variações, como ruídos e
# # leves mudanças no fundo.
# #
# # Parâmetro detectShadows: Ativa (Default = True) para lidar com sombras. Isso
# # classifica pixels de sombra como "quase foreground" com um tom mais escuro.

# fgbgKNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
# # Resumo dos parâmetros do subtrator tipo KNN
# # dist2Threshold: Define a distância para considerar um pixel como parte do
# # foreground. Valores maiores tornam o método mais tolerante a variações.
# # Parâmetro history: Funciona identicamente ao MOG2.

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Pré processamento - redimensionar para visualizar apenas ao final
#     # para não perder informações com os resizes.
#     frametest = cv2.resize(frame, None, fx=0.25, fy=0.25,
#                            interpolation=cv2.INTER_AREA)

#     # Conversão para tons de cinza
#     g_frame = cv2.cvtColor(frametest, cv2.COLOR_BGR2GRAY)

#     # Suavização para reduzir ruído - Usar apenas um nível de filtro
#     # Segunda passagem quebra a definição do contorno do objeto, ainda
#     # mais se for preto refletindo luz forte
#     b_frame = cv2.GaussianBlur(g_frame, (5, 5), 0)
#     # b_frame = cv2.GaussianBlur(b_frame, (5, 5), 0)

#     # b_frame = g_frame
#     # Aplicar subtração de fundo
#     fgmaskMOG2 = fgbgMOG2.apply(b_frame, learningRate=0.001)
#     # Taxa de aprendizado muito reduzida = 0.0001 -> tempo muito longo para recalcular background
#     # Taxa = 0.001 = Recalcula background a cada 14 segundos
#     # A taxa é a velocidade em que o algoritmo atualiza o background
#     # Se for rápida, qualquer objeto em movimento que parar de se mover na imagem será considerado background rapidamente
#     # Com learningRate=0.0001, há bastante tempo para identificar uma bounding box

#     # Pós-Processamento
#     thresh = cv2.threshold(fgmaskMOG2, 200, 255, cv2.THRESH_BINARY)[1]  # Elimina sombras (valores mais baixos que 200)

#     # Operações morfológicas para limpar ruído - mesmo testando
#     # kernel 3x3, ao invés do 5x5, cortou alguns dos objetos e
#     # algumas das pontas dos instrumentais
#     # Ruído de imagem foi drasticamente reduzido
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     # Detecção de contornos na máscara calculada
#     contours, _ = cv2.findContours(
#         clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     valid_contour = None

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         # if area > MIN_CONTOUR_AREA:
#         # Obter as coordenadas do retângulo delimitador
#         x, y, w, h = cv2.boundingRect(contour)

#         # Desenhar contorno válido e exibir mensagem
#         # x, y, w, h = cv2.boundingRect(valid_contour)
#         cv2.rectangle(frametest, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Mostrar o frametest para depuração
#     cv2.imshow("Frame", frametest)
#     cv2.imshow("Mascara", clean_mask)

#     # Finalizar com tecla 'q'
#     if cv2.waitKey(130) & 0xFF == ord("q"):
#         break

# cv2.waitKey(0)
# cap.release()
# cv2.destroyAllWindows()

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

import time

mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

mog2_time, knn_time = 0, 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_count += 1

    # MOG2
    start = time.time()
    mog2_mask = mog2.apply(frame, learningRate=0.0001)
    mog2_mask = cv2.threshold(mog2_mask, 200, 255, cv2.THRESH_BINARY)[1]
    # Ruído de imagem foi drasticamente reduzido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
    mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_CLOSE, kernel)
    mog2_time += time.time() - start

    # KNN
    start = time.time()
    knn_mask = knn.apply(frame, learningRate=0.01)
    knn_mask = cv2.threshold(knn_mask, 200, 255, cv2.THRESH_BINARY)[1]
    # Ruído de imagem foi drasticamente reduzido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # fgmaskMOG2 = cv2.morphologyEx(fgmaskMOG2, cv2.MORPH_OPEN, kernel)
    knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_CLOSE, kernel)
    knn_time += time.time() - start

    # Exibir comparativo
    combined_view = cv2.vconcat([mog2_mask, knn_mask])
    cv2.imshow("MOG2 (Left) vs KNN (Right)", combined_view)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"MOG2 Average Time per Frame: {mog2_time / frame_count:.5f} seconds")
print(f"KNN Average Time per Frame: {knn_time / frame_count:.5f} seconds")
print(f"Percentual difference of MOG2 in relation to KNN: {(mog2_time/knn_time - 1) * 100}%")
