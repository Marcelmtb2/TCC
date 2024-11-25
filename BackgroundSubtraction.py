import cv2

# Inicializar a captura de vídeo e o subtrator de fundo
cap = cv2.VideoCapture(r"video\video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()
# Habilitando identificação de sombras
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Pré processamento
    # Conversão para tons de cinza
    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suavização para reduzir ruído
    b_frame = cv2.GaussianBlur(g_frame, (5, 5), 0)

    # Aplicar subtração de fundo
    fgmask = fgbg.apply(b_frame)

    # Pós-Processamento
    # Remover sombras
    fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    # Operações morfológicas para limpar ruído - mesmo testando
    # kernel 3x3, ao invés do 5x5, cortou alguns dos objetos e
    # algumas das pontas dos instrumentais
    # Ruído de imagem foi drasticamente reduzido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Mostrar o resultado
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
