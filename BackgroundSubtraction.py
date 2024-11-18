import cv2

# Inicializar a captura de vídeo e o subtrator de fundo
cap = cv2.VideoCapture(r"video\video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar subtração de fundo
    fgmask = fgbg.apply(frame)

    # Mostrar o resultado
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
