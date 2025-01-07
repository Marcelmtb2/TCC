import cv2

# Vídeos originais gravados em 4k. Reduzir resolução para a cópia usada
# para calcular os contornos? Reencodados videos para Full HD para testar
# os algoritmos, depois usar os vídeos originais

pasta = "video\\videosMock\\"

# cap = cv2.VideoCapture(r"video\video.mp4")
# cap = cv2.VideoCapture(r"video\teste.mp4")
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
objeto23 = "TemperaturaCor3k_9k.mp4"
# - para gerar valores HSV do background
objeto24 = "ContrasteTemperaturaCor3k_9k.mp4"
# - valores HSV do background

# selecionar o video do objeto
objeto = objeto18
# ====================================
# Inicializar captura de imagens

cap = cv2.VideoCapture(pasta + objeto)

# cap = cv2.VideoCapture(r"video\video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # fgmask = fgbg.apply(frame)
    # contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Contours', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
