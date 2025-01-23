import cv2
import numpy as np

cap = cv2.VideoCapture(r"video\teste.mp4")
ret, frame = cap.read()
x, y, w, h = 200, 150, 50, 50  # Inicialize a área de rastreamento
track_window = (x, y, w, h)

# Calcular o histograma do ROI (Região de Interesse)
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Configuração de critérios para o Camshift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Aplicar o algoritmo Camshift
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int32(pts)
    img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow('Camshift Tracking', img2)

    if cv2.waitKey(130) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
