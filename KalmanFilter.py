import cv2
import numpy as np

cap = cv2.VideoCapture(r"video\video.mp4")

# Inicializar o filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objeto (aqui como exemplo uma detecção simplificada)
    measurement = np.array([[np.float32(320)], [np.float32(240)]])  # exemplo de posição
    prediction = kalman.predict()
    kalman.correct(measurement)

    # Desenhar posição prevista
    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 10, (0, 255, 0), -1)

    cv2.imshow('Kalman Filter Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
