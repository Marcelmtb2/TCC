import cv2
import numpy as np


def show_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique esquerdo do mouse
        b, g, r = param[y, x]
        # qprint(f"R: {r}, G: {g}, B: {b}")
    # Converte RGB para HSV
        # cria um pixel no formato BGR do OpenCV
        pixel_rgb = np.uint8([[[b, g, r]]])
        pixel_hsv = cv2.cvtColor(pixel_rgb, cv2.COLOR_BGR2HSV)[0][0]

        h, s, v = pixel_hsv
        print(f"R: {r}, G: {g}, B: {b} | H: {h}, S: {s}, V: {v}")


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

    # Variável para controlar pausa
    paused = False

    # Carrega o vídeo
    video_path = pasta + objeto  # Substitua pelo caminho do vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        exit()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler o frame.")
                break
        # verifica se image está no formato 16:9
        height, width = frame.shape[0:2]
        # imagem da câmera BRIO vem em 4k (2160 x 3840)
        # ratio 9x16 = 2160/3840 = 0.56225
        ratio3x4 = 0.75
        cropleft = 0.21875
        cropright = 0.77  # cortar ponta do suporte. 0.78125 original

        if (height/width) < ratio3x4:
            image = frame[0:height, int(cropleft * width): int(cropright * width)]
        else:
            image = frame  # Não cropar se a resolução não for 9x16

        # Redimensionar para acelerar processamento
        reduced = cv2.resize(image, None, fx=0.25, fy=0.25,
                             interpolation=cv2.INTER_AREA)

        # Filtrando o ruído de digitalização da câmera
        # Adicionar as descrições dos parâmetros e variar os valores
        f_bilateral = cv2.bilateralFilter(reduced, d=9, sigmaColor=75,
                                          sigmaSpace=125)
        cv2.imshow("Video Frame", reduced)
        cv2.setMouseCallback("Video Frame", show_rgb, reduced)

        # Controle de teclas
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Pressione 'q' para sair
            break
        elif key == ord(' '):  # Pressione espaço para pausar/retomar
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
