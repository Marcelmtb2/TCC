# TCC
Code for supporting Course Completion Thesis

https://www.overleaf.com/project/66756f7949fa86f5621092dd - main document

https://docs.google.com/spreadsheets/d/1Ei4KCWnIAcxXeuaU7s_CjdYAXi2lLIhsc_yA3M1kWf8/edit?gid=0#gid=0 - related documents

https://docs.google.com/spreadsheets/d/1-JDm41Z7ekAmB6yn_b-J0OSfbQO-CdqjbCHCk1QTmv4/edit?usp=sharing - Reading list

## Testing code

- Background subtraction
- Contour Detection
- HSV Histogram
- Feature Matching*
- Kalman Filter*
- Optical FLow*

## Journal

18-nov -> 1st commits of initial code.

02-dez -> update on advances

Background subtraction is functional.
Comparison between Mix of Gaussians and K-Nearest Neighbors showed K-NN is more demanding
Background noise and a lot of shadow movements were removed by image filtering and
proper Background Subtractor configuration.

Object detection will follow 4 steps: 
1st - Background Subtractor calculates a mask of moving objects entering the scene. It is configured
for maintaining the image of the object up to 10 seconds after stopping in the scene.

2nd - Contour detection methods will draw bounding boxes around objects detected. The greater will be
the first to be analyzed. If there is a distinct color histogram from the background, other contours
are discarded and the detected one is accepted as a valid object

3rd - A State Machine will be applied, to evaluate whether the conditions for capturing images have been met.
If all conditions are satisfied, a square region of interest (ROI) will be selected to match
the input format required by the surgical instrument identification service.

More experiments will be made in code for Contour Detection, HSV Histogram and Kalman Filters, as them seem easier and 
faster to progress

Following initial work, Optical Flow and Feature Matching will be discontinued. 
Optical flow does not applies well in a scenario in which may have illumination changes or rapid movement of objects.
Feature matching needs to apply an extensive amount of masks, in different format, sizes and directions in each frame,
before detecting the presence of an object entering the scene. It will consume more time than the application demands.

# New Video examples

It was used an open-source application named BORIS, designed as an annotation tool for biological and animal behaviour.

The annotations of the first version are stored, but it will not be used at the first version.

The annotated video had the problem of unstable movement between camera and background. 

It was built a stable platform for acquiring images, in better controlled conditions.

Shorter videos were captured, with low and high illumination conditions, testing the following conditions:
1 - simple object entering the scene, after occasional oclusion of a hand
2 - substitition of objects in scene, one after another.
3 - repositioning of surgical tool in the scene (rotations and translations)
4 - object into scene since the beginning of the video
5 - sudden changes of illumination

# Major tests

## Comparison between KNN and MOG2 methods for the background subtractor.

The configuration detectShadows must be True, in order to filter small
variations in brightness in any lightning conditions, in both methods.
After detection of objects and their shadows, it will be used a threshold
to remove the detected shadows of the final mask.

1. Critérios de Avaliação
- Velocidade (Desempenho Computacional)
Compare o tempo necessário para processar cada frame com os dois métodos.
Use a função cv2.getTickCount() para medir o tempo.
- Precisão (Qualidade de Detecção)
Avalie como cada método separa o foreground do background:

Taxa de Falsos Positivos: Áreas identificadas como foreground erroneamente.
Taxa de Falsos Negativos: Áreas de foreground não detectadas.
Robustez: Sensibilidade a mudanças bruscas de iluminação ou ruídos na cena.
- Adaptabilidade
Teste como cada subtrator se adapta a cenários em que o objeto entra na cena, sai ou permanece estático.
Verifique a velocidade de adaptação para considerar novos objetos como parte do background.
2. Etapas de Comparação
- Configuração Inicial
Configure ambos os subtratores com parâmetros similares
- Medição de Tempo
Use o tempo médio por frame como métrica de desempenho
- Qualidade Visual
Exiba os resultados lado a lado para análise qualitativa

- Teste com Cenários Variados
Use vídeos com condições diferentes para testar a robustez:

Mudança de iluminação: Exemplo, luzes acendendo ou apagando.
Movimento de background: Ventilador ou cortinas em movimento.
Objetos estáticos por um período longo.
Após realizar os testes, você pode basear sua escolha no seguinte:

MOG2:
Geralmente mais rápido.
Melhor em lidar com mudanças suaves no background.
Pode ser mais sensível a mudanças bruscas de iluminação.
KNN:
Mais robusto em cenários com iluminação variável e ruído.
Processamento mais pesado.
Pode produzir mais falsos positivos em áreas com sombras ou luz difusa.

# State Machine

Para definir a máquina de estados que vai selecionar as condições ideais para capturar a imagem da instrumentação cirúrgica, foi desenhado o cenário básico de aplicação:

Uma câmera modelo Logitech C270 ou BRIO está à distância fixa de uma mesa de trabalho forrada com lençol esterilizado de cor verde. Se espera que o lençol tenha marcas de vincos, dobras e amassados e admitindo algumas variações de tonalidade do verde, pela condição de uso normal do CME. 

Um operador coloca sobre a mesa instrumentos cirúrgicos, um a um, evitando manter as mãos sobre a mesa, no campo de visão da câmera, até receber a confirmação que pode substituir o instrumental.

O sistema monitora a entrada de um objeto em movimento no campo de visão, aguarda que o operador afaste as mãos. Também identifica se não há algum objeto genérico se movimentando na cena ou que esteja próximo das bordas da imagem.

Devem existir as seguintes condições:

- objeto permanecer parado por pelo menos 1 segundo próximo ao centro da imagem;
- não haver oclusões parciais ou totais da mão sobre o objeto;

Se existirem, a imagem será capturada e enviada para o serviço de identificação de instrumentos cirúrgicos por IA.

As condições adversas para a captura da imagem a verificar são as seguintes:

### Estado inicial do background com algum objeto

Se houver algum objeto sobre o lençol verde no início da operação do sistema, pode haver dificuldades em estabelecer o background.

### Movimentação do objeto:

Objeto em movimento resulta em imagens borradas. Detectar variação significativa de posição entre quadros consecutivos.

### Oclusão parcial ou total:

Presença de objetos estranhos na cena, como mãos do operador cobrindo parte do objeto.
Sombras que obscurecem o objeto e o plano de fundo parcialmente ou totalmente.

### Objeto cortado na imagem:

Parte do objeto posicionada fora das bordas do campo de visão, ou objeto muito próximo das bordas da imagem (margem mínima deve ser respeitada).

### Iluminação inadequada e plano de fundo não homogêneo:

Luz insuficiente ou excesso de brilho (imagens muito escuras ou com áreas estouradas como fachos de luz do sol sobre a mesa).
Alterações bruscas na textura ou na cor do plano de fundo, que podem confundir o modelo de IA e dificultar a segmentação do objeto.

### Tamanho inadequado do objeto na imagem:

Objeto muito distante ou pequeno, dificultando a detecção de detalhes.
Objeto muito próximo, cortado pelas bordas do quadro.

### Ruído visual ou obstruções temporárias:

Outros objetos entrando e saindo do campo de visão (interferências dinâmicas).

## Estados da máquina

### Inicial
-  inicializa câmera e o modelo de background (deve contemplar o caso que o sistema seja ligado/acionado com algum objeto já sobre a mesa)
-  Configurar o sistema para os parâmetros de detecção necessários (margens, histograma esperado do background, limites de contorno, entre outros)

### Estado de Subtração de background

- Identificar a Região de Interesse (ROI) na imagem, ao estabelecer uma margem afastada das bordas, e validar a ROI:
  - ROI encontrada?
    - Sim -> avançar para o estado de verificar condições de imagem;
    - Não -> avançar para o estado de exceção.

### Estado de Verificação de Condições:
- Detecção de contornos: Objeto detectado com contorno contínuos?
- Centralidade: ROI centralizada na imagem?
- Iluminação: Histograma dentro dos limites aceitáveis?
  - Se todas as condições são atendidas -> Avançar para o estado Captura de imagem
  - Se não, avançar para o estado de exceção.
  
### Estado de exceção
- Fornecer feedback sobre o erro identificado e sugerir correções para o operador.
- Retorna ao estado inicial.

### Estado de Captura de Imagem:
- Capturar a imagem com o ROI validade e enviar a imagem para o serviço externo de identificação.
- Retornar ao estado inicial ou encerrar o ciclo.