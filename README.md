# TCC
Code for supporting Course Completion Thesis

https://www.overleaf.com/read/qtvktdpbqmhn#b49f6f - main document

https://docs.google.com/spreadsheets/d/1Ei4KCWnIAcxXeuaU7s_CjdYAXi2lLIhsc_yA3M1kWf8/edit?gid=0#gid=0 - related documents

https://docs.google.com/spreadsheets/d/1-JDm41Z7ekAmB6yn_b-J0OSfbQO-CdqjbCHCk1QTmv4/edit?usp=sharing - Reading list

## Testing code

- Background subtraction
- Contour Detection
- HSV Histogram 

## Journal

18-nov -> 1st commits of initial code.

02-dez -> update on advances

06-dez -> Background detection achieved. Mix of Gaussians and K-Nearest Neighbours tested.
By comparing performance and image quality, the choice for detection is the MoG.

Comparison shows that K-NN is more computationally demanding, it generates more background noise
and it is more sensitive to variations in the illumination levels of the scene that the MoG method.

20-dez -> Milestone version. Machine state draft decided, with states and transitions designed.
Background detection established, Contour detection established, color histogram analysis as redundancy
for contour detection.

26-dez -> Resolvido problemas de ruído de cor. Descartar imagens com iluminação
baixa. Padronizar iluminação resolve parcialmente. Iluminação usada segue
o padronizado pelo ministério do trabalho, pela NR-17, e ABNT, pela NBR5413.
Videos com o instrumento mock padronizaram o nível mínimo de iluminação para
800 lux, acima dos 750 lux exigidos para um ambiente de trabalho como o do
CME. Foi usado um aplicativo de celular para fazer a medição. A margem de
50 lux foi para contornar algum problema de calibração da câmera de celular
usada para a medição do nível de luminosidade.
Filtragem adicional usada foi a Bilateral. Filtrou ruídos tipo "shot noise"
e suavizou o "temporal noise", característicos de sensores tipo CCD ou CMOS.
Testado pré-processamento antes de aplicar o subtrator, como equalização
de histograma grayscale para aumentar o contraste, mas não funciona, pois
adicionou muito ruído de imagem.
Para processar a máscara, usar operações morfológicas para melhorar a
qualidade da máscara calculada. Testado dilatação, evitou quebras de
contiguidade do objeto em cena. Funcionou melhor do que usar o Fechamento,
pois evitou a divisão do contorno de objetos presentes na cena devido a
variações pequenas de cor.
Usar Fechamento causou quebras de contorno e partiu o contorno do objeto
em algumas situações de sombreamento em objetos no campo de visão.
Comparado o subtrator tanto o MOG2 com o KNN, com as conclusões de
não usar KNN por gerar mais ruído de pixels. MoG2 será configurado para
ter taxa de aprendizado (learningRate) variando, de acordo com a condição
do ambiente.
Primeiro caso a resolver é do de alteração brusca de iluminação, que cria
vários artefatos de identificação pequenos. O segundo é quando o objeto
está presente na cena desde o início, atrapalhando a aquisição do frame
de referência. Tornar o learningRate muito alto (0.9~1.0), menos frames
são usados para calcular o background, e o algoritmo se adapta mais
rápido às novas condições.
Ideia de implementação é iniciar o programa com taxa muito alta e baixar
a taxa para muito lenta quando se perceber alguma movimentação de objeto.
Gravados novos vídeos com o mock de instrumental de laparoscopia
Não foi possível reajustar a altura da câmera para não pegar o suporte
Ver como se faz para calibrar a câmera para corrigir distorções
tipo "pincushion" que produzem distorção esférica em torno do centro
da imagem
Refazer os vídeos com as seguintes condições:
- Entrada e saída simples de objeto do campo de visão da câmera
- Entrada, e rotação do objeto antes da saída
- Entrada e saída de 2x do mesmo objeto, para identificar 2 objetos
diferentes
- Mudanças de iluminação
- Baixa iluminação
Resumo dos parâmetros do subtrator tipo MixOfGaussians2

Parâmetro history: Define número de frames para calcular o background.
Aumentar o valor torna o sistema mais robusto contra mudanças
rápidas, mas com resposta mais lenta a objetos que se tornam estáticos.
Padrão = 500

Parâmetro varThreshold: Controla a sensibilidade para decidir se um pixel
pertence ao foreground. Aumente para reduzir ruídos de iluminação.
Limiar para decisão de "objeto em movimento" com base na variância das
Gaussianas do modelo.
Padrão = 16.0
Valores menores: Mais sensível a pequenas variações, como ruídos e
leves mudanças no fundo.

Parâmetro detectShadows: Ativa (Default = True) para lidar com sombras. Isso
classifica pixels de sombra como "quase foreground" com um tom mais escuro.
fgbgKNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
Resumo dos parâmetros do subtrator tipo KNN
dist2Threshold: Define a distância para considerar um pixel como parte do
foreground. Valores maiores tornam o método mais tolerante a variações.
Parâmetro history: Funciona identicamente ao MOG2.

## Optical detection state machine

Object detection will follow 4 steps: 

1st - Background Subtractor calculates a mask of moving objects entering the scene. It is configured
for maintaining the image of the object up to 10 seconds after stopping in the scene.

2nd - Contour detection methods will draw bounding boxes around objects detected. The greater will be
the first to be analyzed. If there is a distinct color histogram from the background, other contours
are discarded and the detected one is accepted as a valid object

new 3rd state - After detected and ordered by size all contours, the contour must be retained. So, if
the system operator does not remove the surgical tool from scene, it should recognize that a similar
contour is still present, and not allow the capture of a new image. 
The method of choice for a first attempts will be the Hu Moments and cv2.matchshape().

4th- A State Machine will be applied, to evaluate whether the conditions for capturing images have been met.
If all conditions are satisfied, a square region of interest (ROI) will be selected to match
the input format required by the surgical instrument identification service.

More experiments will be made in code for Contour Detection, HSV Histogram and Kalman Filters, as them seem easier and 
faster to progress.

Feature matching will be used in the new approach of identifying if the object does not leave the scene, and is just 
rotated in the scene.

Following initial work, Optical Flow will be discontinued. Optical flow does not applies well in a scenario in which may have illumination changes or rapid movement of objects.


# Video examples

The first video had the problem of unstable movement between camera and background. 

It was built a stable platform for acquiring images, with better controlled conditions.

Shorter videos were captured, with low and high illumination conditions, testing the following conditions:
1 - simple object entering the scene, after occasional oclusion of a hand
2 - substitition of objects in scene, one after another.
3 - repositioning of surgical tool in the scene (rotations and translations)
4 - object into scene since the beginning of the video
5 - sudden changes of illumination

# Major tests

## Comparison between KNN and MOG2 methods for the background subtractor.

Setting equivalent default configuration. The detectShadows=True, to filter
variations in brightness in any lightning conditions, in both methods.
After detection of objects and their shadows, it will be used a threshold
to remove the detected shadows of the final mask.

1. Criteria
- Velocity:  Using cv2.getTickCount() for measuring time, MOG2 was usually faster (20% faster than K-NN).

- Precision (Detection quality): Evaluating how each method separates foreground from background, both were
capable of executing the task, but the KNN method generates more pixel noise
to the final mask of identified foreground objects.

2. Testing in variate conditions
  
  Using videos in different conditions for testing robustness, the hardest conditions are: 

  - sudden changes in illumination condition;
  - the presence of the object in the scene since the method start (the object is assumed as part of background and will interfere with the background detection);
  - Objects being static for long periods.

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

Outros objetos ou sombras de objetos entrando e saindo do campo de visão (interferências dinâmicas).

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