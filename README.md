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

Only Background subtraction is barely at basic functional state.
Some background noise and a lot of shadow movements are interfering with the extracted contours. More work is needed.

More experiments will be made in code for Contour Detection and HSV Histogram, as them seem easier and 
faster to progress

Following initial work, testing Feature Matching will be the next step.

At last, Kalman filter and Optical flow seemed harder to understand. They are listed as alternatives to explore.

# Boris Annotation Tools
It was used an open-source application named BORIS, intended to be used as an annotation tool for biological and animal behaviour. it generates an
.boris file, resembling an .json, with organized metadata alongside the
annotations themselves.

https://www.boris.unito.it/user_guide/

That annotation file can be exported into other formats.

After some basic editing, removing unnecessary metadata, the exported
annotations are in the following google spreadsheet:

https://docs.google.com/spreadsheets/d/19KAGfeUxoV8xWah_oUuUOramBOONbrc8uq4Fhitezaw/edit?usp=sharing

People involved already have necessary access

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