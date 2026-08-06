# poor-mans-kinect

Webcam full-body motion controller — a low-cost, camera-only alternative to a Kinect/depth sensor for casual full-body input, demoed with a built-in dodge mini-game.

## O que é e por quê

O Microsoft Kinect (e sensores de profundidade similares) permitia controlar jogos com o corpo inteiro, mas depende de hardware dedicado (câmera de profundidade, infravermelho) que a maioria das pessoas não tem mais à mão — os sensores originais foram descontinuados e equivalentes atuais (LiDAR, câmeras de profundidade) ainda são caros ou incomuns em notebooks comuns.

Este projeto é a versão "pão-e-queijo" desse conceito: usa **apenas uma webcam RGB comum** e estimação de pose 2D (sem profundidade real) para detectar a postura da pessoa e traduzir isso em comandos discretos — mover para esquerda/direita e pular — que controlam um mini-jogo de desvio de obstáculos. Não há sensor de profundidade nem hardware especial: toda a "mágica" é feita em software, por cima de uma câmera qualquer.

## Como funciona (visão técnica)

O script (`poor_mans_kinect.py`) roda um loop único que, a cada frame da webcam:

1. **Estimação de pose** — usa o `mediapipe` (Tasks API, `PoseLandmarker`, modelo `pose_landmarker_lite.task` embutido no repositório) para extrair 33 landmarks corporais (nariz, ombros, cotovelos, pulsos, quadris, joelhos, tornozelos) por frame, em modo `VIDEO`.
2. **Fallback por subtração de fundo** — quando nenhuma pessoa é detectada pelo MediaPipe, o código cai para um detector de movimento baseado em `cv2.createBackgroundSubtractorMOG2` sobre a imagem em escala de cinza + blur gaussiano, usando a densidade de pixels em movimento por zona da tela como substituto grosseiro da pose.
3. **Classificação de comando** (`CommandClassifier`) — a partir da pose (ou da máscara de movimento no fallback), decide entre `NEUTRAL`, `LEFT`, `RIGHT` ou `JUMP`:
   - `JUMP`: os dois pulsos acima da linha dos ombros (ou, no fallback, muitos pixels de movimento na zona superior da tela).
   - `LEFT` / `RIGHT`: posição X do nariz (ou do centróide do movimento, no fallback) além de um limiar horizontal.
   - Um filtro de persistência de N frames evita comandos "tremidos" por ruído de detecção.
   - Também calcula uma **intensidade analógica** (0.0–1.0) de acordo com o quanto o corpo está inclinado além do limiar, usada para variar a velocidade do jogador.
4. **Overlay / HUD** (`OverlayRenderer`) — desenha zonas de comando, esqueleto (bones + joints coloridos por comando ativo), indicador de modo (POSE vs MOTION), FPS, latência média e barra de velocidade.
5. **Log de latência** (`LatencyLogger`) — grava em CSV (`latency_log.csv`) o timestamp de captura e de emissão de cada frame, com a latência em ms, além de manter uma média móvel exibida no HUD.
6. **Mini-jogo de desvio** (`DodgeGame`) — um jogo simples renderizado em uma janela OpenCV separada: o jogador se move lateralmente conforme os comandos `LEFT`/`RIGHT` (com velocidade proporcional à intensidade) e pula (arco parabólico via seno) com `JUMP`, desviando de obstáculos que caem e aumentam de velocidade com o tempo.

Duas janelas OpenCV são abertas lado a lado: a câmera com o esqueleto e o HUD sobrepostos, e o jogo.

## Requisitos

- Python 3.10+ (testado com Python 3.14 no ambiente do autor)
- Webcam
- Sistema operacional com suporte a janelas do OpenCV (o script tem ajustes específicos de variáveis de ambiente Qt para rodar via XWayland no GNOME/Wayland — ver comentários no topo do arquivo)

### Dependências Python

- [`opencv-python`](https://pypi.org/project/opencv-python/) — captura de vídeo, processamento de imagem e renderização das janelas
- [`numpy`](https://pypi.org/project/numpy/) — operações numéricas
- [`mediapipe`](https://pypi.org/project/mediapipe/) — estimação de pose (Tasks API / `PoseLandmarker`)

> Não há arquivo `requirements.txt` no repositório no momento — as dependências acima foram identificadas lendo os `import`s do script.

## Instalação

```bash
git clone <url-do-repositorio>
cd poor-mans-kinect

python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install opencv-python numpy mediapipe
```

O modelo de pose (`pose_landmarker_lite.task`) já está incluído no repositório, na raiz — não é necessário baixá-lo separadamente.

## Uso

```bash
python poor_mans_kinect.py
```

Ao rodar:

1. O script carrega o modelo do MediaPipe e abre a webcam.
2. Duas janelas aparecem: **"Poor Man's Kinect — Camera"** (feed da webcam com esqueleto/HUD) e **"Poor Man's Kinect — Dodge Game"** (o mini-jogo).
3. Há uma fase de **calibração de ~3 segundos** — fique parado em frente à câmera até ela terminar.
4. Comandos:
   - Incline o corpo (ou desloque o nariz) para a **esquerda/direita** da tela → move o jogador nessa direção; quanto mais longe do centro, mais rápido (velocidade analógica de 4 a 32 px/frame).
   - **Levante os braços** acima dos ombros → o jogador pula.
5. Teclas no jogo:
   - `Q` ou `ESC` — sai do programa.
   - `R` — reinicia o jogo após "game over".
6. Ao sair, um resumo da sessão é impresso no terminal (frames processados, total de comandos emitidos, latência média, pontuação final) e o log de latência é salvo em `latency_log.csv`.

## Estrutura do projeto

```
poor-mans-kinect/
├── poor_mans_kinect.py       # Script único: pose, fallback de movimento, classificador,
│                              # overlay/HUD, mini-jogo e loop principal
├── pose_landmarker_lite.task # Modelo MediaPipe Pose Landmarker (Lite) usado para estimação de pose
├── latency_log.csv           # Log de latência por frame, gerado/sobrescrito a cada execução
└── README.md
```

Todo o código está concentrado em um único arquivo (`poor_mans_kinect.py`, ~800 linhas), organizado em classes: `LatencyLogger`, `PoseTracker`, `MotionDetector`, `CommandClassifier`, `OverlayRenderer`, `DodgeGame`, e a função `main()` que orquestra o loop.

## Limitações conhecidas

- **Sem profundidade real**: diferente de um Kinect de verdade, não há sensor de profundidade — a "detecção 3D" é inferida apenas de coordenadas 2D normalizadas de uma pose estimada por IA, o que a torna sensível a iluminação, oclusão e ao ângulo da câmera.
- **Uma única pessoa por vez** (`num_poses=1` no `PoseLandmarkerOptions`).
- **Câmera e resolução fixas no código** (`CAMERA_INDEX = 0`, 640×480, 30 FPS) — não há flags de linha de comando para configurar isso; é preciso editar as constantes no topo do arquivo.
- **Comandos limitados**: apenas `LEFT`, `RIGHT` e `JUMP` são de fato usados no jogo (o rótulo `SPECIAL` existe na definição de cores/labels, mas não há lógica que o dispare).
- **Sem testes automatizados** e sem `requirements.txt`/empacotamento — é um script standalone.
- **Dependência de ambiente gráfico**: usa janelas nativas do OpenCV/Qt; as variáveis de ambiente fixadas no início do script (`QT_QPA_PLATFORM=xcb`, etc.) foram ajustadas para funcionar em GNOME/Wayland via XWayland, e podem precisar de adaptação em outros ambientes.
- **Fallback de movimento é rudimentar**: quando a pose não é detectada, a heurística de subtração de fundo (MOG2) é bem mais grosseira que o rastreamento por landmarks, servindo apenas como modo de contingência.

## Licença

Não há arquivo de licença no repositório. Sem uma licença explícita, os termos padrão de direitos autorais se aplicam (todos os direitos reservados ao autor) — considere adicionar um arquivo `LICENSE` (por exemplo MIT ou Apache 2.0) se pretende permitir reuso por terceiros.
