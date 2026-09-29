# Desafio Técnico – Detecção de Pessoas em Vídeo com YOLO26

Aplicação em Python que usa **YOLO26** (Ultralytics) e **OpenCV** para
detectar pessoas em um vídeo, desenhar bounding boxes com o nível de
confiança, contar pessoas por frame e calcular Precision, Recall e
F1-Score.

## Estrutura

```
desafio-yolo/
├── main.py              # roda a detecção e gera o vídeo + CSVs
├── calc_metrics.py      # calcula Precision/Recall/F1
├── requirements.txt
├── README.md
├── input/video.mp4
├── output/video_detectado.mp4
└── results/
    ├── deteccoes.csv
    ├── resumo.csv
    ├── sample_frames/
    ├── avaliacao_manual.csv
    └── metricas.csv
```

## Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

O modelo YOLO26 (`yolo26n.pt`) é baixado automaticamente na primeira
execução.

## Como executar

1. Coloque o vídeo em `input/video.mp4`.
2. Rode:
   ```bash
   python main.py
   ```
   Gera o vídeo anotado em `output/`, as detecções em
   `results/deteccoes.csv`, o resumo em `results/resumo.csv`, e ~20 frames
   de amostra em `results/sample_frames/` com o template
   `results/avaliacao_manual.csv` pronto pra preencher.
3. Abra os frames de amostra e preencha à mão as colunas `corretas`,
   `nao_detectadas` e `deteccoes_incorretas` (comparando com o que
   realmente aparece em cada frame).
4. Rode:
   ```bash
   python calc_metrics.py
   ```
   Calcula Precision/Recall/F1 e salva em `results/metricas.csv`.

Para testar outro threshold: abra `main.py`, mude o valor de
`THRESHOLD_CONFIANCA` no topo do arquivo, rode os passos 2-4 de novo. Vale
o mesmo para `calc_metrics.py` — atualize `THRESHOLD_TESTADO` antes de
rodar, pra ficar registrado certo na tabela final.

## Modelo utilizado

`yolo26n.pt` (variante *nano*) — a mais leve da família, treinada no
COCO, com boa relação velocidade/precisão mesmo rodando em CPU.

## Resultados

- **Threshold escolhido:** 0.30
- **Total de detecções:** 1364
- **Confiança média / máxima / mínima:** 80.5% / 91.3% / 30.0%
- **Precision / Recall / F1-Score:** 96.9% / 94.0% / 95.4%

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 96.9% | 94.0% | 95.4% |
| 0.50 | 98.3% | 89.7% | 93.8% |
| 0.70 | 100%  | 79.4% | 88.5% |

## Principais dificuldades

Oclusão entre pessoas, pessoas parcialmente fora do quadro e pessoas
muito distantes da câmera.

---

## Perguntas do desafio

**O que significa o confidence score do YOLO?**
O confidence score do YOLO é a confiança que ele tem que a deteccção está correta.

**Qual seria o impacto de utilizar um threshold de 0,30 em comparação com 0,70?**
O verdadeiro impacto seria na métrica de Recall e no equilíbrio de F1-Score,
pois com um threshold de 0,30 o modelo consegue detectar mais pessoas no frame,
o que afeta a precisão de confiança do modelo e pode gerar mais Falsos Positivos(FP). 

**O que é uma Bounding Box?**
O retângulo (coordenadas `x1,y1,x2,y2`) que delimita onde o modelo
identificou a pessoa na imagem.

**O que significa a confiança apresentada pelo YOLO?**
A probabilidade estimada pelo modelo de que a detecção está correta —
não é certeza absoluta, é um score relativo aprendido no treinamento.

**Qual a diferença entre Precision e Recall?**
Precision mede quanto do que o modelo detectou realmente era pessoa.
Recall mede quantas das pessoas reais o modelo conseguiu encontrar.

**O que é um falso positivo?**
O modelo desenha uma bounding box em algo que não é pessoa.

**O que é um falso negativo?**
Existe uma pessoa real no frame e o modelo não detecta.

**O que aconteceu quando o threshold foi aumentado?**
A Precision subiu (menos falsos positivos), mas o Recall caiu — o
modelo passou a deixar de detectar pessoas reais com confiança um
pouco mais baixa.

**Qual threshold você escolheria para esse vídeo e por quê?**
0.30, porque manteve o melhor equilíbrio entre Precision e Recall (maior
F1 entre os três testados).

**Em quais situações o modelo apresentou maior dificuldade?**
Pessoas sobrepostas, longe da câmera, parcialmente cobertas por objetos
ou parcialmente fora do quadro.
