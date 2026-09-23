# Desafio Técnico – Detecção de Pessoas em Vídeo com YOLO26

Aplicação em Python que usa o modelo **YOLO26** (Ultralytics) e **OpenCV**
para detectar pessoas em um vídeo, desenhar bounding boxes com o nível de
confiança, contar pessoas por frame e calcular métricas de qualidade
(Precision, Recall, F1-Score).

## Estrutura do projeto

```
desafio-yolo/
├── main.py              # detecção + geração do vídeo anotado + métricas gerais
├── calc_metrics.py      # calcula Precision/Recall/F1 a partir da avaliação manual
├── requirements.txt
├── README.md
├── input/
│   └── video.mp4        # coloque aqui o vídeo de entrada
├── output/
│   └── video_detectado.mp4   # gerado pelo main.py
└── results/
    ├── deteccoes.csv         # confiança de cada detecção, frame a frame
    ├── resumo.csv            # métricas gerais (total, média, máx, mín, FPS)
    ├── sample_frames/        # frames extraídos para avaliação manual
    ├── avaliacao_manual.csv  # template para contagem manual (preencher à mão)
    └── metricas.csv          # Precision/Recall/F1 por threshold (gerado por calc_metrics.py)
```

## 1. Instalação

Requer Python 3.9+.

```bash
# criar e ativar ambiente virtual (evita conflito de versão)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# instalar dependências
pip install -r requirements.txt
```

O pacote `ultralytics` baixa automaticamente o modelo YOLO26 (`yolo26n.pt`
por padrão) na primeira execução — não é necessário baixar nada manualmente.

## 2. Como executar

Coloque seu vídeo em `input/video.mp4` e rode:

```bash
python main.py
```

As configurações (vídeo de entrada,
threshold, etc.) ficam no topo do arquivo `main.py`, num bloco chamado
`CONFIGURAÇÕES`. Se quiser mudar o vídeo, o threshold de confiança ou
qualquer outra coisa, é só abrir o `main.py` e editar os valores ali, sem
precisar mexer no resto do código.

Isso gera o vídeo anotado em `output/`, o CSV de cada detecção em
`results/deteccoes.csv` e o resumo geral em `results/resumo.csv`.

### Avaliação manual de acurácia 

Por padrão o script já extrai 1 frame a cada 15 (até 20 no total) para
`results/sample_frames/` e cria `results/avaliacao_manual.csv` com uma
linha por frame amostrado, já preenchida com o número de pessoas
**detectadas** pelo modelo. Abra cada imagem, conte manualmente quantas
pessoas realmente aparecem e preencha as colunas restantes:

| coluna | o que preencher |
|---|---|
| `pessoas_reais` | quantas pessoas você vê de fato no frame |
| `corretas` | quantas detecções do modelo bateram com pessoas reais (TP) |
| `nao_detectadas` | pessoas reais que o modelo não pegou (FN) |
| `deteccoes_incorretas` | boxes que o modelo criou mas não são pessoas reais (FP) |

Depois, calcule Precision/Recall/F1:

```bash
python calc_metrics.py --file results/avaliacao_manual.csv --threshold 0.5
```

O resultado é impresso no terminal e anexado em `results/metricas.csv`.

### Comparando thresholds (seção 6 do desafio)

Repita o processo acima para cada threshold: abra o `main.py`, mude o
valor de `THRESHOLD_CONFIANCA` (0.3, depois 0.5, depois 0.7), rode
`python main.py` de novo, faça a avaliação manual, e depois:

```bash
python calc_metrics.py --file results/avaliacao_manual.csv --threshold 0.3
```
(ajuste o `--threshold` para o valor testado a cada rodada)

Cada rodada de `main.py` vai sobrescrever `results/avaliacao_manual.csv` —
copie o arquivo para um nome diferente (ex: `avaliacao_manual_030.csv`)
antes de rodar a próxima, se quiser manter o histórico. Ao final,
`results/metricas.csv` terá uma linha por threshold, pronta para comparar.

## 3. Modelo YOLO26 utilizado

`yolo26n.pt` (variante *nano*) — a mais leve da família YOLO26, treinada
no dataset COCO, com boa relação velocidade/precisão para rodar até em
CPU. Pode ser trocado por `yolo26s.pt` ou `yolo26m.pt` via `--model` se
for necessária mais precisão (ao custo de velocidade).

## 4. Threshold escolhido e resultados

> ⚠️ Preencher após rodar o pipeline com o seu vídeo.

- Threshold escolhido: **0.30**
- Total de detecções: 1364
- Confiança média / máxima / mínima: (80.5 | 91.3 | 30.0)
- Precision / Recall / F1-Score: (96.9 | 94.0 | 95.4)
- Tabela comparativa dos 3 thresholds (0.3 / 0.5 / 0.7):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 96.9 | 94.0 | 95.4 |
| 0.50 | 98.3 | 89.7 | 93.8 |
| 0.70 | 100  | 79.4 | 88.5 |

## 5. Principais dificuldades encontradas

> oclusão entre pessoas, pessoas parcialmente fora do quadro, pessoas muito distantes.

---

## 6. Perguntas do desafio

**O que é uma Bounding Box?**
É o retângulo (definido pelas coordenadas dos cantos, `x1,y1,x2,y2`) que
delimita a região da imagem onde o modelo identificou um objeto — neste
caso, uma pessoa.

**O que significa a confiança apresentada pelo YOLO?**
É a probabilidade estimada pelo modelo de que a detecção está correta,
tanto em relação à classe (é realmente uma "pessoa") quanto à qualidade
da caixa desenhada. Não é uma medida absoluta de certeza, mas um score
relativo baseado no que a rede aprendeu durante o treinamento.

**Qual a diferença entre Precision e Recall?**
Precision mede, entre tudo que o modelo detectou, quanto realmente era
pessoa (evita falsos positivos). Recall mede, entre todas as pessoas que
realmente existiam no vídeo, quantas o modelo conseguiu encontrar (evita
falsos negativos). Um modelo pode ter Precision alta e Recall baixo (é
conservador, mas perde detecções) ou o contrário (detecta demais, com
mais erros).

**O que é um falso positivo?**
Quando o modelo desenha uma bounding box de "pessoa" em algo que não é
uma pessoa (ou em uma região vazia).

**O que é um falso negativo?**
Quando existe uma pessoa real no frame e o modelo não a detecta.

**O que aconteceu quando o threshold foi aumentado?**
> aumentar o threshold reduz falsos positivos (mais Precision) mas também faz o
> modelo deixar passar detecções de menor confiança que na verdade eram
> pessoas reais (menos Recall).

**Qual threshold você escolheria para esse vídeo e por quê?**
> Threshold 0.30, pois conseguiu manter o equilíbrio entre Precision e Recall.

**Em quais situações o modelo apresentou maior dificuldade?**
>  pessoas sobrepostas, longe da câmera, parcialmente cobertas por objetos e fora do quadro.
