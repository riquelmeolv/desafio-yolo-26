"""
O QUE ESSE SCRIPT FAZ, EM 4 PASSOS:
  1. Abre o vídeo de entrada
  2. Passa cada frame (imagem) pelo modelo YOLO, que devolve as pessoas
     encontradas (posição da caixa + confiança)
  3. Desenha as caixas na imagem e salva um vídeo novo com elas
  4. Ao final, salva um resumo em CSV (total de detecções, confiança
     média, etc.) e algumas imagens de amostra para avaliação manual
"""
# bibliotecas de csv, tempo e caminhos do sistema.
import csv
import time
from pathlib import Path

# OpenCV e YOLO
import cv2
from ultralytics import YOLO

# CONFIGURAÇÕES - altere aqui se quiser mudar vídeo, threshold, etc.
VIDEO_ENTRADA = "input/video.mp4" 
VIDEO_SAIDA = "output/video_detectado.mp4"
PASTA_RESULTADOS = "results"
MODELO = "yolo26n.pt"          # baixado automaticamente na 1ª execução da 'main'
THRESHOLD_CONFIANCA = 0.7      # só aceita detecções com confiança >= 30/50/70%
EXTRAIR_AMOSTRA_A_CADA = 15    # salva 1 frame a cada 15 para avaliação manual
MAX_FRAMES_AMOSTRA = 20        # no máximo 20 frames de amostra

PERSON_CLASS_ID = 0  # no dataset COCO, "pessoa" é sempre a classe número 0


# PASSO 1: carregar o modelo YOLO e abrir o vídeo
def carregar_modelo():
    print(f"Carregando modelo {MODELO}...")
    return YOLO(MODELO)


def abrir_video(caminho):
    video = cv2.VideoCapture(caminho)
    if not video.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho}")
    return video


# PASSO 2: detectar pessoas em UM frame e desenhar as caixas
def detectar_e_desenhar(frame, modelo):
    """
    Recebe um frame (uma imagem) e o modelo YOLO.
    Devolve: o frame com as caixas desenhadas, quantas pessoas foram
    encontradas, e a lista de detecções (para salvar no CSV depois).
    """
    resultado = modelo.predict(
        frame,
        conf=THRESHOLD_CONFIANCA,
        classes=[PERSON_CLASS_ID],  # só detecta pessoas, ignora o resto
        verbose=False,
    )[0]

    deteccoes = []  # vai guardar (id, confiança, x1, y1, x2, y2) de cada pessoa
    quantidade = 0

    for caixa in resultado.boxes: # Revisar
        quantidade += 1
        confianca = float(caixa.conf[0])
        x1, y1, x2, y2 = map(int, caixa.xyxy[0])
        deteccoes.append((quantidade, confianca, x1, y1, x2, y2))

        # desenha o retângulo verde ao redor da pessoa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # escreve o texto "Pessoa X: NN%" em cima da caixa
        texto = f"Pessoa {quantidade}: {confianca * 100:.0f}%"
        cv2.putText(frame, texto, (x1 + 2, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # escreve o total de pessoas no canto do frame
    cv2.putText(frame, f"Pessoas: {quantidade}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame, quantidade, deteccoes


# PASSO 3: processar o vídeo inteiro, frame por frame
def processar_video():
    # referencia o modelo e o vídeo de entrada
    modelo = carregar_modelo()
    video = abrir_video(VIDEO_ENTRADA)

    # cria as pastas de saída (output, result, sample_frames), se ainda não existirem
    Path(VIDEO_SAIDA).parent.mkdir(parents=True, exist_ok=True)
    Path(PASTA_RESULTADOS).mkdir(parents=True, exist_ok=True)
    pasta_amostras = Path(PASTA_RESULTADOS) / "sample_frames" # Revisar
    pasta_amostras.mkdir(parents=True, exist_ok=True) # Revisar

    # pega as informações do vídeo (tamanho, fps) para criar o vídeo de saída
    fps = video.get(cv2.CAP_PROP_FPS) or 30 # Revisar
    largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gravador = cv2.VideoWriter(
        VIDEO_SAIDA, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )

    # arquivo onde cada detecção individual vai ser registrada
    arquivo_csv = open(Path(PASTA_RESULTADOS) / "deteccoes.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(arquivo_csv)
    csv_writer.writerow(["frame", "pessoa_id", "confianca", "x1", "y1", "x2", "y2"])

    todas_confiancas = []   # guarda a confiança de TODAS as detecções do vídeo
    linhas_amostra = []     # frames escolhidos para avaliação manual depois

    numero_frame = 0
    inicio = time.time()

    while True:
        sucesso, frame = video.read()
        if not sucesso:
            break  # acabou o vídeo
        
        # o modelo devolve o frame já com as caixas desenhadas, a quantidade de pessoas e a lista de detecções
        frame, qtd_pessoas, deteccoes = detectar_e_desenhar(frame, modelo)

        # salva cada detecção desse frame no CSV
        for pessoa_id, confianca, x1, y1, x2, y2 in deteccoes:
            csv_writer.writerow([numero_frame, pessoa_id, round(confianca, 4), x1, y1, x2, y2])
            todas_confiancas.append(confianca)

        gravador.write(frame)  # salva o frame (já com as caixas) no vídeo de saída

        # a cada N frames, guarda uma cópia para você avaliar manualmente depois
        if numero_frame % EXTRAIR_AMOSTRA_A_CADA == 0 and len(linhas_amostra) < MAX_FRAMES_AMOSTRA:
            cv2.imwrite(str(pasta_amostras / f"frame_{numero_frame}.jpg"), frame)
            linhas_amostra.append({
                "frame": numero_frame,
                "detectadas": qtd_pessoas,
                "pessoas_reais": "",
                "corretas": "",
                "nao_detectadas": "",
                "deteccoes_incorretas": "",
            })

        numero_frame += 1
        if numero_frame % 30 == 0:
            print(f"Processando frame {numero_frame}...")

    video.release()
    gravador.release()
    arquivo_csv.close()

    salvar_resumo(numero_frame, todas_confiancas, time.time() - inicio)
    salvar_template_avaliacao(linhas_amostra)


# PASSO 4: salvar as estatísticas finais
def salvar_resumo(total_frames, confiancas, tempo_gasto):
    total_deteccoes = len(confiancas)
    media = sum(confiancas) / total_deteccoes if total_deteccoes else 0
    maxima = max(confiancas) if confiancas else 0
    minima = min(confiancas) if confiancas else 0

    with open(Path(PASTA_RESULTADOS) / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerow(["modelo", MODELO])
        w.writerow(["threshold_confianca", THRESHOLD_CONFIANCA])
        w.writerow(["total_frames", total_frames])
        w.writerow(["total_deteccoes", total_deteccoes])
        w.writerow(["confianca_media", round(media, 4)])
        w.writerow(["confianca_maxima", round(maxima, 4)])
        w.writerow(["confianca_minima", round(minima, 4)])

    print("\n=== Resumo ===")
    print(f"Frames processados: {total_frames}")
    print(f"Total de detecções: {total_deteccoes}")
    print(f"Confiança média: {media * 100:.1f}%")
    print(f"Confiança máxima: {maxima * 100:.1f}%")
    print(f"Confiança mínima: {minima * 100:.1f}%")
    print(f"Vídeo salvo em: {VIDEO_SAIDA}")


def salvar_template_avaliacao(linhas_amostra):
    if not linhas_amostra:
        return
    caminho = Path(PASTA_RESULTADOS) / "avaliacao_manual.csv"
    campos = ["frame", "pessoas_reais", "detectadas", "corretas", "nao_detectadas", "deteccoes_incorretas"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for linha in linhas_amostra:
            w.writerow({campo: linha[campo] for campo in campos})
    print(f"\nTemplate de avaliação manual criado em: {caminho}")
    print("Abra os frames em results/sample_frames/, preencha as colunas restantes")
    print("e rode: python calc_metrics.py")


if __name__ == "__main__":
    processar_video()
