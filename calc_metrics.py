"""
calc_metrics.py - Calcula Precision, Recall e F1-Score

Lê o CSV preenchido manualmente (results/avaliacao_manual.csv), soma os
acertos (TP), falsos positivos (FP) e falsos negativos (FN) de todos os
frames avaliados, e calcula as três métricas.

Antes de rodar, preencha as colunas 'corretas', 'nao_detectadas' e
'deteccoes_incorretas' no CSV, olhando os frames em results/sample_frames/.

Para rodar:
    python calc_metrics.py
"""

import csv
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ARQUIVO_AVALIACAO = "results/avaliacao_manual.csv"   # CSV preenchido à mão
ARQUIVO_SAIDA = "results/metricas.csv"                # onde o resultado é salvo
THRESHOLD_TESTADO = 0.7  # só para registro na tabela final — use o mesmo valor do main.py


# ============================================================
# PASSO 1: somar TP, FP e FN a partir do CSV preenchido à mão
# ============================================================
def contar_acertos_e_erros():
    caminho = Path(ARQUIVO_AVALIACAO)
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não encontrado. Rode main.py primeiro.")

    tp = fp = fn = 0
    frames_avaliados = 0

    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = csv.DictReader(f)  # lê cada linha do CSV como um dicionário
        for linha in linhas:
            # pula frames que você ainda não preencheu manualmente
            if linha["corretas"] == "" or linha["nao_detectadas"] == "" or linha["deteccoes_incorretas"] == "":
                continue

            frames_avaliados += 1
            tp += int(linha["corretas"])
            fn += int(linha["nao_detectadas"])
            fp += int(linha["deteccoes_incorretas"])

    return tp, fp, fn, frames_avaliados


# ============================================================
# PASSO 2: calcular Precision, Recall e F1 a partir de TP, FP, FN
# ============================================================
def calcular_metricas(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1


# ============================================================
# PASSO 3: mostrar o resultado e salvar no CSV final
# ============================================================
def salvar_resultado(precision, recall, f1, tp, fp, fn):
    caminho = Path(ARQUIVO_SAIDA)
    arquivo_novo = not caminho.exists()

    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if arquivo_novo:
            w.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        w.writerow([THRESHOLD_TESTADO, tp, fp, fn, round(precision, 4), round(recall, 4), round(f1, 4)])

    print(f"Resultado salvo em: {caminho}")


def main():
    tp, fp, fn, frames_avaliados = contar_acertos_e_erros()

    if frames_avaliados == 0:
        print(f"Nenhum frame preenchido ainda em {ARQUIVO_AVALIACAO}.")
        print("Abra os frames em results/sample_frames/ e preencha as colunas antes de rodar de novo.")
        return

    precision, recall, f1 = calcular_metricas(tp, fp, fn)

    print(f"Frames avaliados: {frames_avaliados}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision: {precision * 100:.1f}%")
    print(f"Recall: {recall * 100:.1f}%")
    print(f"F1-Score: {f1 * 100:.1f}%")

    salvar_resultado(precision, recall, f1, tp, fp, fn)


if __name__ == "__main__":
    main()
