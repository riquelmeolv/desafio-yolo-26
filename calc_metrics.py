"""
calc_metrics.py - Calcula Precision, Recall e F1-Score a partir da
avaliação manual de frames (results/avaliacao_manual.csv).

Antes de rodar, preencha as colunas 'pessoas_reais', 'corretas',
'nao_detectadas' e 'deteccoes_incorretas' no CSV, assistindo aos frames
salvos em results/sample_frames/. Veja o README.md para o passo a passo.

Uso:
    python calc_metrics.py --file results/avaliacao_manual.csv --threshold 0.5

Cada execução ANEXA uma linha em results/metricas.csv. Rode este script
uma vez para cada threshold testado (0.3, 0.5, 0.7...) para montar a
tabela comparativa pedida na seção 6 do desafio.
"""
import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Calcula Precision/Recall/F1")
    parser.add_argument("--file", default="results/avaliacao_manual.csv",
                         help="CSV preenchido manualmente com a contagem real de pessoas")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Threshold de confiança usado nesta rodada (apenas para registro)")
    parser.add_argument("--out", default="results/metricas.csv",
                         help="CSV onde o resultado desta rodada será anexado")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode main.py com --sample-every primeiro.")

    tp = fp = fn = 0
    rows_read = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["corretas"] == "" or row["nao_detectadas"] == "" or row["deteccoes_incorretas"] == "":
                continue  # linha ainda não preenchida manualmente
            rows_read += 1
            tp += int(row["corretas"])
            fn += int(row["nao_detectadas"])
            fp += int(row["deteccoes_incorretas"])

    if rows_read == 0:
        print(f"Nenhuma linha preenchida ainda em {path}.")
        print("Preencha 'pessoas_reais', 'corretas', 'nao_detectadas' e 'deteccoes_incorretas' e rode de novo.")
        return

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"Frames avaliados: {rows_read}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision: {precision * 100:.1f}%")
    print(f"Recall: {recall * 100:.1f}%")
    print(f"F1-Score: {f1 * 100:.1f}%")

    out_path = Path(args.out)
    is_new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        w.writerow([args.threshold, tp, fp, fn, round(precision, 4), round(recall, 4), round(f1, 4)])
    print(f"\nResultado anexado em: {out_path}")


if __name__ == "__main__":
    main()
