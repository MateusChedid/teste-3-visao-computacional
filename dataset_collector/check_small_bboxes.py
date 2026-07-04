#!/usr/bin/env python3
"""
check_small_bboxes.py — Lista as imagens com menores bounding boxes.

Lê todos os .txt de labels em dataset/labels/train/ (e val/),
calcula a área de cada bbox, e exibe as N menores — para inspeção
manual antes do treino.

Uso:
    python dataset_collector/check_small_bboxes.py
    python dataset_collector/check_small_bboxes.py --n 50
    python dataset_collector/check_small_bboxes.py --split val
    python dataset_collector/check_small_bboxes.py --threshold 0.5
        (lista todas com área < 0.5% da imagem)
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=100,
                        help="Quantas imagens listar (padrão: 100)")
    parser.add_argument("--split",     type=str,   default="train",
                        choices=["train", "val", "all"],
                        help="Qual split analisar (padrão: train)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Listar todas com área < X%% (ex: --threshold 0.5)")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    entries = []  # (area_pct, img_path, lbl_path)

    for split in splits:
        lbl_dir = DATASET_ROOT / "labels" / split
        img_dir = DATASET_ROOT / "images" / split

        if not lbl_dir.exists():
            print(f"[AVISO] Pasta não encontrada: {lbl_dir}")
            continue

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            try:
                line = lbl_path.read_text().strip().split()
                if len(line) < 5:
                    continue
                bw, bh = float(line[3]), float(line[4])
                area_pct = bw * bh * 100.0
                img_path = img_dir / (lbl_path.stem + ".jpg")
                if not img_path.exists():
                    img_path = img_dir / (lbl_path.stem + ".png")
                entries.append((area_pct, img_path, lbl_path))
            except Exception:
                pass

    if not entries:
        print("[ERRO] Nenhum label encontrado. Verifique se o dataset foi gerado.")
        return

    # Ordenar por área (menor primeiro)
    entries.sort(key=lambda x: x[0])

    # Calcular média e mediana
    areas = [e[0] for e in entries]
    mean   = sum(areas) / len(areas)
    median = sorted(areas)[len(areas) // 2]

    print(f"\n{'='*64}")
    print(f"  BBOXES MAIS PEQUENAS — {args.split.upper()}")
    print(f"{'='*64}")
    print(f"  Total de imagens analisadas : {len(entries)}")
    print(f"  Área média das bboxes       : {mean:.3f}%")
    print(f"  Área mediana das bboxes     : {median:.3f}%")
    print(f"{'─'*64}\n")

    if args.threshold is not None:
        to_show = [(a, i, l) for a, i, l in entries if a < args.threshold]
        print(f"  Imagens com bbox < {args.threshold:.2f}% ({len(to_show)} encontradas):\n")
    else:
        to_show = entries[:args.n]
        print(f"  {args.n} imagens com menor bbox (média geral: {mean:.3f}%):\n")

    print(f"  {'#':<5} {'Área':>8}  {'vs média':>9}  Nome do arquivo")
    print(f"  {'─'*60}")

    for rank, (area, img_path, lbl_path) in enumerate(to_show, 1):
        diff = area - mean
        flag = "  ⚠" if area < mean * 0.3 else ("  ?" if area < mean * 0.5 else "")
        print(f"  {rank:<5} {area:>7.3f}%  {diff:>+8.3f}%  {img_path.name}{flag}")

    print(f"\n  ⚠ = área menor que 30% da média (provavelmente errada)")
    print(f"  ? = área menor que 50% da média (suspeita, confira o preview)")
    print(f"\n  Previews em:")
    print(f"    {PROJECT_ROOT / 'dataset_collector' / '_work' / 'bbox_preview'}")
    print(f"\n  Para remover uma imagem errada, delete o par:")
    print(f"    dataset/images/{args.split}/NOME.jpg")
    print(f"    dataset/labels/{args.split}/NOME.txt\n")


if __name__ == "__main__":
    main()
