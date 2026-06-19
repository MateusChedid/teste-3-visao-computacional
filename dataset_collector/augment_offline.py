#!/usr/bin/env python3
"""
augment_offline.py — Augmentation offline de luminosidade, contraste e ruído.

Gera variações permanentes das imagens de treino no próprio dataset,
cada uma com o mesmo arquivo .txt de label (a bbox não muda).

Variações geradas por imagem:
  _bright      → brilho aumentado (+60 em todos os canais)
  _dark        → brilho reduzido (-60 em todos os canais)
  _hicontrast  → contraste elevado (alpha=1.6, beta=-40)
  _noise       → ruído gaussiano (granularidade, sigma=18)

Uso:
    python dataset_collector/augment_offline.py
    python dataset_collector/augment_offline.py --dry-run   # mostra o que faria, sem criar arquivos
    python dataset_collector/augment_offline.py --undo      # remove todas as imagens augmentadas (_bright/_dark/_hicontrast/_noise)
"""

import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_IMG    = PROJECT_ROOT / "dataset" / "images" / "train"
TRAIN_LBL    = PROJECT_ROOT / "dataset" / "labels" / "train"

# Sufixos gerados por este script — usados também para --undo
SUFFIXES = ["_bright", "_dark", "_hicontrast", "_noise"]


# ─── Transformações ────────────────────────────────────────────────────────────

def apply_bright(img: np.ndarray) -> np.ndarray:
    """Aumenta o brilho somando 60 em todos os canais (clampado em 255)."""
    return np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8)


def apply_dark(img: np.ndarray) -> np.ndarray:
    """Reduz o brilho subtraindo 60 em todos os canais (clampado em 0)."""
    return np.clip(img.astype(np.int16) - 60, 0, 255).astype(np.uint8)


def apply_hicontrast(img: np.ndarray) -> np.ndarray:
    """
    Eleva o contraste: output = alpha * input + beta
    alpha > 1 → mais contraste; beta negativo → evita saturar as altas luzes.
    """
    alpha, beta = 1.6, -40
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def apply_noise(img: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    """Adiciona ruído gaussiano (simula granularidade/sensor fraco)."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


TRANSFORMS = {
    "_bright":     apply_bright,
    "_dark":       apply_dark,
    "_hicontrast": apply_hicontrast,
    "_noise":      apply_noise,
}


# ─── Main ──────────────────────────────────────────────────────────────────────

def collect_originals(img_dir: Path) -> list[Path]:
    """
    Retorna apenas as imagens ORIGINAIS (sem sufixo de augmentation),
    para não augmentar imagens que já são augmentadas.
    """
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    originals = []
    for p in all_imgs:
        stem = p.stem
        if any(stem.endswith(sfx) for sfx in SUFFIXES):
            continue
        originals.append(p)
    return originals


def undo(img_dir: Path, lbl_dir: Path, dry_run: bool = False):
    """Remove todas as imagens e labels augmentadas geradas por este script."""
    removed = 0
    for p in sorted(img_dir.glob("*")):
        if any(p.stem.endswith(sfx) for sfx in SUFFIXES):
            if not dry_run:
                p.unlink()
            removed += 1
            print(f"  [rm] {p.name}")

    for p in sorted(lbl_dir.glob("*.txt")):
        if any(p.stem.endswith(sfx) for sfx in SUFFIXES):
            if not dry_run:
                p.unlink()
            removed += 1
            print(f"  [rm] {p.name}")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{tag}Removidos: {removed} arquivos.")


def augment(img_dir: Path, lbl_dir: Path, dry_run: bool = False):
    originals = collect_originals(img_dir)

    if not originals:
        print(f"[AVISO] Nenhuma imagem original encontrada em {img_dir}")
        print("        Rode primeiro: python dataset_collector/auto_collect.py --from-folder")
        return

    print(f"  {len(originals)} imagens originais encontradas")
    print(f"  {len(TRANSFORMS)} variações por imagem → "
          f"+{len(originals) * len(TRANSFORMS)} imagens no total\n")

    created = 0
    skipped = 0

    for img_path in originals:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [ERRO] não foi possível ler {img_path.name}")
            continue

        # Label original (.txt) — mesma bbox, só copiamos
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  [AVISO] label não encontrado para {img_path.name}, pulando")
            continue

        for suffix, transform in TRANSFORMS.items():
            out_img  = img_dir / (img_path.stem + suffix + img_path.suffix)
            out_lbl  = lbl_dir / (img_path.stem + suffix + ".txt")

            if out_img.exists():
                skipped += 1
                continue

            if not dry_run:
                augmented = transform(img)
                cv2.imwrite(str(out_img), augmented, [cv2.IMWRITE_JPEG_QUALITY, 92])
                shutil.copy2(str(lbl_path), str(out_lbl))

            created += 1
            if dry_run:
                print(f"  [DRY] {out_img.name}")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{tag}Criados: {created} | Já existiam (pulados): {skipped}")
    if not dry_run and created > 0:
        print(f"\n[✓] Augmentation concluída.")
        print(f"    Próximo passo: python training/train.py")


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation offline de brilho/contraste/ruído no dataset de treino."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que seria feito sem criar/remover arquivos")
    parser.add_argument("--undo", action="store_true",
                        help="Remove todos os arquivos augmentados gerados anteriormente")
    args = parser.parse_args()

    print("=" * 58)
    print("  AUGMENTATION OFFLINE — luminosidade / contraste / ruído")
    print("=" * 58)
    print(f"  Diretório: {TRAIN_IMG}\n")

    if not TRAIN_IMG.exists():
        print(f"[ERRO] Pasta não encontrada: {TRAIN_IMG}")
        print("       Verifique se o dataset já foi gerado.")
        return

    if args.undo:
        print("  Modo: DESFAZER (remover imagens augmentadas)\n")
        undo(TRAIN_IMG, TRAIN_LBL, dry_run=args.dry_run)
    else:
        print("  Modo: GERAR variações\n")
        augment(TRAIN_IMG, TRAIN_LBL, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
