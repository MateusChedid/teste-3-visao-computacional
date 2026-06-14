#!/usr/bin/env python3
"""
validate_dataset.py — Valida o dataset e exibe estatísticas detalhadas.

Uso:
    python dataset_collector/validate_dataset.py
    python dataset_collector/validate_dataset.py --fix-split
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_YAML = DATASET_ROOT / "dataset.yaml"

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}
MIN_EXAMPLES_PER_CLASS = 30


def load_classes():
    with open(DATASET_YAML) as f:
        return yaml.safe_load(f)["names"]


def scan_split(split, class_names):
    images_dir = DATASET_ROOT / "images" / split
    labels_dir = DATASET_ROOT / "labels" / split

    images = {p.stem: p for p in images_dir.glob("*.jpg")}
    images.update({p.stem: p for p in images_dir.glob("*.png")})
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}

    orphan_images = set(images) - set(labels)
    orphan_labels = set(labels) - set(images)
    paired        = set(images) & set(labels)

    class_counts  = defaultdict(int)
    invalid_boxes = []

    for stem in paired:
        with open(labels[stem]) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_boxes.append((labels[stem].name, lineno, "formato incorreto"))
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                except ValueError:
                    invalid_boxes.append((labels[stem].name, lineno, "valores não numéricos"))
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    invalid_boxes.append((labels[stem].name, lineno,
                                          f"coordenadas fora do range: {parts[1:]}"))
                    continue
                if cid not in class_names:
                    invalid_boxes.append((labels[stem].name, lineno,
                                          f"class_id {cid} não existe"))
                    continue
                class_counts[cid] += 1

    return {
        "split":        split,
        "n_images":     len(images),
        "n_labels":     len(labels),
        "n_paired":     len(paired),
        "orphan_images":orphan_images,
        "orphan_labels":orphan_labels,
        "class_counts": dict(class_counts),
        "invalid_boxes":invalid_boxes,
    }


def print_report(results, class_names):
    total_images = sum(r["n_images"] for r in results)
    total_annots = sum(sum(r["class_counts"].values()) for r in results)

    print("\n" + "="*65)
    print("  RELATÓRIO DO DATASET — RPG DICE CV v2")
    print("="*65)

    print(f"\n{'Split':<8} {'Imagens':>8} {'Labels':>8} {'Anotações':>10} {'Cobertura':>10}")
    print("-"*50)
    for r in results:
        n_annots = sum(r["class_counts"].values())
        coverage = 100 * r["n_paired"] / r["n_images"] if r["n_images"] else 0
        print(f"{r['split']:<8} {r['n_images']:>8} {r['n_labels']:>8} {n_annots:>10} {coverage:>9.1f}%")
    print(f"{'TOTAL':<8} {total_images:>8} {'':>8} {total_annots:>10}")

    any_problem = False
    for r in results:
        if r["orphan_images"]:
            any_problem = True
            print(f"\n[!] {r['split']}: {len(r['orphan_images'])} imagem(ns) sem label")
        if r["orphan_labels"]:
            any_problem = True
            print(f"\n[!] {r['split']}: {len(r['orphan_labels'])} label(s) sem imagem")
        if r["invalid_boxes"]:
            any_problem = True
            print(f"\n[!] {r['split']}: {len(r['invalid_boxes'])} box(es) inválida(s):")
            for (fname, lineno, msg) in r["invalid_boxes"][:5]:
                print(f"      {fname} linha {lineno}: {msg}")
            if len(r["invalid_boxes"]) > 5:
                print(f"      … e mais {len(r['invalid_boxes'])-5}")

    if not any_problem:
        print("\n[✓] Sem problemas estruturais detectados.")

    # Distribuição por classe (train)
    train_r = next((r for r in results if r["split"] == "train"), None)
    if train_r and train_r["class_counts"]:
        print("\n" + "-"*65)
        print("  Distribuição de classes (train)")
        print("-"*65)

        grouped = defaultdict(list)
        for cid, cnt in sorted(train_r["class_counts"].items()):
            name  = class_names.get(cid, f"class_{cid}")
            dtype = name.split("_")[0] if "_" in name else name
            grouped[dtype].append((cid, name, cnt))

        for dtype, entries in sorted(grouped.items()):
            print(f"\n  {dtype}:")
            for (cid, name, cnt) in entries:
                bar  = "█" * (cnt // 5) + ("░" if cnt % 5 else "")
                warn = "  ← POUCOS EXEMPLOS" if cnt < MIN_EXAMPLES_PER_CLASS else ""
                print(f"    {name:<12} {cnt:>4}  {bar}{warn}")

        all_cids  = set(class_names.keys()) - {56}   # 56 = unknown
        seen_cids = set(train_r["class_counts"].keys())
        missing   = all_cids - seen_cids
        if missing:
            print(f"\n  [!] {len(missing)} classes sem nenhum exemplo em train:")
            for cid in sorted(missing):
                print(f"      {class_names[cid]}")

    print("\n" + "="*65)
    total_ok = sum(1 for cid, cnt in (train_r["class_counts"].items() if train_r else [])
                   if cnt >= MIN_EXAMPLES_PER_CLASS and cid != 56)
    total_cls = len(class_names) - 1
    print(f"  Classes prontas para treino: {total_ok}/{total_cls}")
    if total_ok == total_cls:
        print("  [✓] Dataset pronto! Execute: python training/train.py")
    else:
        print(f"  [ ] Complete a coleta para as classes com < {MIN_EXAMPLES_PER_CLASS} exemplos.")
    print("="*65 + "\n")


def fix_split(class_names):
    print("\n[INFO] Redistribuindo splits...")
    for split in SPLITS:
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for split in SPLITS:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split
        for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                all_pairs.append((img, lbl))

    if not all_pairs:
        print("[AVISO] Nenhum par encontrado.")
        return

    random.shuffle(all_pairs)
    n       = len(all_pairs)
    n_val   = max(1, int(n * SPLIT_RATIOS["val"]))
    n_test  = max(1, int(n * SPLIT_RATIOS["test"]))
    n_train = n - n_val - n_test

    assignment = ["train"]*n_train + ["val"]*n_val + ["test"]*n_test
    random.shuffle(assignment)

    moved = 0
    for (img_path, lbl_path), target in zip(all_pairs, assignment):
        dst_img = DATASET_ROOT / "images" / target / img_path.name
        dst_lbl = DATASET_ROOT / "labels" / target / lbl_path.name
        if img_path.resolve() != dst_img.resolve():
            shutil.move(str(img_path), str(dst_img))
        if lbl_path.resolve() != dst_lbl.resolve():
            shutil.move(str(lbl_path), str(dst_lbl))
        moved += 1

    print(f"[✓] {moved} pares redistribuídos: {n_train} train / {n_val} val / {n_test} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-split", action="store_true")
    args = parser.parse_args()

    class_names = load_classes()
    if args.fix_split:
        fix_split(class_names)

    results = [scan_split(split, class_names) for split in SPLITS]
    print_report(results, class_names)
