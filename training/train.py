#!/usr/bin/env python3
"""
train.py — Treina YOLOv8 para detecção de dados de RPG.

Uso:
    python training/train.py
    python training/train.py --resume
    python training/train.py --model yolov8s --epochs 100
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[ERRO] Execute: pip install ultralytics torch")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "dataset" / "dataset.yaml"
CONFIG_YAML  = PROJECT_ROOT / "training" / "config.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"


def load_config():
    with open(CONFIG_YAML) as f:
        return yaml.safe_load(f)


def check_dataset():
    images_train = PROJECT_ROOT / "dataset" / "images" / "train"
    images_val   = PROJECT_ROOT / "dataset" / "images" / "val"
    labels_train = PROJECT_ROOT / "dataset" / "labels" / "train"

    n_train = len(list(images_train.glob("*.jpg")) + list(images_train.glob("*.png")))
    n_val   = len(list(images_val.glob("*.jpg"))   + list(images_val.glob("*.png")))
    n_lbl   = len(list(labels_train.glob("*.txt")))

    print(f"  Train: {n_train} imagens, {n_lbl} labels")
    print(f"  Val:   {n_val} imagens")

    if n_train < 50:
        print("[AVISO] Poucas imagens de treino.")
    if n_val < 10:
        print("[AVISO] Poucas imagens de validação.")
    if n_lbl < n_train * 0.8:
        print(f"[AVISO] Apenas {n_lbl}/{n_train} imagens têm labels.")

    return n_train > 0 and n_val > 0 and n_lbl > 0


def print_gpu_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print("  Nenhuma GPU — usando CPU (treino lento)")


def main(args):
    print("\n" + "="*60)
    print("  RPG DICE CV v2 — TREINO YOLOV8")
    print("="*60)

    cfg = load_config()

    model_name = args.model if args.model else cfg.get("model", "yolov8m.pt")
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    epochs = args.epochs or cfg.get("epochs", 150)
    batch  = args.batch  or cfg.get("batch", 8)
    imgsz  = args.imgsz  or cfg.get("imgsz", 640)
    device = args.device or cfg.get("device", "")
    exp_name = f"rpg_dice_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n  Modelo:  {model_name}")
    print(f"  Épocas:  {epochs}  |  Batch: {batch}  |  ImgSz: {imgsz}")
    print(f"  Device:  {device or 'auto'}")
    print(f"  Dataset: {DATASET_YAML}")
    print(f"  Saída:   runs/train/{exp_name}\n")

    print("[INFO] Hardware:")
    print_gpu_info()

    print("\n[INFO] Verificando dataset...")
    if not check_dataset():
        print("[ERRO] Dataset inválido. Execute auto_collect.py primeiro.")
        sys.exit(1)

    if not args.resume:
        model = YOLO(model_name)
    else:
        last_pt = sorted(RUNS_DIR.glob("train/*/weights/last.pt"))
        if not last_pt:
            print("[ERRO] Nenhum checkpoint encontrado para --resume.")
            sys.exit(1)
        print(f"[INFO] Retomando de: {last_pt[-1]}")
        model = YOLO(str(last_pt[-1]))

    print("\n[INICIO] Treinamento iniciado...\n")

    results = model.train(
        data         = str(DATASET_YAML),
        epochs       = epochs,
        patience     = cfg.get("patience", 25),
        batch        = batch,
        imgsz        = imgsz,
        device       = device or None,
        workers      = cfg.get("workers", 0),
        project      = str(RUNS_DIR / "train"),
        name         = exp_name if not args.resume else None,
        resume       = args.resume,
        optimizer    = cfg.get("optimizer", "AdamW"),
        lr0          = cfg.get("lr0", 0.001),
        lrf          = cfg.get("lrf", 0.01),
        momentum     = cfg.get("momentum", 0.937),
        weight_decay = cfg.get("weight_decay", 0.0005),
        warmup_epochs= cfg.get("warmup_epochs", 3.0),
        hsv_h        = cfg.get("hsv_h", 0.015),
        hsv_s        = cfg.get("hsv_s", 0.5),
        hsv_v        = cfg.get("hsv_v", 0.3),
        degrees      = cfg.get("degrees", 180.0),
        translate    = cfg.get("translate", 0.1),
        scale        = cfg.get("scale", 0.3),
        shear        = cfg.get("shear", 5.0),
        flipud       = cfg.get("flipud", 0.3),
        fliplr       = cfg.get("fliplr", 0.5),
        mosaic       = cfg.get("mosaic", 0.8),
        mixup        = cfg.get("mixup", 0.1),
        copy_paste   = cfg.get("copy_paste", 0.1),
        box          = cfg.get("box", 7.5),
        cls          = cfg.get("cls", 0.5),
        dfl          = cfg.get("dfl", 1.5),
        save         = True,
        save_period  = cfg.get("save_period", 10),
        plots        = True,
        verbose      = True,
        val          = True,
        amp          = True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "="*60)
    print("  TREINO CONCLUÍDO")
    print(f"\n  Melhor modelo: {best_pt}")

    if best_pt.exists():
        val_model   = YOLO(str(best_pt))
        val_results = val_model.val(data=str(DATASET_YAML))
        print(f"\n  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
        print(f"  Precisão: {val_results.box.mp:.4f}")
        print(f"  Recall:   {val_results.box.mr:.4f}")

    print(f"\n  Inferência: python inference/detect.py --weights \"{best_pt}\"")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch",  type=int, default=None)
    parser.add_argument("--imgsz",  type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
