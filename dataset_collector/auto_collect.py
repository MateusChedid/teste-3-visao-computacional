#!/usr/bin/env python3
"""
auto_collect.py v3 — Coleta simplificada do dataset.

Regras desta versão:
  • Câmera mantém configurações padrão do sistema (fundo branco, sem ajuste)
  • ROI de coleta é QUADRADA (selecionada com utils/roi_collect.py)
  • 1 foto por face → gera 80 rotações (passo de 4.5°)
  • SEM augmentation de exposição, SEM conversão para P&B
  • Imagens vão para o dataset exatamente como capturadas (apenas rotacionadas)

Fluxo:
  1. Seleciona/carrega a ROI quadrada do tray
  2. Para cada face: 1 foto vai para um pool, dividido depois em train/val
  3. Gera 80 rotações da foto (0° a 360°, passo 4.5°)
  4. YOLOv8 genérico gera a bbox automaticamente para cada rotação
  5. Distribui em train (queixo a maioria) / val (uma fração)

Uso:
    python dataset_collector/auto_collect.py
    python dataset_collector/auto_collect.py --from-folder
    python dataset_collector/auto_collect.py --skip-roi
    python dataset_collector/auto_collect.py --clear-roi
    python dataset_collector/auto_collect.py --dice d20 --face 17
"""

import cv2
import math
import numpy as np
import yaml
import argparse
import shutil
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
INPUT_DIR    = PROJECT_ROOT / "dataset_collector" / "input"
WORK_DIR     = PROJECT_ROOT / "dataset_collector" / "_work"

sys.path.insert(0, str(PROJECT_ROOT))
from utils.roi_collect import load_collect_roi, run_selector, save_collect_roi, crop_to_roi

N_ROTATIONS  = 80                       # rotações por foto
ROTATION_STEP_DEG = 360.0 / N_ROTATIONS  # = 4.5°

# Fração de rotações que vai para val (o restante vai para train)
VAL_FRACTION = 0.10   # 10% das 80 rotações → val (= 8 imagens)

DICE_FACES = {
    "d6":  list(range(1, 7)),
    "d8":  list(range(1, 9)),
    "d10": list(range(0, 10)),
    "d12": list(range(1, 13)),
    "d20": list(range(1, 21)),
}


# ─── Classes ──────────────────────────────────────────────────────────────────

def load_class_map():
    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)
    return {v: k for k, v in cfg["names"].items()}


# ─── Rotator (sem augmentation, sem grayscale) ────────────────────────────────

def generate_rotations(img_path: Path, out_dir: Path, n_rotations: int = N_ROTATIONS):
    """
    Gera n_rotations rotações da imagem, igualmente espaçadas em 360°.
    A imagem já deve estar recortada (quadrada) antes de chamar esta função.
    Mantém as cores originais — sem nenhuma alteração de exposição/cor.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERRO] Não foi possível ler {img_path.name}")
        return []

    h, w = img.shape[:2]
    if h != w:
        # Garantir quadrado: recortar para o menor lado, centralizado
        side = min(h, w)
        cy, cx = h // 2, w // 2
        half = side // 2
        img = img[cy-half:cy+half, cx-half:cx+half]
        h = w = img.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    center = (w / 2.0, h / 2.0)
    step   = 360.0 / n_rotations

    generated = []
    for i in range(n_rotations):
        angle = i * step
        M       = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_path = out_dir / f"{img_path.stem}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated.append(out_path)

    return generated


# ─── Auto Boxer ───────────────────────────────────────────────────────────────

def auto_bbox(model, img_path: Path, class_id: int, label_dir: Path, preview_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    img_h, img_w = img.shape[:2]

    results = model(str(img_path), conf=0.15, augment=True, verbose=False)
    boxes   = results[0].boxes
    if len(boxes) == 0:
        results = model(str(img_path), conf=0.05, augment=True, verbose=False)
        boxes   = results[0].boxes

    detected = len(boxes) > 0
    if detected:
        xmin, ymin, xmax, ymax = map(int, boxes[0].xyxy[0].tolist())
    else:
        m = 0.15
        xmin, ymin = int(img_w*m), int(img_h*m)
        xmax, ymax = int(img_w*(1-m)), int(img_h*(1-m))

    xmin, xmax = max(0, xmin), min(img_w, xmax)
    ymin, ymax = max(0, ymin), min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        xmin, ymin = int(img_w*.15), int(img_h*.15)
        xmax, ymax = int(img_w*.85), int(img_h*.85)
        bw, bh = xmax - xmin, ymax - ymin

    xc = (xmin + bw/2) / img_w
    yc = (ymin + bh/2) / img_h

    label_dir.mkdir(parents=True, exist_ok=True)
    with open(label_dir / (img_path.stem + ".txt"), "w") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = img.copy()
    color   = (0, 200, 80) if detected else (80, 80, 220)
    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), color, 3)
    cv2.putText(preview, "OK" if detected else "FALLBACK",
                (xmin, max(ymin - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(preview_dir / img_path.name), preview)

    return detected


# ─── Distribuição em splits ────────────────────────────────────────────────────

def copy_to_split(img_path: Path, lbl_path: Path, split: str):
    for subdir, src in [("images", img_path), ("labels", lbl_path)]:
        dst = DATASET_ROOT / subdir / split / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(images: list, class_map: dict, model, roi):
    """
    images: lista de (img_path, class_id)
    Para cada foto: gera N_ROTATIONS rotações, bbox automática,
    e divide entre train/val por VAL_FRACTION.
    """
    preview_dir  = WORK_DIR / "bbox_preview"
    fallback_log = []
    counts       = {"train": 0, "val": 0}
    n_val_per_face = max(1, int(N_ROTATIONS * VAL_FRACTION))

    print(f"\n  Cada foto gera {N_ROTATIONS} rotações "
          f"(passo {ROTATION_STEP_DEG:.1f}°)")
    print(f"  {n_val_per_face} vão para val, "
          f"{N_ROTATIONS - n_val_per_face} para train\n")

    for idx, (img_path, class_id) in enumerate(images, 1):
        stem    = img_path.stem
        rot_dir = WORK_DIR / "rotations" / stem
        lbl_dir = WORK_DIR / "labels"    / stem

        print(f"  [{idx}/{len(images)}] {stem}")

        # Recortar para ROI quadrada antes de rotacionar
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [ERRO] não foi possível ler {img_path.name}")
            continue
        cropped = crop_to_roi(img, roi)

        tmp_path = WORK_DIR / "cropped" / f"{stem}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 97])

        rotated = generate_rotations(tmp_path, rot_dir, N_ROTATIONS)
        if not rotated:
            continue

        det_ok = sum(
            auto_bbox(model, r, class_id, lbl_dir, preview_dir)
            for r in rotated
        )
        fallback_n = len(rotated) - det_ok
        if fallback_n > 0:
            fallback_log.append(f"{stem}: {fallback_n}/{len(rotated)} fallback")

        # Dividir entre val e train (índices aleatórios para val)
        idxs = list(range(len(rotated)))
        random.shuffle(idxs)
        val_idxs = set(idxs[:n_val_per_face])

        for i, rot_img in enumerate(rotated):
            lbl = lbl_dir / (rot_img.stem + ".txt")
            if not lbl.exists():
                continue
            split = "val" if i in val_idxs else "train"
            copy_to_split(rot_img, lbl, split)
            counts[split] += 1

        print(f"    {len(rotated)} rotações — {det_ok} OK, {fallback_n} fallback")

    print(f"\n  [✓] Dataset gerado:")
    print(f"      train: {counts['train']} imagens")
    print(f"      val:   {counts['val']} imagens")

    if fallback_log:
        print(f"\n  [AVISO] Verifique previews em {preview_dir}:")
        for l in fallback_log:
            print(f"    {l}")

    print(f"\n  Próximo passo: python dataset_collector/validate_dataset.py\n")


# ─── Webcam capture ────────────────────────────────────────────────────────────

def webcam_capture_one(dice_type: str, face: int, camera_index: int, roi) -> Path | None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERRO] Câmera {camera_index} não disponível.")
        return None

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    face_str = str(face)
    saved    = None
    cv2.namedWindow("Captura", cv2.WINDOW_NORMAL)

    print(f"\n  {dice_type.upper()} — face {face_str}")
    print("  ESPAÇO = capturar   Q = pular\n")

    while saved is None:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        if roi:
            x1, y1, x2, y2 = roi
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            dark = (display * 0.4).astype(np.uint8)
            display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 80), 2)

        cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(display,
                    f"{dice_type.upper()} face {face_str}  ESPACO=capturar  Q=pular",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

        cv2.imshow("Captura", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            fname = f"{dice_type}_{face_str}.jpg"
            path  = INPUT_DIR / fname
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved = path
            flash = frame.copy()
            if roi:
                x1,y1,x2,y2 = roi
                cv2.rectangle(flash, (x1,y1), (x2,y2), (60,220,60), 4)
            cv2.putText(flash, "SALVO!", (w//2-80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (60,220,60), 3)
            cv2.imshow("Captura", flash)
            cv2.waitKey(400)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coleta de dataset v3 — ROI quadrada, 80 rotações")
    parser.add_argument("--from-folder", action="store_true",
                        help="Processa imagens em dataset_collector/input/ (nome: d6_1.jpg)")
    parser.add_argument("--skip-roi",  action="store_true",
                        help="Usar ROI salva sem perguntar")
    parser.add_argument("--clear-roi", action="store_true",
                        help="Ignorar ROI salva e selecionar nova")
    parser.add_argument("--dice", type=str, default=None, choices=list(DICE_FACES.keys()))
    parser.add_argument("--face", type=int, default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    class_map = load_class_map()

    # ── ROI quadrada de coleta ────────────────────────────────────────────────
    roi = None if args.clear_roi else load_collect_roi()

    if roi:
        print(f"\n[INFO] ROI de coleta carregada: {roi}")
        if not args.skip_roi:
            ans = input("  Usar esta ROI? [S/n]: ").strip().lower()
            if ans == "n":
                roi = None

    if roi is None and not args.skip_roi:
        print("\n[INFO] Selecione a área quadrada do tray.")
        roi = run_selector(args.camera)
        if roi:
            save_collect_roi(roi)
        else:
            print("  [i] Sem ROI — usando frame inteiro (não recomendado).")

    # ── Modelo de detecção automática ─────────────────────────────────────────
    print("\n[INFO] Carregando modelo de detecção automática...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
    except ImportError:
        print("[ERRO] Execute: pip install ultralytics")
        sys.exit(1)

    # ── Modo pasta ─────────────────────────────────────────────────────────────
    if args.from_folder:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in valid_ext)
        if not all_images:
            print(f"[ERRO] Nenhuma imagem em {INPUT_DIR}")
            print("       Nome esperado: d6_1.jpg, d20_17.jpg, d10_0.jpg, etc.")
            sys.exit(1)

        images = []
        for img_path in all_images:
            parts = img_path.stem.split("_")
            if len(parts) < 2:
                print(f"  [AVISO] Nome inválido: {img_path.name}")
                continue
            key = f"{parts[0]}_{parts[1]}"
            if key not in class_map:
                print(f"  [AVISO] '{key}' não existe no yaml")
                continue
            images.append((img_path, class_map[key]))

        print(f"\n[INFO] {len(images)} fotos encontradas\n")
        run_pipeline(images, class_map, model, roi)
        return

    # ── Modo face específica ─────────────────────────────────────────────────
    if args.dice and args.face is not None:
        photo = webcam_capture_one(args.dice, args.face, args.camera, roi)
        if not photo:
            return
        key = f"{args.dice}_{args.face}"
        cid = class_map.get(key)
        if cid is None:
            print(f"[ERRO] Classe '{key}' não encontrada.")
            return
        run_pipeline([(photo, cid)], class_map, model, roi)
        return

    # ── Modo interativo completo ─────────────────────────────────────────────
    print("\n" + "="*58)
    print("  RPG DICE AUTO COLLECT v3")
    print("="*58)
    print(f"  1 foto por face → {N_ROTATIONS} rotações automáticas\n")

    images = []
    for dice_type, faces in DICE_FACES.items():
        print(f"\n{'─'*40}")
        ans = input(f"  Coletar {dice_type}? [S/n]: ").strip().lower()
        if ans == "n":
            continue

        for face in faces:
            key = f"{dice_type}_{face}"
            cid = class_map.get(key)
            if cid is None:
                continue

            existing = INPUT_DIR / f"{dice_type}_{face}.jpg"
            if existing.exists():
                print(f"  [✓] {key} já existe — pulando")
                images.append((existing, cid))
                continue

            photo = webcam_capture_one(dice_type, face, args.camera, roi)
            if photo:
                images.append((photo, cid))

    if not images:
        print("\n[INFO] Nenhuma imagem capturada.")
        return

    print(f"\n[INFO] {len(images)} fotos. Iniciando pipeline...\n")
    run_pipeline(images, class_map, model, roi)


if __name__ == "__main__":
    main()
