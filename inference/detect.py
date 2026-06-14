#!/usr/bin/env python3
"""
detect.py v3 — Inferência em tempo real.

Mudanças v3:
  • SEM conversão para escala de cinza (modelo treinado em RGB original)
  • SEM aplicação de configuração de exposição (câmera no padrão do sistema)
  • Octógono de inferência opcional, definido via utils/roi_inference.py

Uso:
    python inference/detect.py                        # webcam
    python inference/detect.py --source foto.jpg       # imagem
    python inference/detect.py --weights caminho/best.pt
    python inference/detect.py --no-roi                # ignora octógono salvo

Controles:
    ESPAÇO → congelar frame e exibir resultado
    R      → voltar ao live feed
    S      → salvar screenshot
    Q      → sair
"""

import cv2
import numpy as np
import argparse
import sys
import yaml
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML  = PROJECT_ROOT / "training" / "config.yaml"
DATASET_YAML = PROJECT_ROOT / "dataset" / "dataset.yaml"

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERRO] Execute: pip install ultralytics")
    sys.exit(1)

sys.path.insert(0, str(PROJECT_ROOT))
from inference.result_reader import interpret_detections, RollResult
from utils.roi_inference import (
    load_inference_roi, crop_to_polygon_bbox, apply_polygon_mask
)

DICE_COLORS_BGR = {
    "d6":  (80,  200, 80),
    "d8":  (80,  160, 220),
    "d10": (220, 150, 80),
    "d12": (80,  200, 200),
    "d20": (200, 80,  220),
    "unknown": (120, 120, 120),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─── ROI helpers ───────────────────────────────────────────────────────────────

def apply_roi(frame, polygon):
    """Recorta frame para bbox do octógono. Retorna (crop, off_x, off_y)."""
    if polygon is None:
        return frame, 0, 0
    return crop_to_polygon_bbox(frame, polygon)


def draw_roi_overlay(frame, polygon):
    if polygon is None:
        return frame
    out, _ = apply_polygon_mask(frame, polygon, darken_outside=True)
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 80), thickness=2)
    return out


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_YAML) as f:
        return yaml.safe_load(f)


def load_class_names():
    with open(DATASET_YAML) as f:
        return yaml.safe_load(f)["names"]


def find_best_weights():
    candidates = sorted((PROJECT_ROOT / "runs" / "train").glob("*/weights/best.pt"))
    return candidates[-1] if candidates else None


# ─── Renderização ──────────────────────────────────────────────────────────────

def draw_detection(frame, die, color):
    x1, y1, x2, y2 = [int(v) for v in die.bbox]
    label = f"{die.die_type}={die.display_value}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
    cv2.putText(frame, label, (x1+3, y1-5), FONT, 0.65, (0,0,0), 2)
    cv2.putText(frame, f"{die.confidence:.0%}", (x2-45, y2-6), FONT, 0.42, color, 1)


def draw_result_panel(frame, roll: RollResult, fps: float = 0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-60), (w, h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if roll.dice:
        cv2.putText(frame, roll.summary(), (12, h-38), FONT, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"TOTAL: {roll.total()}", (12, h-10), FONT, 0.75, (80,220,120), 2)
    else:
        cv2.putText(frame, "Nenhum dado detectado", (12, h-30), FONT, 0.65, (160,160,160), 1)

    if fps > 0:
        cv2.putText(frame, f"{fps:.0f} fps", (w-75, h-38), FONT, 0.45, (100,100,100), 1)


def draw_hud_top(frame, frozen=False, roi_active=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    msg = "RPG Dice CV v3  |  ESPACO=congelar  S=salvar  Q=sair"
    cv2.putText(frame, msg, (8, 21), FONT, 0.44, (180,180,180), 1)
    if frozen:
        cv2.putText(frame, "[ CONGELADO ]", (w-175, 21), FONT, 0.55, (0,200,255), 2)
    if not roi_active:
        cv2.putText(frame, "[ sem octogono — frame inteiro ]",
                    (w-300, h-1) if False else (8, 50),
                    FONT, 0.4, (180,150,80), 1)


def yolo_to_detections(yolo_results, class_names, offset_x=0, offset_y=0):
    detections = []
    for result in yolo_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            x1 += offset_x; x2 += offset_x
            y1 += offset_y; y2 += offset_y
            cls_name = class_names.get(cid, "unknown")
            detections.append((cls_name, conf, (x1, y1, x2, y2)))
    return detections


def render_frame(frame, roll, frozen=False, fps=0, roi_active=False):
    out = frame.copy()
    for die in roll.dice:
        color = DICE_COLORS_BGR.get(die.die_type, (200,200,200))
        draw_detection(out, die, color)
    draw_result_panel(out, roll, fps)
    draw_hud_top(out, frozen, roi_active)
    return out


# ─── Webcam ────────────────────────────────────────────────────────────────────

def run_webcam(model, class_names, cfg, camera_index=0, use_roi=True):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERRO] Câmera {camera_index} não disponível.")
        sys.exit(1)

    conf_thr = cfg.get("conf_threshold", 0.45)
    screenshots_dir = PROJECT_ROOT / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    roi = load_inference_roi() if use_roi else None
    if roi:
        print(f"[INFO] Octógono de inferência ativo ({len(roi)} pontos)")
    else:
        print("[INFO] Sem octógono — processando frame inteiro.")
        print("       Para definir: python utils/roi_inference.py")

    cv2.namedWindow("RPG Dice Detector", cv2.WINDOW_NORMAL)
    frozen = False; frozen_frame = None; frozen_roll = None
    prev_time = time.time(); fps = 0.0

    print("[OK] Câmera aberta. Aponte para os dados e pressione ESPAÇO.\n")

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.time()
            fps = 0.9*fps + 0.1*(1.0/max(now-prev_time, 1e-5))
            prev_time = now

            # Recortar para o octógono (se definido) — sem alterar cor/exposição
            crop, off_x, off_y = apply_roi(frame, roi)

            results    = model.predict(crop, conf=conf_thr, verbose=False,
                                       iou=cfg.get("iou_threshold", 0.45))
            detections = yolo_to_detections(results, class_names, off_x, off_y)
            roll       = interpret_detections(detections, conf_threshold=conf_thr)

            display = draw_roi_overlay(frame.copy(), roi)
            display = render_frame(display, roll, fps=fps, roi_active=bool(roi))
        else:
            display = render_frame(frozen_frame, frozen_roll, frozen=True, roi_active=bool(roi))

        cv2.imshow("RPG Dice Detector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if not frozen:
                frozen = True; frozen_frame = frame.copy(); frozen_roll = roll
                print(f"\n[RESULTADO] {roll.summary()}")
            else:
                frozen = False
        elif key == ord("r"):
            frozen = False
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = screenshots_dir / f"result_{ts}.jpg"
            cv2.imwrite(str(path), display)
            print(f"[✓] Screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()


# ─── Imagem estática ───────────────────────────────────────────────────────────

def run_image(model, class_names, cfg, source, use_roi=True):
    conf_thr = cfg.get("conf_threshold", 0.45)
    source   = Path(source)
    images   = list(source.glob("*.jpg")) + list(source.glob("*.png")) \
               if source.is_dir() else [source]

    out_dir = PROJECT_ROOT / "inference_results"
    out_dir.mkdir(exist_ok=True)

    roi = load_inference_roi() if use_roi else None
    if roi:
        print(f"[INFO] Octógono de inferência ativo ({len(roi)} pontos)")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        crop, off_x, off_y = apply_roi(frame, roi)
        results    = model.predict(crop, conf=conf_thr, verbose=False)
        detections = yolo_to_detections(results, class_names, off_x, off_y)
        roll       = interpret_detections(detections, conf_threshold=conf_thr)

        display = draw_roi_overlay(frame.copy(), roi)
        display = render_frame(display, roll, roi_active=bool(roi))
        cv2.imwrite(str(out_dir / img_path.name), display)
        print(f"  {img_path.name} → {roll.summary()}")

        if len(images) == 1:
            cv2.imshow("Resultado", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\n[✓] Resultados em: {out_dir}")


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="0")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--no-roi",  action="store_true",
                        help="Ignora o octógono salvo e processa o frame inteiro")
    args = parser.parse_args()

    cfg         = load_config()
    class_names = load_class_names()

    weights = args.weights or find_best_weights()
    if weights is None:
        print("[ERRO] Nenhum modelo treinado. Execute: python training/train.py")
        sys.exit(1)

    print(f"[INFO] Carregando modelo: {weights}")
    model = YOLO(str(weights))

    use_roi = not args.no_roi

    try:
        run_webcam(model, class_names, cfg, int(args.source), use_roi)
    except ValueError:
        run_image(model, class_names, cfg, args.source, use_roi)


if __name__ == "__main__":
    main()
