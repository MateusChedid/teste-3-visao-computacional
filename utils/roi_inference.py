#!/usr/bin/env python3
"""
roi_inference.py — Seleção de ROI poligonal (octógono) para inferência em tempo real.

Esta ROI é usada SOMENTE pelo detect.py — define a área visual onde o programa
procura por dados durante o uso em tempo real. Não afeta a coleta do dataset.

Uso:
    python utils/roi_inference.py
    python utils/roi_inference.py --camera 1

Controles:
    Clique esquerdo   → adicionar ponto
    Clique direito/Z  → remover último ponto
    ENTER ou C        → confirmar (1ª vez) / salvar (2ª vez)
    R                 → recomeçar
    Q                 → sair sem salvar
"""

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CAMERA_CONFIG = PROJECT_ROOT / "camera_config.yaml"


class PolygonSelector:
    def __init__(self):
        self.points   = []
        self.hover_pt = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []

    @property
    def polygon(self):
        return self.points if len(self.points) >= 3 else None


def render(frame, sel: PolygonSelector, confirmed: bool):
    display = frame.copy()
    h, w = display.shape[:2]
    pts = sel.points

    if len(pts) >= 3:
        poly = np.array(pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        dark = (display * 0.35).astype(np.uint8)
        display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)

        color = (0, 220, 80) if confirmed else (0, 180, 255)
        cv2.polylines(display, [poly], isClosed=True, color=color, thickness=2)

        area = cv2.contourArea(poly)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(display, f"{area/1000:.0f}k px2", (cx-30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    for i, p1 in enumerate(pts):
        cv2.circle(display, p1, 5, (0,200,255), -1)
        cv2.circle(display, p1, 7, (0,0,0), 1)
        cv2.putText(display, str(i+1), (p1[0]+8, p1[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)
        if i < len(pts)-1:
            cv2.line(display, p1, pts[i+1], (0,180,255), 1)

    if pts and sel.hover_pt and not confirmed:
        cv2.line(display, pts[-1], sel.hover_pt, (100,100,100), 1)

    cv2.rectangle(display, (0,0), (w,38), (0,0,0), -1)
    if confirmed:
        msg = f"Octogono confirmado ({len(pts)} pontos)!  ENTER=salvar  R=refazer"
        c = (80,220,80)
    elif len(pts) >= 3:
        msg = "ENTER=confirmar  Z=desfazer  R=recomecar"
        c = (220,220,220)
    elif pts:
        msg = f"{len(pts)} ponto(s) — minimo 3"
        c = (200,180,80)
    else:
        msg = "Clique nos cantos do octogono do tray  |  Q=sair"
        c = (180,180,180)
    cv2.putText(display, msg, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)

    cv2.rectangle(display, (0,h-26), (w,h), (0,0,0), -1)
    cv2.putText(display, "Clique esq=ponto  Clique dir/Z=desfazer  R=recomecar  Q=sair",
                (10,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1)

    return display


def run_selector(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERRO] Câmera não disponível.")
        return None

    sel = PolygonSelector()
    confirmed = False
    win = "Selecionar octogono do tray (inferencia)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sel.mouse_cb)

    print("\n" + "="*58)
    print("  SELEÇÃO DO OCTÓGONO — INFERÊNCIA EM TEMPO REAL")
    print("="*58)
    print("  Clique nos 8 cantos do tray em ordem.")
    print("  ENTER = confirmar   Z = desfazer   R = recomeçar\n")

    result = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = render(frame, sel, confirmed)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, ord("c")):
            if confirmed and sel.polygon:
                result = list(sel.points)
                break
            elif sel.polygon:
                confirmed = True

        elif key == ord("z"):
            sel.undo()
            confirmed = False

        elif key == ord("r"):
            sel.reset()
            confirmed = False

        elif key == ord("q"):
            result = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return result


def polygon_to_mask(polygon, frame_shape):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_polygon_mask(frame, polygon, darken_outside=True):
    if not polygon:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)
    mask = polygon_to_mask(polygon, frame.shape)
    if darken_outside:
        dark = (frame * 0.35).astype(np.uint8)
        result = np.where(np.stack([mask]*3, axis=2) > 0, frame, dark)
    else:
        result = frame.copy()
        result[mask == 0] = 0
    pts = np.array(polygon, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    return result, (x, y, bw, bh)


def crop_to_polygon_bbox(frame, polygon):
    if not polygon:
        return frame, 0, 0
    h, w = frame.shape[:2]
    pts = np.array(polygon, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x = max(0, x); y = max(0, y)
    bw = min(bw, w-x); bh = min(bh, h-y)
    crop = frame[y:y+bh, x:x+bw].copy()
    local_poly = pts - np.array([x, y])
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly], 255)
    crop[mask == 0] = 0
    return crop, x, y


def save_inference_roi(polygon):
    cfg = {}
    if CAMERA_CONFIG.exists():
        with open(CAMERA_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
    cfg["inference_roi_polygon"] = [[int(p[0]), int(p[1])] for p in polygon]
    with open(CAMERA_CONFIG, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"[✓] Octógono de inferência salvo ({len(polygon)} pontos)")


def load_inference_roi():
    if not CAMERA_CONFIG.exists():
        return None
    try:
        with open(CAMERA_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        poly = cfg.get("inference_roi_polygon")
        return [tuple(p) for p in poly] if poly else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    existing = load_inference_roi()
    if existing:
        print(f"\n[INFO] Octógono atual: {len(existing)} pontos")
        ans = input("  Redefinir? [s/N]: ").strip().lower()
        if ans != "s":
            print("  Mantido.")
            return

    polygon = run_selector(args.camera)
    if polygon and len(polygon) >= 3:
        save_inference_roi(polygon)
    else:
        print("\n[i] Seleção cancelada.")


if __name__ == "__main__":
    main()
