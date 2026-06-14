#!/usr/bin/env python3
"""
roi_collect.py — Seleção de ROI QUADRADA e ajustável para a coleta do dataset.

Por que quadrada: o pipeline de rotação recorta um quadrado central da imagem
para girar sem distorção. Se a ROI não for quadrada, a imagem é esticada antes
de rotacionar — o que distorce a forma do dado e contamina o dataset.

Esta ROI é usada SOMENTE na coleta (auto_collect.py). A inferência em tempo
real usa um seletor de octógono separado (utils/roi_inference.py).

Uso:
    python utils/roi_collect.py
    python utils/roi_collect.py --camera 1

Controles:
    Arrastar dentro do quadrado → mover
    Arrastar nas bordas/cantos  → redimensionar (mantém proporção quadrada)
    Scroll do mouse             → redimensionar (alternativa ao arrasto)
    ENTER ou C                  → confirmar e salvar
    R                            → resetar para quadrado central padrão
    Q                            → sair sem salvar
"""

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CAMERA_CONFIG = PROJECT_ROOT / "camera_config.yaml"

HANDLE_SIZE = 12   # raio de detecção dos cantos para redimensionar


class SquareROI:
    """ROI quadrada com arrasto para mover e cantos para redimensionar."""

    def __init__(self, frame_w, frame_h):
        # Quadrado inicial: centrado, 50% da menor dimensão
        size = int(min(frame_w, frame_h) * 0.5)
        cx, cy = frame_w // 2, frame_h // 2
        self.x = cx - size // 2
        self.y = cy - size // 2
        self.size = size
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.dragging   = False
        self.resizing   = False
        self.drag_start = None
        self.orig_state = None

    def reset(self):
        size = int(min(self.frame_w, self.frame_h) * 0.5)
        cx, cy = self.frame_w // 2, self.frame_h // 2
        self.x = cx - size // 2
        self.y = cy - size // 2
        self.size = size

    @property
    def rect(self):
        """Retorna (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.size, self.y + self.size)

    def _corner_near(self, px, py):
        """Retorna qual canto está perto de (px,py), ou None."""
        x1, y1, x2, y2 = self.rect
        corners = {
            "tl": (x1, y1), "tr": (x2, y1),
            "bl": (x1, y2), "br": (x2, y2),
        }
        for name, (cx, cy) in corners.items():
            if abs(px - cx) <= HANDLE_SIZE and abs(py - cy) <= HANDLE_SIZE:
                return name
        return None

    def _inside(self, px, py):
        x1, y1, x2, y2 = self.rect
        return x1 <= px <= x2 and y1 <= py <= y2

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corner = self._corner_near(x, y)
            if corner:
                self.resizing   = corner
                self.orig_state = (self.x, self.y, self.size)
                self.drag_start = (x, y)
            elif self._inside(x, y):
                self.dragging   = True
                self.drag_start = (x, y)
                self.orig_state = (self.x, self.y, self.size)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_start:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                ox, oy, osz = self.orig_state
                self.x = int(np.clip(ox + dx, 0, self.frame_w - osz))
                self.y = int(np.clip(oy + dy, 0, self.frame_h - osz))

            elif self.resizing and self.drag_start:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                # Usa a maior variação (mantém quadrado)
                delta = dx if abs(dx) > abs(dy) else dy
                ox, oy, osz = self.orig_state

                if self.resizing == "br":
                    new_size = osz + delta
                elif self.resizing == "tl":
                    new_size = osz - delta
                elif self.resizing == "tr":
                    new_size = osz + (dx if abs(dx) > abs(dy) else -dy)
                elif self.resizing == "bl":
                    new_size = osz + (-dx if abs(dx) > abs(dy) else dy)
                else:
                    new_size = osz

                new_size = int(np.clip(new_size, 60,
                                       min(self.frame_w, self.frame_h)))

                # Ajustar posição para manter o canto oposto fixo
                if self.resizing == "br":
                    self.size = new_size
                elif self.resizing == "tl":
                    self.x = ox + (osz - new_size)
                    self.y = oy + (osz - new_size)
                    self.size = new_size
                elif self.resizing == "tr":
                    self.y = oy + (osz - new_size)
                    self.size = new_size
                elif self.resizing == "bl":
                    self.x = ox + (osz - new_size)
                    self.size = new_size

                self.x = int(np.clip(self.x, 0, self.frame_w - self.size))
                self.y = int(np.clip(self.y, 0, self.frame_h - self.size))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.drag_start = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 10 if flags > 0 else -10
            new_size = int(np.clip(self.size + delta, 60,
                                   min(self.frame_w, self.frame_h)))
            cx = self.x + self.size // 2
            cy = self.y + self.size // 2
            self.size = new_size
            self.x = int(np.clip(cx - new_size // 2, 0, self.frame_w - new_size))
            self.y = int(np.clip(cy - new_size // 2, 0, self.frame_h - new_size))


def render(frame, roi: SquareROI, confirmed=False):
    display = frame.copy()
    h, w = display.shape[:2]
    x1, y1, x2, y2 = roi.rect

    # Escurecer fora do quadrado
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    dark = (display * 0.35).astype(np.uint8)
    display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)

    color = (0, 220, 80) if confirmed else (0, 180, 255)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    # Guia da "zona segura" pós-rotação (~71% do quadrado, centralizado)
    # — o dado deve ficar dentro desta área interna mesmo após girar
    safe_size = int(roi.size / 1.41421356)
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    sx1, sy1 = cx - safe_size//2, cy - safe_size//2
    sx2, sy2 = cx + safe_size//2, cy + safe_size//2
    cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(display, "zona segura (pos-rotacao)", (sx1, sy1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

    # Desenhar alças nos 4 cantos
    for (cx, cy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        cv2.circle(display, (cx, cy), 6, color, -1)
        cv2.circle(display, (cx, cy), 8, (0,0,0), 1)

    # Tamanho
    cv2.putText(display, f"{roi.size} x {roi.size} px",
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # HUD
    cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
    if confirmed:
        msg = "Area confirmada!  ENTER=salvar  R=resetar"
    else:
        msg = "Arraste para mover, cantos para redimensionar (fica quadrado)"
    cv2.putText(display, msg, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    cv2.rectangle(display, (0, h-26), (w, h), (0,0,0), -1)
    cv2.putText(display,
                "ENTER=confirmar  R=resetar  Q=sair sem salvar",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1)

    return display


def run_selector(camera_index: int = 0):
    """Retorna (x1,y1,x2,y2) em pixels ou None se cancelado."""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERRO] Câmera não disponível.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível ler frame da câmera.")
        cap.release()
        return None

    h, w = frame.shape[:2]
    roi  = SquareROI(w, h)
    confirmed = False

    win = "Selecionar area do dice tray (QUADRADO)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, roi.mouse_cb)

    print("\n" + "="*58)
    print("  SELEÇÃO DA ÁREA DO DICE TRAY (QUADRADO)")
    print("="*58)
    print("  Arraste o quadrado para a área do tray.")
    print("  Use os cantos para redimensionar (mantém quadrado).")
    print("  A linha pontilhada interna é a 'zona segura' pós-rotação:")
    print("  o DADO deve caber dentro dela mesmo girado 360°.")
    print("  ENTER = confirmar   R = resetar   Q = sair\n")

    result = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = render(frame, roi, confirmed)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, ord("c")):
            if confirmed:
                result = roi.rect
                break
            confirmed = True

        elif key == ord("r"):
            roi.reset()
            confirmed = False

        elif key == ord("q"):
            result = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return result


def save_collect_roi(rect):
    cfg = {}
    if CAMERA_CONFIG.exists():
        with open(CAMERA_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
    cfg["collect_roi"] = [int(v) for v in rect]
    with open(CAMERA_CONFIG, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"[✓] ROI de coleta salva: {rect}")


def load_collect_roi():
    if not CAMERA_CONFIG.exists():
        return None
    try:
        with open(CAMERA_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        rect = cfg.get("collect_roi")
        return tuple(int(v) for v in rect) if rect else None
    except Exception:
        return None


def crop_to_roi(image: np.ndarray, rect) -> np.ndarray:
    """Recorta a imagem para o retângulo (x1,y1,x2,y2). Garante quadrado."""
    if rect is None:
        return image
    x1, y1, x2, y2 = rect
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    existing = load_collect_roi()
    if existing:
        print(f"\n[INFO] ROI de coleta atual: {existing}")
        ans = input("  Redefinir? [s/N]: ").strip().lower()
        if ans != "s":
            print("  Mantida.")
            return

    rect = run_selector(args.camera)
    if rect:
        save_collect_roi(rect)
    else:
        print("\n[i] Seleção cancelada.")


if __name__ == "__main__":
    main()
