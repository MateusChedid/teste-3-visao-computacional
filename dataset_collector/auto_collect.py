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
from utils.roi_inference import (
    run_selector as run_octagon_selector,
    save_inference_roi as save_octagon,
    load_inference_roi as load_octagon,
    crop_to_polygon_bbox,
)

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

    Para evitar esticar/replicar bordas: a imagem de entrada deve ter uma
    margem extra em torno do dado (maior que o quadrado final desejado).
    Esta função rotaciona o quadrado inteiro e depois recorta o quadrado
    central "seguro" (inscrito), que nunca contém pixels replicados —
    apenas conteúdo real capturado pela câmera.

    Mantém as cores originais — sem nenhuma alteração de exposição/cor.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERRO] Não foi possível ler {img_path.name}")
        return []

    h, w = img.shape[:2]
    if h != w:
        side = min(h, w)
        cy, cx = h // 2, w // 2
        half = side // 2
        img = img[cy-half:cy+half, cx-half:cx+half]
        h = w = img.shape[0]

    # Tamanho do quadrado "seguro" inscrito após rotação:
    # para qualquer ângulo, um quadrado de lado L/sqrt(2) centralizado
    # dentro de um quadrado de lado L permanece sempre dentro após
    # qualquer rotação em torno do centro.
    safe_size = int(math.floor(h / math.sqrt(2)))
    if safe_size % 2 != 0:
        safe_size -= 1
    if safe_size < 10:
        safe_size = h  # imagem muito pequena, usa inteira mesmo

    out_dir.mkdir(parents=True, exist_ok=True)
    center = (w / 2.0, h / 2.0)
    step   = 360.0 / n_rotations
    half_safe = safe_size // 2
    cx, cy = w // 2, h // 2

    generated = []
    for i in range(n_rotations):
        angle   = i * step
        M       = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_full = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        # Recorte central seguro — só pixels reais, sem replicação
        rotated = rotated_full[cy-half_safe:cy+half_safe, cx-half_safe:cx+half_safe]

        out_path = out_dir / f"{img_path.stem}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated.append(out_path)

    return generated


# ─── Auto Boxer ───────────────────────────────────────────────────────────────

# Filtro de sanidade: a bbox precisa cobrir entre MIN_AREA_FRAC e MAX_AREA_FRAC
# da área total da imagem. Fora desse range, a detecção é rejeitada.
MIN_AREA_FRAC = 0.01   # bbox menor que 1% da imagem → provavelmente ruído
MAX_AREA_FRAC = 0.70   # bbox maior que 70% da imagem → provavelmente pegou o tray/fundo


def _box_area_frac(xmin, ymin, xmax, ymax, img_w, img_h):
    bw, bh = max(0, xmax-xmin), max(0, ymax-ymin)
    return (bw * bh) / (img_w * img_h)


def auto_bbox(model, img_path: Path, class_id: int, label_dir: Path, preview_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    img_h, img_w = img.shape[:2]

    # Tentar detecção em duas confianças
    results = model(str(img_path), conf=0.15, augment=True, verbose=False)
    boxes   = results[0].boxes
    if len(boxes) == 0:
        results = model(str(img_path), conf=0.05, augment=True, verbose=False)
        boxes   = results[0].boxes

    # Procurar, entre todas as boxes candidatas, a primeira que respeite
    # o filtro de tamanho (ordenadas por confiança, já vem ordenado pelo YOLO)
    valid_box = None
    for box in boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        frac = _box_area_frac(bx1, by1, bx2, by2, img_w, img_h)
        if MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC:
            valid_box = (bx1, by1, bx2, by2)
            break

    detected = valid_box is not None
    if detected:
        xmin, ymin, xmax, ymax = valid_box
    else:
        # Fallback: caixa central de tamanho fixo (40% da imagem)
        # — mais conservador que o fallback anterior de 70%
        m = 0.30
        xmin, ymin = int(img_w*m), int(img_h*m)
        xmax, ymax = int(img_w*(1-m)), int(img_h*(1-m))

    xmin, xmax = max(0, xmin), min(img_w, xmax)
    ymin, ymax = max(0, ymin), min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        xmin, ymin = int(img_w*.30), int(img_h*.30)
        xmax, ymax = int(img_w*.70), int(img_h*.70)
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


def octagon_to_square(img: np.ndarray, polygon: list) -> np.ndarray:
    """
    Recorta a imagem para a bounding box do octógono, mascara pixels fora
    dele (preto), e depois centraliza esse recorte num CANVAS QUADRADO
    (lado = maior dimensão da bbox), preenchendo a sobra com preto.

    Essa máscara/canvas é o que tanto o auto_collect (treino) quanto o
    detect.py (inferência) vão produzir a partir do octógono — garantindo
    consistência total de escala e formato.
    """
    masked, x, y = crop_to_polygon_bbox(img, polygon)
    h, w = masked.shape[:2]
    side = max(h, w)

    canvas = np.full((side, side, 3), 255, dtype=masked.dtype)
    off_y = (side - h) // 2
    off_x = (side - w) // 2
    canvas[off_y:off_y+h, off_x:off_x+w] = masked
    return canvas


# ─── Seleção manual de bbox (para fallbacks) ──────────────────────────────────

class _BoxDrawer:
    """Desenho de um único bbox por clique-e-arraste."""

    def __init__(self):
        self.start = None
        self.end   = None
        self.drawing = False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end   = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.drawing = False

    @property
    def box(self):
        if self.start is None or self.end is None:
            return None
        x1, y1 = self.start
        x2, y2 = self.end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return (x1, y1, x2, y2)


def manual_bbox_select(img_path: Path, stem_label: str) -> tuple | None:
    """
    Exibe a imagem e permite desenhar um bbox manualmente.
    Retorna (xc_frac, yc_frac, bw_frac, bh_frac) ou None se pulado.

    Controles:
      Arrastar      → desenhar caixa
      ENTER / C     → confirmar
      Z / R         → limpar e redesenhar
      Q             → pular esta face (mantém fallback automático)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    drawer = _BoxDrawer()
    win = f"Selecao manual - {stem_label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, drawer.mouse_cb)

    result = None
    while True:
        display = img.copy()
        box = drawer.box
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)

        cv2.rectangle(display, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(display,
                    f"{stem_label}: arraste a caixa no dado  "
                    f"ENTER=ok  Z=limpar  Q=pular face",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("c")):
            if box:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                xc = (x1 + bw/2) / w
                yc = (y1 + bh/2) / h
                result = (xc, yc, bw/w, bh/h)
                break

        elif key in (ord("z"), ord("r")):
            drawer.start = None
            drawer.end   = None

        elif key == ord("q"):
            result = None
            break

    cv2.destroyAllWindows()
    return result


def write_label_from_fraction(img_path: Path, class_id: int,
                               box_frac: tuple, label_dir: Path,
                               preview_dir: Path, tag: str = "MANUAL"):
    """
    Escreve o .txt YOLO e o preview a partir de uma bbox já em fração
    (xc, yc, bw, bh) — usado para aplicar a seleção manual a outras rotações.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    xc, yc, bwf, bhf = box_frac

    label_dir.mkdir(parents=True, exist_ok=True)
    with open(label_dir / (img_path.stem + ".txt"), "w") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bwf:.6f} {bhf:.6f}\n")

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = img.copy()
    x1 = int((xc - bwf/2) * w); x2 = int((xc + bwf/2) * w)
    y1 = int((yc - bhf/2) * h); y2 = int((yc + bhf/2) * h)
    color = (0, 200, 255)
    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
    cv2.putText(preview, tag, (x1, max(y1-8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(preview_dir / img_path.name), preview)


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
          f"{N_ROTATIONS - n_val_per_face} para train")
    print(f"  [i] Após rotacionar, recorta-se o quadrado central seguro "
          f"(~71% da ROI) para evitar bordas replicadas.")
    print(f"      Garanta que o dado caiba dentro dessa área central "
          f"mesmo girado — deixe margem na ROI.\n")

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
        cropped = octagon_to_square(img, roi) if roi else img

        tmp_path = WORK_DIR / "cropped" / f"{stem}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 97])

        rotated = generate_rotations(tmp_path, rot_dir, N_ROTATIONS)
        if not rotated:
            continue

        det_flags = [
            auto_bbox(model, r, class_id, lbl_dir, preview_dir)
            for r in rotated
        ]
        det_ok = sum(det_flags)
        fallback_idxs = [i for i, ok in enumerate(det_flags) if not ok]
        fallback_n = len(fallback_idxs)

        if fallback_n > 0:
            fallback_log.append(f"{stem}: {fallback_n}/{len(rotated)} fallback")

            print(f"    [!] {fallback_n} rotação(ões) caíram em fallback "
                  f"automático para '{stem}'.")
            ans = input("        Selecionar manualmente a região do dado "
                         "para corrigir? [S/n]: ").strip().lower()

            if ans != "n":
                # Mostrar a primeira rotação que caiu em fallback
                ref_idx = fallback_idxs[0]
                ref_img = rotated[ref_idx]
                print(f"        Desenhe a caixa em volta do {stem} "
                      f"(rotação {ref_idx:03d}).")
                box_frac = manual_bbox_select(ref_img, stem)

                if box_frac is not None:
                    # Aplicar a MESMA caixa (em fração) a todas as rotações
                    # que caíram em fallback para esta face
                    for i in fallback_idxs:
                        write_label_from_fraction(
                            rotated[i], class_id, box_frac,
                            lbl_dir, preview_dir, tag="MANUAL"
                        )
                    print(f"        [✓] Caixa manual aplicada a "
                          f"{fallback_n} rotação(ões).")
                else:
                    print("        [i] Pulado — mantido fallback automático.")

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
        print(f"\n  [AVISO] Faces com fallback (corrigidas manualmente se aceito):")
        for l in fallback_log:
            print(f"    {l}")
        print(f"  Verifique previews em {preview_dir}")

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

            # Cruz no centro da ROI — posição ideal para o dado
            roi_cx, roi_cy = (x1 + x2) // 2, (y1 + y2) // 2
            cross_len = max(12, (x2 - x1) // 12)
        else:
            roi_cx, roi_cy = w // 2, h // 2
            cross_len = max(12, min(w, h) // 24)

        # Desenhar cruz central (posição ideal para o dado)
        cv2.line(display, (roi_cx - cross_len, roi_cy), (roi_cx + cross_len, roi_cy),
                 (0, 0, 255), 2)
        cv2.line(display, (roi_cx, roi_cy - cross_len), (roi_cx, roi_cy + cross_len),
                 (0, 0, 255), 2)
        cv2.circle(display, (roi_cx, roi_cy), 4, (0, 0, 255), -1)

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

    # ── Octógono de coleta (contorno do tray) ─────────────────────────────────
    roi = None if args.clear_roi else load_octagon()

    if roi:
        print(f"\n[INFO] Octógono do tray carregado ({len(roi)} pontos)")
        if not args.skip_roi:
            ans = input("  Usar este octógono? [S/n]: ").strip().lower()
            if ans == "n":
                roi = None

    if roi is None and not args.skip_roi:
        print("\n[INFO] Selecione o contorno (octógono) do tray.")
        roi = run_octagon_selector(args.camera)
        if roi and len(roi) >= 3:
            save_octagon(roi)
        else:
            roi = None
            print("  [i] Sem octógono — usando frame inteiro (não recomendado).")

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