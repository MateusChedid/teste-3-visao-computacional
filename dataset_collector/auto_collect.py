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
    crop_to_polygon_bbox_paper,
)

N_ROTATIONS  = 90                       # rotações por foto (passo 4°, igual ref.)
ROTATION_STEP_DEG = 360.0 / N_ROTATIONS  # = 4.0°

# Confiança de detecção — 3 níveis decrescentes
PRIMARY_CONF  = 0.15
FALLBACK_CONF = 0.05
MIN_CONF      = 0.001

# Resolução de entrada do modelo na detecção (maior = mais detalhe
# para objetos pequenos, mas mais lento por imagem)
DETECT_IMGSZ = 960

# Fração de rotações que vai para val
VAL_FRACTION = 0.10

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
    Gera n_rotations rotações, igualmente espaçadas em 360° (passo 4°),
    seguindo o padrão de referência:

      1. Extrai o quadrado central "seguro" (lado = menor_lado / sqrt(2))
         da imagem de entrada — esse quadrado, ao ser rotacionado em
         torno do seu próprio centro, nunca expõe área fora da imagem
         original.
      2. Para cada ângulo, rotaciona ESSE quadrado já reduzido (dentro
         de suas próprias dimensões), usando BORDER_REPLICATE para
         preencher os cantos que "saem" — como o quadrado já é o
         inscrito seguro, BORDER_REPLICATE só preenche cantos vazios
         com pixels vizinhos reais, sem esticar conteúdo de fora.

    Converte para escala de cinza (3 canais BGR iguais) — robustez a
    variações de cor/iluminação; o modelo aprende forma/contraste.
    A mesma conversão deve ser aplicada na inferência (detect.py).

    Retorna lista de paths gerados.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERRO] Não foi possível ler {img_path.name}")
        return []

    orig_h, orig_w = img.shape[:2]
    shortest_side = min(orig_w, orig_h)

    safe_size = int(math.floor(shortest_side / math.sqrt(2)))
    if safe_size % 2 != 0:
        safe_size -= 1
    if safe_size < 10:
        safe_size = shortest_side

    # Extrair o quadrado central seguro da imagem original
    cx, cy = orig_w // 2, orig_h // 2
    half = safe_size // 2
    center_square = img[cy-half:cy+half, cx-half:cx+half]

    # Converter para escala de cinza (3 canais, BGR iguais) — o modelo
    # passa a focar em forma/contraste/textura, ignorando cor. A mesma
    # conversão é aplicada na inferência (detect.py).
    gray = cv2.cvtColor(center_square, cv2.COLOR_BGR2GRAY)
    center_square = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_center = (safe_size / 2.0, safe_size / 2.0)
    step = 360.0 / n_rotations

    generated = []
    for i in range(n_rotations):
        angle = i * step
        M = cv2.getRotationMatrix2D(seg_center, angle, 1.0)
        rotated = cv2.warpAffine(
            center_square, M, (safe_size, safe_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_path = out_dir / f"{img_path.stem}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated.append(out_path)

    return generated


def detect_bbox_px(model, img_path: Path, search_frac: float = 1.0):
    """
    Roda o modelo em até 3 níveis de confiança decrescentes
    (PRIMARY_CONF → FALLBACK_CONF → MIN_CONF) e retorna a primeira bbox
    que caiba INTEIRAMENTE dentro da região central de busca (search_frac
    da imagem, centrada). Isso evita que o modelo confunda os cantos/
    bordas do octógono (bboxes grandes, mesmo com centro no meio da
    imagem) com o dado (sempre pequeno e centralizado).

    Tenta os 3 níveis mesmo que o nível anterior tenha retornado boxes —
    se nenhuma delas passar pelo filtro de região, tenta o próximo nível
    de confiança (pode revelar uma detecção menor/melhor posicionada).

    search_frac = 1.0 → sem filtro (usa a imagem inteira)
    search_frac = 0.6 → só aceita bboxes totalmente contidas nos 60%
                         centrais da imagem

    Retorna (xmin,ymin,xmax,ymax) ou None se nada válido for encontrado.
    Não escreve nenhum arquivo.
    """
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    if search_frac >= 0.999:
        rx1, ry1, rx2, ry2 = 0, 0, img_w, img_h
    else:
        margin = (1 - search_frac) / 2.0
        rx1, ry1 = img_w * margin, img_h * margin
        rx2, ry2 = img_w * (1 - margin), img_h * (1 - margin)

    for conf in (PRIMARY_CONF, FALLBACK_CONF, MIN_CONF):
        results = model(str(img_path), conf=conf, imgsz=DETECT_IMGSZ, verbose=False)
        boxes   = results[0].boxes

        for box in boxes:
            xmin, ymin, xmax, ymax = map(float, box.xyxy[0].tolist())
            if rx1 <= xmin and xmax <= rx2 and ry1 <= ymin and ymax <= ry2:
                return (xmin, ymin, xmax, ymax)

    return None


class _CenterSquareSelector:
    """Seletor de quadrado central ajustável (apenas tamanho, sempre centrado)."""

    def __init__(self, img_size: int):
        self.img_size = img_size
        self.frac = 0.6  # fração inicial

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 0.02 if flags > 0 else -0.02
            self.frac = float(np.clip(self.frac + delta, 0.1, 1.0))

    @property
    def rect(self):
        side = int(self.img_size * self.frac)
        off = (self.img_size - side) // 2
        return (off, off, off + side, off + side)


def select_search_region(sample_img_path: Path) -> float:
    """
    Mostra uma rotação de exemplo e permite ajustar (scroll do mouse) um
    quadrado centrado que define a região onde o sistema vai procurar o
    dado. Retorna a fração (0.1 a 1.0) escolhida.

    Controles:
      Scroll        → aumentar/diminuir o quadrado
      ENTER / C     → confirmar
      Q             → cancelar (usa 1.0 = sem filtro)
    """
    img = cv2.imread(str(sample_img_path))
    if img is None:
        return 1.0

    size = img.shape[0]
    sel = _CenterSquareSelector(size)
    win = "Selecionar regiao de busca do dado"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sel.mouse_cb)

    print("\n" + "="*58)
    print("  REGIÃO DE BUSCA DO DADO")
    print("="*58)
    print("  Ajuste o quadrado central para cobrir a área onde o dado")
    print("  fica posicionado (evita confundir com os cantos do tray).")
    print("  SCROLL = redimensionar   ENTER = confirmar   Q = pular\n")

    result = 1.0
    while True:
        display = img.copy()
        x1, y1, x2, y2 = sel.rect
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        dark = (display * 0.35).astype(np.uint8)
        display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)

        cv2.rectangle(display, (0, 0), (size, 34), (0, 0, 0), -1)
        cv2.putText(display,
                    f"Regiao: {sel.frac*100:.0f}%  SCROLL=ajustar  ENTER=ok  Q=pular",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("c")):
            result = sel.frac
            break
        elif key == ord("q"):
            result = 1.0
            break

    cv2.destroyAllWindows()
    return result


def write_label_from_bbox_px(img_path: Path, class_id: int, bbox_px: tuple,
                             label_dir: Path, preview_dir: Path, tag: str = "OK"):
    """
    Escreve o .txt YOLO e o preview a partir de uma bbox em PIXELS
    (xmin,ymin,xmax,ymax) — sem rodar o modelo.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img_h, img_w = img.shape[:2]

    xmin, ymin, xmax, ymax = bbox_px
    xmin = max(0.0, min(xmin, img_w))
    xmax = max(0.0, min(xmax, img_w))
    ymin = max(0.0, min(ymin, img_h))
    ymax = max(0.0, min(ymax, img_h))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        xmin, ymin = img_w*.30, img_h*.30
        xmax, ymax = img_w*.70, img_h*.70
        bw, bh = xmax - xmin, ymax - ymin

    xc = (xmin + bw/2) / img_w
    yc = (ymin + bh/2) / img_h

    label_dir.mkdir(parents=True, exist_ok=True)
    with open(label_dir / (img_path.stem + ".txt"), "w") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = img.copy()
    color = (0, 200, 80) if tag == "OK" else (0, 200, 255)
    cv2.rectangle(preview, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
    cv2.putText(preview, tag, (int(xmin), max(int(ymin)-8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(preview_dir / img_path.name), preview)


# ─── Distribuição em splits ────────────────────────────────────────────────────

def copy_to_split(img_path: Path, lbl_path: Path, split: str):
    for subdir, src in [("images", img_path), ("labels", lbl_path)]:
        dst = DATASET_ROOT / subdir / split / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


def octagon_to_square(img: np.ndarray, polygon: list) -> np.ndarray:
    """
    Recorta a imagem para a bounding box do octógono (sem mascarar) e
    centraliza num CANVAS QUADRADO (lado = maior dimensão da bbox),
    preenchendo a sobra ESTICANDO as bordas reais da imagem
    (cv2.BORDER_REPLICATE) — mantém a coloração real do papel/fundo
    em vez de um branco artificial com contraste abrupto.

    Este é o tamanho FINAL/alvo (lado = bbox do octógono), igual ao que
    detect.py usa na inferência.
    """
    crop, x, y = crop_to_polygon_bbox_paper(img, polygon)
    h, w = crop.shape[:2]
    side = max(h, w)

    pad_y = side - h
    pad_x = side - w
    top    = pad_y // 2
    bottom = pad_y - top
    left   = pad_x // 2
    right  = pad_x - left

    canvas = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                borderType=cv2.BORDER_REPLICATE)
    return canvas


def octagon_to_square_oversized(img: np.ndarray, polygon: list) -> np.ndarray:
    """
    Como octagon_to_square(), mas o canvas final é AMPLIADO por
    sqrt(2) em relação ao tamanho alvo (bordas esticadas com a cor
    real do papel via BORDER_REPLICATE).

    Motivo: generate_rotations() extrai o quadrado central "seguro"
    (lado/sqrt(2)) antes de rotacionar. Se a entrada já for sqrt(2)
    vezes maior que o alvo, esse recorte resulta EXATAMENTE no tamanho
    alvo (= tamanho do octógono usado na inferência) — sem reduzir o
    dado, sem "zoom".
    """
    crop, x, y = crop_to_polygon_bbox_paper(img, polygon)
    h, w = crop.shape[:2]
    target_side = max(h, w)
    over_side   = int(round(target_side * math.sqrt(2)))

    pad_y_total = over_side - h
    pad_x_total = over_side - w
    top    = pad_y_total // 2
    bottom = pad_y_total - top
    left   = pad_x_total // 2
    right  = pad_x_total - left

    canvas = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                borderType=cv2.BORDER_REPLICATE)
    return canvas


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(images: list, class_map: dict, model, roi):
    """
    images: lista de (img_path, class_id)

    Para cada foto, seguindo o padrão de referência:
      1. Gera o canvas do octógono (612×612, cor do papel nas bordas)
      2. generate_rotations() extrai o quadrado seguro (612/√2≈432) e
         gera N_ROTATIONS rotações DENTRO dele (BORDER_REPLICATE)
      3. Para CADA rotação, roda detecção automática (igual a main.py
         de referência): conf 0.15 → fallback conf 0.05 → fallback
         usando a região calibrada (search_frac) se nada detectado
      4. Divide entre train/val por VAL_FRACTION
    """
    preview_dir  = WORK_DIR / "bbox_preview"
    fallback_log = []
    counts       = {"train": 0, "val": 0}
    n_val_per_face = max(1, int(N_ROTATIONS * VAL_FRACTION))

    print(f"\n  Cada foto gera {N_ROTATIONS} rotações "
          f"(passo {ROTATION_STEP_DEG:.1f}°)")
    print(f"  {n_val_per_face} vão para val, "
          f"{N_ROTATIONS - n_val_per_face} para train")
    print(f"  [i] Detecção automática roda em CADA rotação "
          f"(padrão de referência).\n")

    # ── Calibração da região de busca POR TIPO DE DADO ──────────────────────
    # Uma calibração para cada prefixo (d6, d8, d10, d12, d20), usando a
    # primeira foto encontrada de cada tipo como exemplo.
    search_fracs = {}
    seen_types = set()
    for img_path_i, _ in images:
        dtype = img_path_i.stem.split("_")[0]
        if dtype in seen_types:
            continue
        seen_types.add(dtype)

        img0 = cv2.imread(str(img_path_i))
        if img0 is None:
            search_fracs[dtype] = 1.0
            continue

        cropped0 = octagon_to_square_oversized(img0, roi) if roi else img0
        tmp0 = WORK_DIR / "cropped" / f"_calib_{dtype}.jpg"
        tmp0.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp0), cropped0, [cv2.IMWRITE_JPEG_QUALITY, 97])
        rot0 = generate_rotations(tmp0, WORK_DIR / f"_calib_rot_{dtype}", n_rotations=1)

        if rot0:
            print(f"  Calibrando região de busca para {dtype.upper()}...")
            search_fracs[dtype] = select_search_region(rot0[0])
        else:
            search_fracs[dtype] = 1.0

    print()
    for dtype, frac in search_fracs.items():
        print(f"  [i] {dtype.upper()}: região de busca {frac*100:.0f}% central")
    print()

    for idx, (img_path, class_id) in enumerate(images, 1):
        stem    = img_path.stem
        rot_dir = WORK_DIR / "rotations" / stem
        lbl_dir = WORK_DIR / "labels"    / stem
        dtype   = stem.split("_")[0]
        search_frac = search_fracs.get(dtype, 1.0)

        print(f"  [{idx}/{len(images)}] {stem}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [ERRO] não foi possível ler {img_path.name}")
            continue
        cropped = octagon_to_square_oversized(img, roi) if roi else img

        tmp_path = WORK_DIR / "cropped" / f"{stem}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 97])

        rotated = generate_rotations(tmp_path, rot_dir, N_ROTATIONS)
        if not rotated:
            continue

        # ── Detecção automática em CADA rotação ─────────────────────────────
        det_ok = 0
        for r in rotated:
            bbox_px = detect_bbox_px(model, r, search_frac=search_frac)
            if bbox_px is not None:
                det_ok += 1
                write_label_from_bbox_px(r, class_id, bbox_px, lbl_dir, preview_dir, tag="OK")
            else:
                img_r = cv2.imread(str(r))
                h0, w0 = img_r.shape[:2]
                m = (1 - search_frac) / 2.0
                bbox_px = (w0*m, h0*m, w0*(1-m), h0*(1-m))
                write_label_from_bbox_px(r, class_id, bbox_px, lbl_dir, preview_dir, tag="FALLBACK")

        fallback_n = len(rotated) - det_ok
        if fallback_n > 0:
            fallback_log.append(f"{stem}: {fallback_n}/{len(rotated)} fallback (região calibrada)")

        # ── Dividir entre val e train ────────────────────────────────────────
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
        print(f"\n  [AVISO] Faces com fallback:")
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
            pts = np.array(roi, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            dark = (display * 0.4).astype(np.uint8)
            display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 220, 80), thickness=2)

            # Cruz no centro do octógono — posição ideal para o dado
            xs = [p[0] for p in roi]
            ys = [p[1] for p in roi]
            roi_cx = int((min(xs) + max(xs)) / 2)
            roi_cy = int((min(ys) + max(ys)) / 2)
            cross_len = max(12, (max(xs) - min(xs)) // 12)
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
                pts = np.array(roi, dtype=np.int32)
                cv2.polylines(flash, [pts], isClosed=True, color=(60,220,60), thickness=4)
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