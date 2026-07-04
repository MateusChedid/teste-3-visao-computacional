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

N_ROTATIONS  = 45                       # rotações por foto (passo 4°, igual ref.)
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

    Mantém as cores originais — sem alteração de exposição/cor.

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

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_center = (safe_size / 2.0, safe_size / 2.0)
    step = 360.0 / n_rotations

    generated = []
    matrices  = []
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
        matrices.append(M)

    return generated, matrices


def rotate_rect(rect, M, img_size: int):
    """
    Projeta um retângulo (x1,y1,x2,y2) pela matriz de rotação M
    (2×3, de getRotationMatrix2D) e retorna a nova bounding box
    axis-aligned, clipada ao tamanho da imagem.
    """
    x1, y1, x2, y2 = rect
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float64)
    ones = np.ones((4, 1))
    ch = np.hstack([corners, ones])      # (4,3)
    t  = (M @ ch.T).T                   # (4,2)
    nx1, ny1 = t.min(axis=0)
    nx2, ny2 = t.max(axis=0)
    nx1 = max(0.0, nx1); ny1 = max(0.0, ny1)
    nx2 = min(float(img_size), nx2); ny2 = min(float(img_size), ny2)
    return (nx1, ny1, nx2, ny2)


def detect_bbox_px(model, img_path: Path, search_rect=None):
    """
    Roda o modelo em até 3 níveis de confiança decrescentes e retorna a
    primeira bbox que caiba INTEIRAMENTE dentro de search_rect
    (x1,y1,x2,y2) em pixels. Se search_rect=None, aceita qualquer bbox.

    Retorna (xmin,ymin,xmax,ymax) ou None.
    """
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    rx1, ry1 = 0, 0
    rx2, ry2 = img_w, img_h
    if search_rect is not None:
        rx1, ry1, rx2, ry2 = search_rect

    for conf in (PRIMARY_CONF, FALLBACK_CONF, MIN_CONF):
        results = model(str(img_path), conf=conf, imgsz=DETECT_IMGSZ, verbose=False)
        boxes   = results[0].boxes

        for box in boxes:
            xmin, ymin, xmax, ymax = map(float, box.xyxy[0].tolist())
            if rx1 <= xmin and xmax <= rx2 and ry1 <= ymin and ymax <= ry2:
                return (xmin, ymin, xmax, ymax)

    return None


class _RectSelector:
    """
    Seletor de retângulo livre: arrastar = mover, scroll = redimensionar.
    Mantém a proporção quadrada para consistência com a detecção.
    """

    def __init__(self, img_size: int, init_frac: float = 0.5):
        self.size = img_size
        side = int(img_size * init_frac)
        cx, cy = img_size // 2, img_size // 2
        self.x1 = cx - side // 2
        self.y1 = cy - side // 2
        self.x2 = cx + side // 2
        self.y2 = cy + side // 2
        self._drag = False
        self._drag_ox = 0
        self._drag_oy = 0

    def mouse_cb(self, event, x, y, flags, param):
        side = self.x2 - self.x1

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = True
            self._drag_ox = x - self.x1
            self._drag_oy = y - self.y1

        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            nx1 = x - self._drag_ox
            ny1 = y - self._drag_oy
            nx1 = int(np.clip(nx1, 0, self.size - side))
            ny1 = int(np.clip(ny1, 0, self.size - side))
            self.x1, self.y1 = nx1, ny1
            self.x2, self.y2 = nx1 + side, ny1 + side

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            cx = (self.x1 + self.x2) // 2
            cy = (self.y1 + self.y2) // 2
            delta = 8 if flags > 0 else -8
            side = int(np.clip(side + delta, 10, self.size))
            self.x1 = int(np.clip(cx - side // 2, 0, self.size - side))
            self.y1 = int(np.clip(cy - side // 2, 0, self.size - side))
            self.x2 = self.x1 + side
            self.y2 = self.y1 + side

    @property
    def rect(self):
        return (self.x1, self.y1, self.x2, self.y2)


def select_search_region(sample_img_path: Path) -> tuple | None:
    """
    Mostra uma rotação de exemplo e permite posicionar e redimensionar
    um retângulo que define onde o sistema vai procurar o dado.

    Retorna (x1, y1, x2, y2) em pixels da imagem de rotação,
    ou None se pulado (sem filtro).

    Controles:
      Arrastar      → mover a região
      Scroll        → redimensionar
      ENTER / C     → confirmar
      Q             → pular (sem filtro para este tipo/posição)
    """
    img = cv2.imread(str(sample_img_path))
    if img is None:
        return None

    size = img.shape[0]
    sel  = _RectSelector(size, init_frac=0.45)
    win  = "Selecionar regiao de busca do dado"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sel.mouse_cb)

    print("\n" + "="*58)
    print("  REGIÃO DE BUSCA DO DADO")
    print("="*58)
    print("  Posicione o retângulo sobre o dado.")
    print("  ARRASTAR = mover   SCROLL = redimensionar")
    print("  ENTER = confirmar   Q = pular (sem filtro)\n")

    result = None
    while True:
        display = img.copy()
        x1, y1, x2, y2 = sel.rect
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        dark = (display * 0.35).astype(np.uint8)
        display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)

        side = x2 - x1
        cv2.rectangle(display, (0, 0), (size, 34), (0, 0, 0), -1)
        cv2.putText(display,
                    f"Regiao: {side}x{side}px em ({x1},{y1})  "
                    f"ARRASTAR=mover  SCROLL=zoom  ENTER=ok  Q=pular",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("c")):
            result = sel.rect
            break
        elif key == ord("q"):
            result = None
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

    # ── Calibração da região de busca POR TIPO + POSIÇÃO ─────────────────────
    # Chave: (dtype, sufixo)  ex: ("d10", ""), ("d10", "_a"), ("d6", "_b") ...
    # Usa a primeira foto encontrada de cada combinação como exemplo.
    search_fracs = {}   # (dtype, suffix) → float
    seen_keys = set()

    for img_path_i, _ in images:
        stem  = img_path_i.stem          # ex: "d10_2_a"
        parts = stem.split("_")
        dtype = parts[0]                 # "d10"
        # sufixo de posição: tudo depois de dtype_face
        # stem = dtype_face[_suffix]  → suffix = "_a", "_b", ... ou ""
        # parts[0]=dtype, parts[1]=face, parts[2:]=sufixo (pode ser vazio)
        pos_suffix = ("_" + "_".join(parts[2:])) if len(parts) > 2 else ""
        key = (dtype, pos_suffix)

        if key in seen_keys:
            continue
        seen_keys.add(key)

        img0 = cv2.imread(str(img_path_i))
        if img0 is None:
            search_fracs[key] = 1.0
            continue

        cropped0 = octagon_to_square_oversized(img0, roi) if roi else img0
        calib_id = f"_calib_{dtype}{pos_suffix}"
        tmp0 = WORK_DIR / "cropped" / f"{calib_id}.jpg"
        tmp0.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp0), cropped0, [cv2.IMWRITE_JPEG_QUALITY, 97])
        rot0, _ = generate_rotations(tmp0, WORK_DIR / f"_calib_rot{calib_id}", n_rotations=1)

        pos_label = {
            "":   "CENTRO",
            "_a": "NO",
            "_b": "NE",
            "_c": "SO",
            "_d": "SE",
        }.get(pos_suffix, pos_suffix)

        if rot0:
            print(f"  Calibrando região de busca para {dtype.upper()} — {pos_label}...")
            search_fracs[key] = select_search_region(rot0[0])
        else:
            search_fracs[key] = None

    print()
    for (dtype, suf), rect in search_fracs.items():
        pos_label = {"": "CENTRO", "_a": "NO", "_b": "NE", "_c": "SO", "_d": "SE"}.get(suf, suf)
        if rect:
            x1,y1,x2,y2 = rect
            print(f"  [i] {dtype.upper()} {pos_label}: região {x2-x1}x{y2-y1}px em ({x1},{y1})")
        else:
            print(f"  [i] {dtype.upper()} {pos_label}: sem filtro")
    print()

    for idx, (img_path, class_id) in enumerate(images, 1):
        stem  = img_path.stem
        parts = stem.split("_")
        dtype = parts[0]
        pos_suffix = ("_" + "_".join(parts[2:])) if len(parts) > 2 else ""
        search_rect = search_fracs.get((dtype, pos_suffix),
                      search_fracs.get((dtype, ""), None))  # fallback para centro do mesmo tipo

        rot_dir = WORK_DIR / "rotations" / stem
        lbl_dir = WORK_DIR / "labels"    / stem

        print(f"  [{idx}/{len(images)}] {stem}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [ERRO] não foi possível ler {img_path.name}")
            continue
        cropped = octagon_to_square_oversized(img, roi) if roi else img

        tmp_path = WORK_DIR / "cropped" / f"{stem}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 97])

        rotated, matrices = generate_rotations(tmp_path, rot_dir, N_ROTATIONS)
        if not rotated:
            continue

        # safe_size: lado do quadrado de cada rotação
        safe_img = cv2.imread(str(rotated[0]))
        safe_size = safe_img.shape[0] if safe_img is not None else 640

        # ── Detecção automática em CADA rotação ─────────────────────────────
        det_ok = 0
        for r, M in zip(rotated, matrices):
            # Projetar search_rect para o ângulo desta rotação
            rotated_rect = rotate_rect(search_rect, M, safe_size) \
                           if search_rect is not None else None

            bbox_px = detect_bbox_px(model, r, search_rect=rotated_rect)
            if bbox_px is not None:
                det_ok += 1
                write_label_from_bbox_px(r, class_id, bbox_px, lbl_dir, preview_dir, tag="OK")
            else:
                img_r = cv2.imread(str(r))
                h0, w0 = img_r.shape[:2]
                if rotated_rect is not None:
                    # fallback: usar a região rotacionada calibrada
                    bbox_px = tuple(float(v) for v in rotated_rect)
                else:
                    # sem calibração: caixa central de 40%
                    bbox_px = (w0*0.30, h0*0.30, w0*0.70, h0*0.70)
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

    # ── Relatório de bboxes por tamanho (menor → maior) ───────────────────────
    # Lê todos os .txt gerados e calcula a área relativa de cada bbox.
    # Útil para identificar bboxes anormalmente pequenas (detecção errada)
    # e removê-las manualmente antes de treinar.
    print(f"\n  {'─'*56}")
    print(f"  RELATÓRIO DE BBOXES — ordenadas por tamanho (menor→maior)")
    print(f"  {'─'*56}")
    print(f"  {'Face':<20} {'Rotações':<10} {'Área média':>10}  {'Menor':>8}  {'Maior':>8}")
    print(f"  {'─'*56}")

    face_stats = []
    lbl_base = WORK_DIR / "labels"
    for face_lbl_dir in sorted(lbl_base.iterdir()):
        if not face_lbl_dir.is_dir():
            continue
        areas = []
        for txt in face_lbl_dir.glob("*.txt"):
            try:
                line = txt.read_text().strip().split()
                if len(line) >= 5:
                    bw, bh = float(line[3]), float(line[4])
                    areas.append(bw * bh)
            except Exception:
                pass
        if areas:
            face_stats.append((
                face_lbl_dir.name,
                len(areas),
                sum(areas)/len(areas),
                min(areas),
                max(areas),
            ))

    # Ordenar por área média (menor primeiro)
    face_stats.sort(key=lambda x: x[2])

    for name, n, avg, mn, mx in face_stats:
        flag = "  ⚠" if mn < 0.002 else ""  # sinaliza bboxes suspeitas (< 0.2% da imagem)
        print(f"  {name:<20} {n:<10} {avg*100:>9.2f}%  {mn*100:>7.2f}%  {mx*100:>7.2f}%{flag}")

    n_suspicious = sum(1 for *_, mn, _ in face_stats if mn < 0.002)
    if n_suspicious:
        print(f"\n  ⚠  {n_suspicious} face(s) com bboxes menores que 0.2% da imagem.")
        print(f"     Verifique os previews em: {preview_dir}")
        print(f"     Para remover labels inválidos: delete o .txt e o .jpg correspondentes")
        print(f"     em dataset/images/train/ e dataset/labels/train/")

    print(f"\n  Próximo passo: python dataset_collector/validate_dataset.py\n")



# ─── Webcam capture ────────────────────────────────────────────────────────────

# Posições de captura dentro do octógono, como fração da bbox (fx, fy)
# Cada posição tem: chave do arquivo, label para display, fração x, fração y, cor BGR do crosshair
CAPTURE_POSITIONS = [
    ("",   "CENTRO",   0.50, 0.50, (0,   0,   255)),  # vermelho
    ("_a", "NO",       0.27, 0.27, (0,   200, 255)),  # amarelo
    ("_b", "NE",       0.73, 0.27, (0,   200, 255)),
    ("_c", "SO",       0.27, 0.73, (0,   200, 255)),
    ("_d", "SE",       0.73, 0.73, (0,   200, 255)),
]


def _roi_target_point(roi, fx: float, fy: float):
    """Calcula um ponto dentro do octógono a (fx, fy) da bbox."""
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    x = int(min(xs) + (max(xs) - min(xs)) * fx)
    y = int(min(ys) + (max(ys) - min(ys)) * fy)
    return x, y


def _draw_crosshair(display, cx, cy, color, size=22, thickness=2):
    """Desenha crosshair (cruz + círculo) na posição alvo."""
    cv2.line(display, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(display, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(display, (cx, cy), 6, color, -1)
    cv2.circle(display, (cx, cy), size + 4, color, 1)


def webcam_capture_face(dice_type: str, face: int, camera_index: int, roi) -> list[Path]:
    """
    Captura 5 fotos de uma face:
      - foto central  →  d6_1.jpg
      - canto NO      →  d6_1_a.jpg
      - canto NE      →  d6_1_b.jpg
      - canto SO      →  d6_1_c.jpg
      - canto SE      →  d6_1_d.jpg

    Para cada posição, exibe na tela um crosshair indicando onde posicionar o dado.
    Controles: ESPAÇO = capturar  |  Q = pular esta posição  |  ESC = cancelar face inteira
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERRO] Câmera {camera_index} não disponível.")
        return []

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    face_str = str(face)
    saved_paths = []

    cv2.namedWindow("Captura", cv2.WINDOW_NORMAL)

    for suffix, pos_label, fx, fy, cross_color in CAPTURE_POSITIONS:
        fname    = f"{dice_type}_{face_str}{suffix}.jpg"
        out_path = INPUT_DIR / fname

        # Pular se já existe
        if out_path.exists():
            print(f"  [✓] {fname} já existe — pulando")
            saved_paths.append(out_path)
            continue

        pos_num  = len(saved_paths) + 1
        print(f"\n  {dice_type.upper()} face {face_str}  [{pos_num}/5]  → posicione o dado no {pos_label}")

        captured = False
        skipped  = False

        while not captured and not skipped:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # Overlay do octógono
            if roi:
                pts = np.array(roi, dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                dark = (display * 0.4).astype(np.uint8)
                display = np.where(np.stack([mask]*3, axis=2) > 0, display, dark)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 220, 80), thickness=2)
                tx, ty = _roi_target_point(roi, fx, fy)
            else:
                xs = [0, w]; ys = [0, h]
                tx = int(w * fx)
                ty = int(h * fy)

            # Crosshair na posição alvo
            _draw_crosshair(display, tx, ty, cross_color)

            # Label da posição
            cv2.putText(display, pos_label, (tx + 16, ty - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cross_color, 2)

            # HUD top
            cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(display,
                        f"{dice_type.upper()} face {face_str}  [{pos_num}/5: {pos_label}]"
                        f"  ESPACO=capturar  Q=pular  ESC=cancelar face",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

            cv2.imshow("Captura", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_paths.append(out_path)
                # Flash de confirmação
                flash = frame.copy()
                if roi:
                    cv2.polylines(flash, [np.array(roi, dtype=np.int32)],
                                  isClosed=True, color=(60, 220, 60), thickness=4)
                cv2.putText(flash, f"SALVO! {pos_label}", (w//2 - 120, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (60, 220, 60), 3)
                cv2.imshow("Captura", flash)
                cv2.waitKey(500)
                captured = True
                print(f"  [✓] {fname} salvo")

            elif key == ord("q"):
                skipped = True
                print(f"  [—] {pos_label} pulado")

            elif key == 27:  # ESC
                print(f"\n  [x] Face {face_str} cancelada.")
                cap.release()
                cv2.destroyAllWindows()
                return saved_paths

    cap.release()
    cv2.destroyAllWindows()
    return saved_paths



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
        photos = webcam_capture_face(args.dice, args.face, args.camera, roi)
        if not photos:
            return
        key = f"{args.dice}_{args.face}"
        cid = class_map.get(key)
        if cid is None:
            print(f"[ERRO] Classe '{key}' não encontrada.")
            return
        run_pipeline([(p, cid) for p in photos], class_map, model, roi)
        return

    # ── Modo interativo completo ─────────────────────────────────────────────
    print("\n" + "="*58)
    print("  RPG DICE AUTO COLLECT v3")
    print("="*58)
    print(f"  5 fotos por face (centro + NO + NE + SO + SE)")
    print(f"  → {N_ROTATIONS} rotações automáticas por foto\n")

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

            # Verificar quais das 5 fotos já existem
            existing = []
            for suffix, _, _, _, _ in CAPTURE_POSITIONS:
                p = INPUT_DIR / f"{dice_type}_{face}{suffix}.jpg"
                if p.exists():
                    existing.append(p)

            if len(existing) == len(CAPTURE_POSITIONS):
                print(f"  [✓] {key} — todas as {len(CAPTURE_POSITIONS)} fotos já existem")
                for p in existing:
                    images.append((p, cid))
                continue

            # Capturar as fotos que ainda faltam
            photos = webcam_capture_face(dice_type, face, args.camera, roi)
            for p in photos:
                images.append((p, cid))

    if not images:
        print("\n[INFO] Nenhuma imagem capturada.")
        return

    print(f"\n[INFO] {len(images)} fotos. Iniciando pipeline...\n")
    run_pipeline(images, class_map, model, roi)


if __name__ == "__main__":
    main()