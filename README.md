# RPG Dice Vision v3

Detecção de dados de RPG (d6, d8, d10, d12, d20) via YOLOv8.

> **d10**: o valor `0` representa **zero** (não dez).

---

## O que mudou na v3

- **Sem configuração de câmera**: a câmera usa o padrão do sistema. O fundo branco do dice tray garante boa exposição automática.
- **Sem conversão para P&B e sem augmentation de exposição**: o dataset é treinado exatamente com as imagens capturadas, sem alterações.
- **ROI quadrada na coleta** (`utils/roi_collect.py`): você ajusta um quadrado sobre o tray. Sendo quadrada, a rotação de 360° não distorce o dado.
- **Octógono na inferência** (`utils/roi_inference.py`): apenas para uso em tempo real, você desenha o contorno exato do tray (octogonal) para o programa focar só ali.
- **1 foto por face → 80 rotações** (passo de 4.5°), geradas sem qualquer alteração de cor/brilho.

---

## Estrutura

```
rpg_dice_cv_v3/
├── dataset_collector/
│   ├── auto_collect.py      ← coleta: 1 foto/face → 80 rotações + bbox automática
│   ├── validate_dataset.py
│   └── input/                ← fotos para modo --from-folder (d6_1.jpg, d20_17.jpg…)
├── dataset/
│   ├── dataset.yaml          ← 57 classes
│   └── images/ labels/       ← train / val / test
├── training/
│   ├── train.py
│   └── config.yaml
├── inference/
│   ├── detect.py              ← inferência em tempo real
│   └── result_reader.py
├── utils/
│   ├── roi_collect.py         ← ROI QUADRADA p/ coleta
│   └── roi_inference.py       ← octógono p/ inferência
├── fix_boxes.py
└── requirements.txt
```

---

## Dados suportados

| Dado | Faces   | Classes |
|------|---------|---------|
| d6   | 1–6     | 6       |
| d8   | 1–8     | 8       |
| d10  | **0**–9 | 10      |
| d12  | 1–12    | 12      |
| d20  | 1–20    | 20      |
| —    | unknown | 1       |
| **Total** |    | **57**  |

---

## Instalação

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Fluxo de uso

### 1. Definir a área quadrada do tray (coleta)
```bash
python utils/roi_collect.py
```
Arraste o quadrado sobre o tray. Use os cantos para redimensionar (mantém proporção quadrada). `ENTER` confirma e salva.

### 2. Coletar dataset
```bash
python dataset_collector/auto_collect.py
```
1 foto por face → 80 rotações automáticas com bbox.

Modo pasta (fotos já tiradas, nomeadas `d6_1.jpg`, `d20_17.jpg`, `d10_0.jpg`...):
```bash
python dataset_collector/auto_collect.py --from-folder
```

### 3. Validar
```bash
python dataset_collector/validate_dataset.py
python dataset_collector/validate_dataset.py --fix-split
```
Se houver bboxes inválidas:
```bash
python fix_boxes.py --apply
```

### 4. Treinar
```bash
python training/train.py
```

### 5. (Opcional) Definir octógono para inferência
```bash
python utils/roi_inference.py
```
Clique nos 8 cantos do tray em ordem. Usado apenas pelo `detect.py`.

### 6. Inferência
```bash
python inference/detect.py                    # webcam, usa octógono se definido
python inference/detect.py --no-roi            # ignora octógono, frame inteiro
python inference/detect.py --source foto.jpg   # imagem estática
```

---

## Controles da inferência

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Congelar frame e mostrar resultado |
| `R` | Voltar ao live feed |
| `S` | Salvar screenshot |
| `Q` | Sair |
