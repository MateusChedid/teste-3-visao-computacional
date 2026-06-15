# RPG Dice Vision v4

Detecção de resultados de dados de RPG (d6, d8, d10, d12, d20) em tempo real via webcam, usando YOLOv8.

> **d10**: o valor `0` representa **zero** (não dez).

---

## Visão geral da abordagem

O projeto evoluiu por várias iterações até chegar ao fluxo atual, resumido abaixo. Os pontos-chave que resolveram os principais problemas de detecção:

- **Uma única ROI**: um **octógono** desenhado sobre o dice tray (`utils/roi_inference.py`), usado tanto na coleta do dataset quanto na inferência em tempo real. Antes existiam duas ROIs separadas (quadrado de coleta + octógono de inferência), o que causava descasamento de escala entre treino e uso real — o dado aparecia muito menor na inferência do que no treino. Agora ambos usam exatamente o mesmo recorte.
- **Bordas com a cor real do papel**: a área fora do octógono (mas dentro da bbox que o envolve) é preenchida esticando (`cv2.BORDER_REPLICATE`) a cor real do papel/fundo do tray — amostrada automaticamente perto da borda interna do octógono. Isso evita o contraste artificial de bordas pretas/brancas que confundia o detector.
- **Canvas no tamanho real do tray**: o canvas final do dataset tem o mesmo tamanho da bbox do octógono (sem "zoom"/redução) — o dado aparece no dataset na mesma proporção que aparece na inferência ao vivo.
- **90 rotações por face** (passo de 4°), geradas a partir de um quadrado central "seguro" extraído do canvas — evita qualquer pixel replicado/distorcido ao girar.
- **Detecção automática de bbox em cada rotação**, com 3 níveis de confiança decrescentes (0.15 → 0.05 → 0.001) e `imgsz=960`, restrita a uma **região central calibrada por tipo de dado** — isso evita que o YOLOv8 genérico confunda os cantos/bordas do octógono com o dado. Quando falha, usa um fallback no tamanho/posição calibrados (não mais uma caixa genérica).
- **Sem ajuste de câmera**: a câmera roda na configuração padrão do sistema. O fundo branco do tray garante boa exposição automática.
- **Imagens RGB originais** — sem conversão para escala de cinza nem augmentation de exposição na coleta (o `config.yaml` já aplica HSV/brilho via augmentation online do YOLOv8 durante o treino).

---

## Estrutura

```
rpg_dice_cv_v3/
├── dataset_collector/
│   ├── auto_collect.py      ← coleta: 1 foto/face → 90 rotações + bbox automática
│   ├── validate_dataset.py
│   └── input/                ← fotos originais (d6_1.jpg, d20_17.jpg, d10_0.jpg…)
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
│   ├── roi_inference.py       ← octógono (única ROI: coleta + inferência)
│   └── roi_collect.py         ← legado, não usado no fluxo atual
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

### 1. Capturar as fotos originais

Tire 1 foto por face (57 no total), com o **tray inteiro visível** e o dado **centralizado**, sempre na mesma posição/distância de câmera. Salve em `dataset_collector/input/` com o padrão `tipo_face.jpg` (ex: `d6_1.jpg`, `d20_17.jpg`, `d10_0.jpg`).

### 2. Coletar e gerar o dataset

```bash
python dataset_collector/auto_collect.py --from-folder
```

Na primeira execução, será solicitado:

- **Octógono do tray**: clique nos cantos do tray em ordem (mínimo 3 pontos), `ENTER` duas vezes para confirmar e salvar.
- **Calibração da região de busca por tipo de dado**: para cada tipo (d6, d8, d10, d12, d20), uma janela mostra uma rotação de exemplo com um quadrado central ajustável (scroll do mouse). Ajuste para cobrir o tamanho real daquele dado e confirme com `ENTER`. Essa região é usada tanto para filtrar detecções automáticas quanto como bbox de fallback.

O programa então gera, para cada face: o canvas a partir do octógono → 90 rotações → bbox automática (ou fallback calibrado) → divisão em train/val.

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

### 5. Inferência em tempo real

```bash
python inference/detect.py                        # webcam (índice 0)
python inference/detect.py --source 1               # outra webcam
python inference/detect.py --weights caminho/best.pt
python inference/detect.py --source foto.jpg        # imagem estática
python inference/detect.py --no-roi                 # ignora octógono, frame inteiro
```

O octógono salvo é usado automaticamente — o crop enviado ao modelo é gerado com a mesma transformação (`octagon_to_square`) usada na coleta, garantindo que o dado apareça na mesma proporção vista no treino.

---

## Controles da inferência

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Congelar frame e mostrar resultado |
| `R` | Voltar ao live feed |
| `S` | Salvar screenshot (e o canvas exato enviado ao modelo, para debug) |
| `Q` | Sair |

---

## Redefinir o octógono

Para redesenhar a área do tray (ex: mudou a posição da câmera):

```bash
python utils/roi_inference.py
python utils/roi_inference.py --camera 1   # outra webcam
```

---

## Solução de problemas

- **CUDA out of memory / CUDA error: unknown error**: geralmente indica driver NVIDIA em estado inconsistente após um travamento. Reinicie o PC. Se persistir, force CPU com `$env:CUDA_VISIBLE_DEVICES="-1"` antes de rodar.
- **Detecção ruim mesmo com o dado bem posicionado**: confirme que o octógono foi desenhado sobre o **tray inteiro** (não uma área menor), e que o dado fica centralizado dentro dele tanto na coleta quanto no uso real.
- **Muito fallback na coleta**: normal para dados pequenos como o d10. A região de busca calibrada por tipo já produz uma bbox de fallback no tamanho/posição corretos, então um fallback alto não é necessariamente um problema — confira os previews em `dataset_collector/_work/bbox_preview/`.
