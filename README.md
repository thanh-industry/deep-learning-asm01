# Deep Learning Assignment 1 — CO5085
**Deep Learning and Applications in Computer Vision | Semester 2, 2025–2026**

> **Instructor:** Le Thanh Sach | **Deadline:** 23:59, March 25 2026

---

## Project Overview

Compare deep learning model families across **three data modalities**:

| Folder | Task | Models Compared | Default Dataset |
|--------|------|-----------------|----------------|
| [`CNN-vs-ViT/`](CNN-vs-ViT/) | Image Classification | ResNet50 vs ViT-B/16 | CIFAR-100 (100 classes, 60K images) |
| [`RNN-vs-Transformer/`](RNN-vs-Transformer/) | Text Classification | BiLSTM vs DistilBERT | 20 Newsgroups (20 classes, ~18K docs) |
| [`ZeroShot-vs-FewShot/`](ZeroShot-vs-FewShot/) | Multimodal Classification | CLIP Zero-shot vs Few-shot | MS COCO 2014 (12 supercategories, ~124K images + captions, train/val/test splits) |

---

## Repository Structure

```
deep-learning-asm01/
├── CNN-vs-ViT/
│   └── cnn_vs_vit.ipynb            # ResNet50 vs ViT-B/16 on CIFAR-100
├── RNN-vs-Transformer/
│   └── rnn_vs_transformer.ipynb    # BiLSTM vs DistilBERT on 20 Newsgroups
├── ZeroShot-vs-FewShot/
│   └── zeroshot_vs_fewshot.ipynb   # CLIP zero-shot vs few-shot on MS COCO 2014
├── assignment1-vne.pdf
└── README.md
```

---

## How to Run (Google Colab)

### Shared Google Drive Setup (recommended — one download, shared by all team members)

All three notebooks save datasets and results to the **same shared Google Drive folder**.
The first person to run each notebook downloads the dataset; everyone else loads it from the shared Drive.

**Step 1** — Add the shared folder shortcut to your Drive:
> Open: https://drive.google.com/drive/folders/1UVs9LM9N7H_R_cKbr6pNjAGKEgqHzgUp
> Click the folder → **Add shortcut to Drive** → place in **My Drive** → name it `deep-learning-asm01`

**Step 2** — Run any notebook normally (**Runtime → Run all**). The notebook will:
- Mount Google Drive and detect the shared folder automatically
- Load the dataset from the shared Drive if already downloaded, or download it there
- Save results, plots, and model checkpoints to the shared Drive

**Shared Drive folder structure after all notebooks have run:**

```
MyDrive/deep-learning-asm01/
├── data/
│   ├── cifar100/                             # CIFAR-100 (~170 MB) — auto-downloaded by CNN notebook
│   │   └── cifar-100-python/
│   ├── 20newsgroups/                         # 20 Newsgroups cache — auto-created by RNN notebook
│   │   └── 20newsgroups_cache.pkl
│   └── coco2014/                             # MS COCO 2014 — auto-downloaded by CLIP notebook
│       ├── val2014/                          # ~41K JPEG images (~6 GB)  [required]
│       ├── train2014/                        # ~83K JPEG images (~13 GB) [optional]
│       └── annotations/
│           ├── instances_val2014.json
│           ├── captions_val2014.json
│           ├── instances_train2014.json      # (only if DOWNLOAD_TRAIN2014=True)
│           └── captions_train2014.json
└── results/
    ├── cnn_vs_vit/                           # CNN notebook outputs
    ├── rnn_vs_transformer/                   # RNN notebook outputs
    └── zeroshot_vs_fewshot/                  # CLIP notebook outputs
```

If the shared Drive shortcut is not set up, each notebook falls back to `/content` (local, not persisted between sessions) and prints a setup reminder.

### Runtime Settings

1. Open the desired notebook in [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → GPU (T4)**
3. Optionally edit the **Configuration cell** at the top:
   - `GDRIVE_DATASET_PATH`: path to a custom dataset on Drive, or `None` to use the default
   - `DOWNLOAD_TRAIN2014` *(CLIP notebook only)*: `True` to also download the larger `train2014` split (~13 GB)
4. **Runtime → Run all**

---

## Technical Requirements Met

### Core (60%)
- [x] **Image**: ResNet50 (CNN) vs ViT-B/16 (Vision Transformer) — pretrained ImageNet weights, fine-tuned on CIFAR-100
- [x] **Text**: BiLSTM (RNN, trained from scratch) vs DistilBERT (Transformer, pretrained) — on 20 Newsgroups
- [x] **Multimodal**: CLIP Zero-shot vs Few-shot — CLIP ViT-B/32 on MS COCO 2014 with real image–text pairs, three-way train/val/test split
- [x] **Metrics**: Accuracy, F1-macro, Confusion Matrix, Classification Report (per-class precision/recall/F1)

### Extensions (40%)
- [x] **Interpretability (Grad-CAM)** — visualizes image regions driving CNN predictions; included for ResNet50
- [x] **Error analysis** — superclass-level (20×20) and fine-grained (100×100) confusion matrices; per-class F1 breakdown
- [x] **Multimodal fusion** — systematic comparison of image-only, caption-only, and image+caption zero-shot CLIP variants
- [x] **Few-shot learning curve** — accuracy vs number of shots (1 → 20), compared against zero-shot baseline
- [x] **Training curves** — loss, accuracy, and F1-macro per epoch for all trained models (ResNet50, ViT-B/16, BiLSTM, DistilBERT)
- [x] **Data augmentation** — RandomHorizontalFlip, ColorJitter, RandomRotation for image models; label smoothing regularization

---

## Datasets

### CIFAR-100 (Image Classification)
- **100 classes** (fine-grained) across 20 superclasses (animals, vehicles, household items, etc.)
- 50,000 train / 10,000 test — 32×32 RGB (resized to 224×224 for pretrained models)
- Auto-downloaded via `torchvision.datasets.CIFAR100` to the shared Drive (~170 MB)
- Shared: teammates load from `MyDrive/deep-learning-asm01/data/cifar100/` without re-downloading

### 20 Newsgroups (Text Classification)
- 20 newsgroup topics (alt.atheism, sci.space, rec.sport.hockey, talk.politics.guns, ...)
- 11,314 train / 7,532 test documents
- Auto-downloaded via `sklearn.datasets.fetch_20newsgroups`, then pickled to the shared Drive
- Shared: subsequent runs load from `MyDrive/deep-learning-asm01/data/20newsgroups/20newsgroups_cache.pkl`

### MS COCO 2014 (Multimodal — CLIP)
- **Real image–text pairs**: each image has **5 human-written captions** describing its content
- **12 COCO supercategories** as classification labels: person, vehicle, outdoor, animal, accessory, sports, kitchen, food, furniture, electronic, appliance, indoor
- **val2014**: ~41K images — split into train\_pool / val / test (default 40/30/30)
- **train2014**: ~83K images (optional, for a larger few-shot support pool)
- Zero-shot uses CLIP ViT-B/32 with a 5-prompt ensemble; few-shot uses logistic regression linear probe
- **Three dataset splits**: train\_pool (few-shot support) → val (tuning) → test (final evaluation)
- **Three zero-shot variants compared**:
  1. Image-only: CLIP image features vs class text prompts
  2. Caption-only: CLIP caption features vs class text prompts
  3. Fusion: Avg(image + caption) features — true multimodal CLIP
- **Download sources** (auto-handled by the notebook):
  - Images: http://images.cocodataset.org/zips/val2014.zip (~6 GB)
  - Annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip

---

## Actual Results

Results are from a single full run on Google Colab. All metrics are reported on the held-out **test** split.

### Image Classification — CIFAR-100 (GPU: NVIDIA A100-SXM4-80GB)

| Model | Top-1 Acc | Top-5 Acc | F1-macro | Params | Train Time |
|-------|-----------|-----------|----------|--------|------------|
| ResNet50 (CNN) | **82.95%** | 97.38% | 0.8293 | 24.6M | 34.5 min |
| ViT-B/16 | **91.28%** | 98.26% | 0.9127 | 85.9M | 112.8 min |

ViT-B/16 outperforms ResNet50 by **+8.33 pp** in Top-1 accuracy, at the cost of 3.5× more parameters and ~3.3× longer training.

### Text Classification — 20 Newsgroups (GPU: NVIDIA RTX PRO 6000 Blackwell)

| Model | Accuracy | F1-macro | Params | Train Time |
|-------|----------|----------|--------|------------|
| BiLSTM (RNN) | **46.85%** | 0.4563 | 7.45M | 1.1 min |
| DistilBERT | **69.70%** | 0.6802 | 67.0M | 2.0 min |

DistilBERT outperforms BiLSTM by **+22.85 pp** in accuracy. BiLSTM trained from scratch on 11K samples is underfitted; DistilBERT's pretrained representations generalize substantially better.

### Multimodal Classification — MS COCO 2014 (GPU: NVIDIA A100-SXM4-80GB)

**Dataset splits (Option B — train2014 + val2014):** train\_pool = 82,081 | val = 20,068 | test = 20,069 images

**Zero-shot (test split):**

| Variant | Accuracy | F1-macro |
|---------|----------|----------|
| Image-only | 35.03% | 0.3181 |
| Caption-only | 45.58% | 0.3368 |
| Fusion (image + caption) | 43.97% | 0.3616 |

**Few-shot (test split, logistic regression linear probe on CLIP features):**

| Shots | Accuracy | Std |
|-------|----------|-----|
| 1 | 30.41% | ±3.43 pp |
| 5 | 40.73% | ±2.91 pp |
| 10 | 45.27% | ±0.53 pp |
| 20 | **46.98%** | ±1.24 pp |

Caption-only zero-shot (45.58%) matches 10-shot performance. 20-shot few-shot reaches 46.98%, slightly above the zero-shot fusion baseline. The 12-class COCO supercategory task is inherently ambiguous (multi-label images assigned a single dominant label), which suppresses overall accuracy.

*CIFAR-100 accuracy is lower than CIFAR-10 due to 100 fine-grained classes with only 500 training images per class.*

---

## Dependencies

```
torch >= 2.0
torchvision >= 0.15
transformers >= 4.35
timm >= 0.9
open-clip-torch >= 2.20
pycocotools >= 2.0
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
pandas >= 2.0
tqdm >= 4.65
pytorch-grad-cam >= 1.4
Pillow >= 9.0
```

All packages are either pre-installed in Colab or installed via `!pip install` in the first cell of each notebook.

---

*CO5085 — Deep Learning and Applications in Computer Vision | HCMUT*
