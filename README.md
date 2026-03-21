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

1. Open the desired notebook in [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → GPU (T4)**
3. Edit the **Configuration cell** at the top:
   - `GDRIVE_BASE`: your Google Drive base path
   - `GDRIVE_DATASET_PATH`: path to custom dataset, or `None` to auto-download
4. **Runtime → Run all** — the notebook will:
   - Mount Google Drive automatically
   - Download the dataset if not found on Drive
   - Train all models and log progress
   - Save results, plots, and model checkpoints to Google Drive

### COCO 2014 on Shared Google Drive (recommended — shared by all team members)

The notebook uses a shared Google Drive folder so the dataset is downloaded once and reused by everyone.

**Step 1** — Add the shared folder shortcut to your Drive:
> Open: https://drive.google.com/drive/folders/1UVs9LM9N7H_R_cKbr6pNjAGKEgqHzgUp
> Click the folder → **Add shortcut to Drive** → place in **My Drive** → name it `deep-learning-asm01`

**Step 2** — The notebook will find the data automatically at:

```
MyDrive/deep-learning-asm01/data/coco2014/
├── val2014/                              # ~41K JPEG images (~6 GB)  [required]
│   ├── COCO_val2014_000000000042.jpg
│   └── ...
├── train2014/                            # ~83K JPEG images (~13 GB) [optional]
│   └── ...
└── annotations/
    ├── instances_val2014.json            # category/supercategory labels
    ├── captions_val2014.json             # 5 captions per image
    ├── instances_train2014.json          # (only if DOWNLOAD_TRAIN2014=True)
    └── captions_train2014.json           # (only if DOWNLOAD_TRAIN2014=True)
```

If the shared Drive is not set up, the notebook auto-downloads `val2014` (~6 GB) to `/content` (not persisted between sessions).

**Download sources** (auto-handled by the notebook):
- Images: http://images.cocodataset.org/zips/val2014.zip (~6 GB)
- Annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip

---

## Technical Requirements Met

### Core (60%)
- [x] **Image**: ResNet50 (CNN) vs ViT-B/16 (Vision Transformer) — pretrained + fine-tuned on CIFAR-100
- [x] **Text**: BiLSTM (RNN) vs DistilBERT (Transformer) — pretrained + fine-tuned on 20 Newsgroups
- [x] **Multimodal**: CLIP Zero-shot vs Few-shot on MS COCO 2014 (real image–text pairs, train/val/test)
- [x] **Metrics**: Accuracy, F1-macro, Confusion Matrix, Classification Report

### Extensions (40%)
- [x] **Grad-CAM** — CNN interpretability: visualize which image regions drive predictions
- [x] **Error analysis** — confusion matrix heatmap, per-class F1 breakdown
- [x] **Multimodal fusion** — compare image-only vs caption-only vs image+caption zero-shot
- [x] **Few-shot curve** — accuracy vs number of shots (1→20), compared to zero-shot baseline
- [x] **Training curves** — loss, accuracy, F1-macro per epoch for both models
- [x] **Data augmentation** — RandomHorizontalFlip, ColorJitter, RandomRotation (image); label smoothing

---

## Datasets

### CIFAR-100 (Image Classification)
- **100 classes** (fine-grained) across 20 superclasses (animals, vehicles, household items, etc.)
- 50,000 train / 10,000 test — 32×32 RGB (resized to 224×224 for pretrained models)
- Auto-download via `torchvision.datasets.CIFAR100`
- Much more challenging than CIFAR-10: fine-grained intra-class variation, 5× more categories

### 20 Newsgroups (Text Classification)
- 20 newsgroup topics (alt.atheism, sci.space, rec.sport.hockey, talk.politics.guns, ...)
- 11,314 train / 7,532 test documents
- Auto-download via `sklearn.datasets.fetch_20newsgroups`

### MS COCO 2014 (Multimodal — CLIP)
- **Real image–text pairs**: each image has **5 human-written captions** describing its content
- **12 COCO supercategories** as classification labels: person, vehicle, outdoor, animal, accessory, sports, kitchen, food, furniture, electronic, appliance, indoor
- **train2014**: ~83K images (optional, for a larger few-shot support pool)
- **val2014**: ~41K images — split into train\_pool / val / test (default 40/30/30)
- Zero-shot uses CLIP with a 5-prompt ensemble; few-shot uses logistic regression linear probe
- **Three dataset splits**: train\_pool (few-shot support) → val (tuning) → test (final evaluation)
- **Three zero-shot variants compared**:
  1. Image-only: CLIP image features vs class text prompts
  2. Caption-only: CLIP caption features vs class text prompts
  3. Fusion: Avg(image + caption) features — true multimodal CLIP

---

## Expected Results

| Task | Model A | Acc A | Model B | Acc B |
|------|---------|-------|---------|-------|
| Image (CIFAR-100) | ResNet50 (CNN) | ~70–76% | ViT-B/16 | ~72–80% |
| Text (20 Newsgroups) | BiLSTM | ~75–82% | DistilBERT | ~88–92% |
| Multimodal (COCO 2014) | CLIP Zero-shot (fusion) | ~55–65% | CLIP 20-shot | ~70–80% |

*CIFAR-100 accuracy is lower than CIFAR-10 due to 100 fine-grained classes with 500 training images/class.*
*COCO 2014 accuracy reported on the held-out test split (~12K images for Option A).*

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

*CO5085 — Deep Learning and Applications in Computer Vision | HCMUT (Bach Khoa)*
