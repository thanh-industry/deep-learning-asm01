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
| [`ZeroShot-vs-FewShot/`](ZeroShot-vs-FewShot/) | Multimodal Classification | CLIP Zero-shot vs Few-shot | MS COCO Captions (12 supercategories, 5K images + captions) |

---

## Repository Structure

```
deep-learning-asm01/
├── CNN-vs-ViT/
│   └── cnn_vs_vit.ipynb            # ResNet50 vs ViT-B/16 on CIFAR-100
├── RNN-vs-Transformer/
│   └── rnn_vs_transformer.ipynb    # BiLSTM vs DistilBERT on 20 Newsgroups
├── ZeroShot-vs-FewShot/
│   └── zeroshot_vs_fewshot.ipynb   # CLIP zero-shot vs few-shot on MS COCO Captions
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

### COCO Dataset on Google Drive (recommended — avoids re-downloading)

```
MyDrive/deep-learning-asm01/data/coco/
├── val2017/                          # ~5K JPEG images (~1 GB)
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    ├── instances_val2017.json        # category/supercategory labels
    └── captions_val2017.json         # 5 captions per image
```

Download from: http://images.cocodataset.org/zips/val2017.zip and http://images.cocodataset.org/annotations/annotations_trainval2017.zip

---

## Technical Requirements Met

### Core (60%)
- [x] **Image**: ResNet50 (CNN) vs ViT-B/16 (Vision Transformer) — pretrained + fine-tuned on CIFAR-100
- [x] **Text**: BiLSTM (RNN) vs DistilBERT (Transformer) — pretrained + fine-tuned on 20 Newsgroups
- [x] **Multimodal**: CLIP Zero-shot vs Few-shot on MS COCO Captions (real image–text pairs)
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

### MS COCO Captions 2017 (Multimodal — CLIP)
- **Real image–text pairs**: each image has **5 human-written captions** describing its content
- **12 COCO supercategories** used as classification labels: person, vehicle, outdoor, animal, accessory, sports, kitchen, food, furniture, electronic, appliance, indoor
- ~5,000 validation images with full annotations
- Zero-shot classification uses CLIP with prompt templates; few-shot uses linear probe
- **Three zero-shot variants compared**:
  1. Image-only: CLIP image features vs class name prompts
  2. Caption-only: CLIP caption features vs class name prompts
  3. Fusion: combined image + caption features (true multimodal)

---

## Expected Results

| Task | Model A | Acc A | Model B | Acc B |
|------|---------|-------|---------|-------|
| Image (CIFAR-100) | ResNet50 (CNN) | ~70–76% | ViT-B/16 | ~72–80% |
| Text (20 Newsgroups) | BiLSTM | ~75–82% | DistilBERT | ~88–92% |
| Multimodal (COCO) | CLIP Zero-shot | ~55–65% | CLIP 20-shot | ~72–82% |

*CIFAR-100 accuracy is lower than CIFAR-10 due to 100 fine-grained classes with 500 training images/class.*

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
