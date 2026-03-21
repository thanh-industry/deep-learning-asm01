# Deep Learning Assignment 1 — CO5085
**Deep Learning and Applications in Computer Vision | Semester 2, 2025–2026**

> **Instructor:** Le Thanh Sach | **Deadline:** 23:59, March 25 2026

---

## Project Overview

Compare deep learning model families across **three data modalities**:

| Folder | Task | Models Compared | Default Dataset |
|--------|------|-----------------|----------------|
| [`CNN-vs-ViT/`](CNN-vs-ViT/) | Image Classification | ResNet50 vs ViT-B/16 | CIFAR-10 (10 classes, 60K images) |
| [`RNN-vs-Transformer/`](RNN-vs-Transformer/) | Text Classification | BiLSTM vs DistilBERT | 20 Newsgroups (20 classes, ~18K docs) |
| [`ZeroShot-vs-FewShot/`](ZeroShot-vs-FewShot/) | Multimodal Classification | CLIP Zero-shot vs Few-shot | Oxford Pets (37 classes, ~7K images) |

---

## Repository Structure

```
deep-learning-asm01/
├── CNN-vs-ViT/
│   └── cnn_vs_vit.ipynb            # ResNet50 vs ViT-B/16 on CIFAR-10
├── RNN-vs-Transformer/
│   └── rnn_vs_transformer.ipynb    # BiLSTM vs DistilBERT on 20 Newsgroups
├── ZeroShot-vs-FewShot/
│   └── zeroshot_vs_fewshot.ipynb   # CLIP zero-shot vs few-shot on Oxford Pets
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

### Using a Custom Dataset on Google Drive

```
MyDrive/deep-learning-asm01/
└── data/
    └── my_image_dataset/     # for CNN vs ViT
        ├── train/
        │   ├── class_1/
        │   └── class_2/
        └── val/
            ├── class_1/
            └── class_2/
```

Set `GDRIVE_DATASET_PATH = '/content/drive/MyDrive/deep-learning-asm01/data/my_image_dataset'` in the config cell.

---

## Technical Requirements Met

### Core (60%)
- [x] **Image**: ResNet50 (CNN) vs ViT-B/16 (Vision Transformer) — pretrained + fine-tuned
- [x] **Text**: BiLSTM (RNN) vs DistilBERT (Transformer) — pretrained + fine-tuned
- [x] **Multimodal**: CLIP Zero-shot vs Few-shot (N=1,5,10,20 shots)
- [x] **Metrics**: Accuracy, F1-macro, Confusion Matrix, Classification Report

### Extensions (40%)
- [x] **Grad-CAM** — CNN interpretability: visualize which image regions drive predictions
- [x] **Error analysis** — confusion matrix heatmap, per-class F1 breakdown
- [x] **Few-shot curve** — accuracy vs number of shots (1→20), compared to zero-shot baseline
- [x] **Training curves** — loss, accuracy, F1-macro per epoch for both models
- [x] **Data augmentation** — RandomHorizontalFlip, ColorJitter, RandomRotation (image); label smoothing

---

## Datasets

### CIFAR-10 (Image Classification)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 train / 10,000 test — 32×32 RGB (resized to 224×224)
- Auto-download via `torchvision.datasets.CIFAR10`

### 20 Newsgroups (Text Classification)
- 20 newsgroup topics (alt.atheism, sci.space, rec.sport.hockey, talk.politics.guns, ...)
- 11,314 train / 7,532 test documents
- Auto-download via `sklearn.datasets.fetch_20newsgroups`

### Oxford-IIIT Pet (Multimodal — CLIP)
- 37 pet breeds (Abyssinian, Beagle, Bengal, Boxer, ...)
- ~3,680 train / ~3,669 test images
- Auto-download via `torchvision.datasets.OxfordIIITPet`
- CLIP (ViT-B/32) processes both image features and text prompts: `"a photo of a {breed}, a type of pet."`

---

## Expected Results

| Task | Model A | Acc A | Model B | Acc B |
|------|---------|-------|---------|-------|
| Image | ResNet50 (CNN) | ~87–90% | ViT-B/16 | ~88–92% |
| Text | BiLSTM | ~75–82% | DistilBERT | ~88–92% |
| Multimodal | CLIP Zero-shot | ~85–90% | CLIP 20-shot | ~92–95% |

---

## Dependencies

```
torch >= 2.0
torchvision >= 0.15
transformers >= 4.35
timm >= 0.9
open-clip-torch >= 2.20
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
pandas >= 2.0
tqdm >= 4.65
pytorch-grad-cam >= 1.4
```

All packages are either pre-installed in Colab or installed via `!pip install` in the first cell of each notebook.

---

*CO5085 — Deep Learning and Applications in Computer Vision | HCMUT (Bach Khoa)*
