# Food Image Segmentation

Semantic segmentation of individual food items on a plate, built for coursework at Newcastle University (MSc Data Science & AI — Advanced AI module, CSC8645).

**Why:** dietary-intake monitoring matters for patient and resident well-being in hospitals and care homes. Automatically identifying what's on a plate — rather than relying on manual logging — is a step toward that. The brief asked for a self-designed segmentation algorithm (optionally literature-inspired), evaluated against ground truth with mean Intersection-over-Union (mIoU).

Full write-up: [`report/Task1_Report.pdf`](report/Task1_Report.pdf).

## Dataset

[FoodSeg103](https://xiongweiwu.github.io/foodseg103.html) [1] — a benchmark of 7,118 images with pixel-level masks over 104 classes (background + 103 food categories).

Using the dataset's official split files:
- **4,983 training images**, further divided 80/20 into **3,986 training / 997 validation** images
- **2,135 test images**

An exploratory scan of 300 training masks found 78 of the 103 food categories present, with background appearing in every image — confirming class imbalance (background vs. food pixels) as the main data challenge to design around.

**Access:** FoodSeg103 is publicly available for research use from the [official project page](https://xiongweiwu.github.io/foodseg103.html) and the authors' [benchmark repository](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) (Apache-2.0-licensed code; the dataset itself is distributed via a password-protected download link on request, with citation of the original paper [1] requested). Because of that gating and its size, the raw dataset is **not included in this repo** — download it yourself from the link above and update `BASE_PATH` in the notebook to point at it.

## Approach

**Architecture:** U-Net [2] with a ResNet34 encoder, implemented via `segmentation-models-pytorch`. Images and masks are resized to 256×256; the decoder outputs `(batch, 104, 256, 256)` logits, and per-pixel class labels come from an argmax over the class axis.

**Evaluation metric:** mIoU as the primary metric — it averages per-class IoU equally across all 104 classes, which keeps it robust to the background-pixel imbalance (a plain pixel-accuracy score would be dominated by background). Pixel accuracy is tracked as a secondary metric.

Two models were trained and compared:

**1. Baseline**
- ResNet34 encoder, randomly initialised (no pretraining)
- No augmentation, no checkpoint selection
- `CrossEntropyLoss` with no class weights
- Adam, lr = 1×10⁻⁴, 10 epochs, batch size 32
- Validation mIoU moved from 0.0219 (epoch 1) to 0.0523 (epoch 10)

**2. Enhanced** — same U-Net + ResNet34 structure, with five additions:
1. ImageNet-pretrained encoder weights [3]
2. Background class down-weighted to 0.1 in the loss (food classes stay at 1.0)
3. `ReduceLROnPlateau` scheduler (factor = 0.5, patience = 3)
4. Joint image/mask augmentation — horizontal flip, 90° rotation, zoom (0.75–0.95), colour jitter
5. Checkpointing on validation mIoU improvement (first save at epoch 1, Val mIoU 0.0248; 45 epochs total)

## Results

| Model | Accuracy | mIoU |
|---|---|---|
| Baseline — random init, 10 epochs | 0.5397 | 0.0523 |
| Enhanced — ImageNet + weighted loss + augmentation + scheduler + checkpointing, 45 epochs | 0.6829 | 0.2118 |

Pre-training, weighted loss, scheduling, augmentation, and checkpointing together produced a **~4× improvement in mIoU** (0.0523 → 0.2118) and lifted pixel accuracy from 0.5397 to 0.6829. The enhanced model's validation mIoU climbed steadily across all 45 epochs without plateauing early, unlike the baseline, which stalled almost immediately (see training curves in the report, Fig. 3).

## Repository structure

```
food-image-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Task1_Food_Segmentation.ipynb
└── report/
    └── Task1_Report.pdf
```

## Running the notebook

Developed and run on Google Colab with a GPU runtime. The notebook mounts Google Drive and copies a zipped dataset from a personal folder (`drive.mount(...)`, then `!cp .../Task 1 Food Segmentation Dataset.zip`). To run it elsewhere:

1. Remove or skip the `google.colab` import and `drive.mount(...)` cell.
2. Download FoodSeg103 from the link above and set `BASE_PATH` (in the Settings cell) to point at it — the notebook expects the same `Images/img_dir`, `Images/ann_dir`, and `ImageSets` layout the dataset ships with.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run cells top to bottom — EDA, baseline training, enhanced training, then the comparison/visualisation cells.

## References

[1] X. Wu et al., "A large-scale benchmark for food image segmentation," *Proc. ACM Int. Conf. Multimedia (MM)*, 2021.
[2] O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," *Proc. MICCAI*, 2015.
[3] K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," *Proc. IEEE CVPR*, 2016.
