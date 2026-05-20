# Compression–Performance Trade-offs in Brain Tumor MRI Classification

This repository contains the code, notebooks, results discussion, and presentation assets for a deep learning study on **brain tumor MRI classification** using CNN baselines, EfficientNet-B0 transfer learning, and structured pruning.

The project investigates the research question:

> At what structured pruning sparsity level does an EfficientNet-B0 brain tumor classifier cease to be clinically viable, and how does its compressed performance compare to a simple CNN baseline?

Clinical viability is defined as **Macro F1 ≥ 0.95**.

---

## Project Summary

The task is a 4-class MRI classification problem:

- Glioma tumor
- Meningioma tumor
- Pituitary tumor
- No tumor

The final leakage-safe strategy combines the original Kaggle `Training` and `Testing` folders, splits original images into train/validation/test first, and applies conservative zoom augmentation **only to the training subset**. This avoids augmented copies of test images appearing in training.

---

## Final Leakage-Safe Results

| Model | Test Accuracy | Macro F1 | Clinically Viable? |
|---|---:|---:|---|
| CNN Baseline | 95.72% | 95.66% | Yes |
| Dense EfficientNet-B0 | 96.64% | 96.70% | Yes |
| EfficientNet-B0 Pruned 20% | 96.94% | 97.18% | Yes |
| EfficientNet-B0 Pruned 40% | 93.58% | 93.86% | No |
| EfficientNet-B0 Pruned 60% | 94.50% | 94.43% | No |
| EfficientNet-B0 Pruned 80% | 28.75% | 11.16% | No |

**Main finding:** 20% structured Conv2d pruning is clinically viable under the leakage-safe internal validation protocol. Beyond 20%, performance falls below the 95% Macro F1 threshold.

---

## Repository Structure

```text
brain-tumor-compression-study/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_leakage_safe_final_experiment.ipynb
│   ├── 00_report_method_original_test.ipynb
│   └── 00_combined_augmented_experiment.ipynb
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── pruning.py
├── docs/
│   ├── assignment2_data_methodology.pdf
│   └── assignment3_results_presentation.pptx
├── results/
│   └── README.md
├── assets/
│   └── montage.png
└── models/
    └── README.md
```

---

## Running the Project

The recommended environment is Kaggle with GPU enabled.

1. Add the Kaggle dataset:
   `sartajbhuvaji/brain-tumor-classification-mri`
2. Open `notebooks/01_leakage_safe_final_experiment.ipynb`.
3. Run all cells.
4. Export outputs from `/kaggle/working`.

The notebook saves result CSVs, model checkpoints, compressed model files, and summary outputs to `/kaggle/working`.

---

## Dataset

Dataset: Kaggle Brain Tumor MRI Classification Dataset by Sartaj Bhuvaji.

Expected Kaggle path:

```text
/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri
├── Training/
└── Testing/
```

---

## Methodology

### Model 1: CNN Baseline

A custom CNN trained from scratch using convolution, batch normalization, ReLU, max pooling, dropout, and fully connected classification layers.

### Model 2: Dense EfficientNet-B0

EfficientNet-B0 pretrained on ImageNet-1K and adapted for 4-class MRI classification.

### Model 3: Structured-Pruned EfficientNet-B0

L1-norm structured pruning using `torch.nn.utils.prune.ln_structured` is applied to convolutional filters at:

- 20% sparsity
- 40% sparsity
- 60% sparsity
- 80% sparsity

Each pruned model is fine-tuned and evaluated using accuracy, Macro F1, compressed size, and inference time.

---

## Leakage-Safe Improvement

An earlier version used offline augmentation before splitting, which risked leakage because an original image and its zoomed copy could end up in different subsets.

The final implementation fixes this by:

1. Combining original Training + Testing folders.
2. Splitting original images into train/validation/test.
3. Applying zoom augmentation only to the training subset.
4. Keeping validation and test images unaugmented.

---

## Key Interpretation

The 20% pruned model slightly outperformed the dense model in Macro F1, suggesting moderate pruning acted as a regularizer. However, the standard PyTorch model size remained similar because pruning zeroes weights without physically rebuilding the architecture. Compressed file size improved, but inference latency did not significantly improve.

---

## Limitations

- Results are based on an internal re-split protocol rather than an external clinical dataset.
- PyTorch pruning creates sparse weights but does not automatically reduce architecture dimensions.
- True edge acceleration would require model surgery, sparse runtime support, ONNX/TensorRT export, or deployment-aware pruning.
- Medical validation would require expert-labelled external data.

---

## Author

Muhammad Ammar
