# Digit Recognizer (Kaggle)

Solution for the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition (MNIST handwritten digit classification).

**Kaggle public leaderboard score: 0.99657**

## Data

- `train.csv`: 42,000 labeled 28x28 grayscale digit images (784 pixel columns + 1 label column).
- `test.csv`: unlabeled 28x28 images to predict for submission.
- Train data is split 90/10 (stratified by label, `random_state=42`) into a training set and a held-out validation set.
- Pixel values are normalized to `[0, 1]` (divided by 255) and reshaped to `(1, 28, 28)` tensors.

### Data augmentation

The training set is expanded with a pool of statically-generated variants, each covering a different type of distortion, concatenated onto the original images:

| Variant | Transform |
|---|---|
| Rotation | ±4° (left/right) |
| Stretch | horizontal & vertical (resize + center-crop) |
| Elastic distortion | `ElasticTransform(alpha=34, sigma=4)` |
| Shear | ±10° (left/right) |
| Stroke thickness | dilate & erode (via max/min pooling) |

This roughly 10x's the effective training set size. An on-the-fly random-augmentation path (`AugmentedDataset` + `transforms.Compose`, applying fresh random rotation/scale/translate/elastic distortion per sample per epoch) is also implemented but currently unused in favor of the static pool, which converged faster and scored higher in testing.

## Model

A CNN (`CNN` class in [digit_recognizer.ipynb](digit_recognizer.ipynb)) with three convolutional blocks followed by global average pooling and a small classifier head:

| Block | Layers | Output channels |
|---|---|---|
| block1 | (Conv3x3 → BatchNorm → ReLU) x2 → MaxPool2x2 → Dropout(0.5) | 32 |
| block2 | (Conv3x3 → BatchNorm → ReLU) x2 → MaxPool2x2 → Dropout(0.5) | 64 |
| block3 | (Conv3x3 → BatchNorm → ReLU) x2 → Dropout(0.5) | 128 |
| head | AdaptiveAvgPool(1x1) → Linear(128,64) → ReLU → Dropout(0.5) → Linear(64,10) | 10 classes |

Global average pooling instead of flattening, with dropout at every stage for regularization.

## Training

- **Loss:** CrossEntropyLoss with label smoothing (`label_smoothing=0.1`)
- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=0.001`)
- **LR schedule:** `ReduceLROnPlateau` (factor 0.5, patience 1 epoch on validation loss)
- **Batch size:** 64
- **Epochs:** up to 50, with early stopping (patience 5 epochs on validation loss)
- **Checkpointing:** best model (lowest validation loss) saved to `CNN.pt` after every improving epoch
- **Seed:** fixed (`torch.manual_seed(42)`) for reproducibility

Label smoothing was the single change that consistently pushed validation accuracy past the ~99.5% ceiling other single-model tweaks (architecture depth, dropout tuning, split ratio, augmentation variety) kept converging to.

## Inference

Two prediction paths are implemented for the Kaggle test set:

- **Standard inference** — a single forward pass per image through the best checkpoint.
- **Test-Time Augmentation (TTA)** — the same checkpoint runs several perturbed views of each test image (rotation, shift, scale, shear, dilate/erode) and averages the softmax probabilities across views for the final prediction. Gives a small, consistent accuracy boost over standard inference.

## Error Analysis

A dedicated cell reloads the best checkpoint, evaluates it on the held-out validation set, and displays every misclassified image (sorted by how confidently wrong the model was), alongside its predicted and true label. On the best run, this surfaced 14 errors out of 4,200 validation images (~0.33%), spread across many different digit pairs with no dominant, repeating confusion — consistent with genuinely ambiguous/poorly-written handwriting rather than a systematic model weakness.

## Results

| Metric | Value |
|---|---|
| Best validation accuracy (held-out 10%) | ~99.7% |
| Validation error rate | 14 / 4,200 (~0.33%) |
| **Kaggle public leaderboard score** | **0.99657** |

This score is competitive with well-known public benchmark solutions for this competition (e.g., the widely-referenced ["Introduction to CNN Keras - 0.997"](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) notebook, which reaches 99.671% with a comparable, though non-BatchNorm, architecture).
