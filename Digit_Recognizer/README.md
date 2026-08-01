# Digit Recognizer (Kaggle)

Solution for the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition (MNIST handwritten digit classification).

**Kaggle public leaderboard score: 0.99464**

## Data

- `train.csv`: 42,000 labeled 28x28 grayscale digit images (784 pixel columns + 1 label column).
- `test.csv`: unlabeled 28x28 images to predict for submission.
- Train data was split 80/20 (stratified by label, `random_state=42`) into a training set and a held-out validation set.
- Pixel values were used as raw floats (0–255, no normalization) and reshaped to `(1, 28, 28)` tensors.

## Model

A CNN (`CNN` class in [digit_recognizer.ipynb](digit_recognizer.ipynb)) with three convolutional blocks followed by global average pooling and a small classifier head:

| Block | Layers | Output channels |
|---|---|---|
| block1 | Conv3x3 → ReLU → Conv3x3 → BatchNorm → ReLU → MaxPool2x2 | 32 |
| block2 | Conv3x3 → ReLU → Conv3x3 → BatchNorm → ReLU → MaxPool2x2 | 64 |
| block3 | Conv3x3 → ReLU → Conv3x3 → BatchNorm → ReLU | 128 |
| head | AdaptiveAvgPool(1x1) → Linear(128,128) → ReLU → Dropout(0.5) → Linear(128,10) | 10 classes |

Total: ~6 conv layers with batch normalization, double-conv blocks per stage, global average pooling instead of flattening, and dropout for regularization in the classifier.

## Training

- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=0.01`)
- **LR schedule:** `ReduceLROnPlateau` (factor 0.5, patience 1 epoch on validation loss)
- **Batch size:** 64
- **Epochs:** up to 50, with early stopping (patience 5 epochs on validation loss)
- **Checkpointing:** best model (lowest validation loss) saved to `CNN.pt` after every improving epoch
- **Seed:** fixed (`torch.manual_seed(42)`) for reproducibility

Training stopped early at **epoch 18** (no val-loss improvement for 5 consecutive epochs). The learning rate was halved five times over the run (0.001 → ~0.00003) as validation loss plateaued.

## Results

| Metric | Value |
|---|---|
| Best epoch | 13 |
| Best validation loss | 0.0183 |
| Best validation accuracy (held-out 20%) | 99.5119% |
| **Kaggle public leaderboard score** | **0.99464** |

Train and validation loss both dropped sharply in the first 3 epochs (0.27 → 0.03), then validation loss fluctuated in a narrow band (~0.018–0.03) while training loss kept decreasing toward ~0.001, indicating the model was close to its generalization limit on this dataset well before overfitting became severe.
