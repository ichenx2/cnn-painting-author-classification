# CNN Painting Author Classification

This project focuses on building a Convolutional Neural Network (CNN) from scratch to classify paintings by their authors. Instead of using pretrained models, I designed and trained a custom CNN architecture to tackle the challenges of a small and imbalanced dataset.

## Dataset
- The dataset consists of paintings categorized by artist.
- The original dataset was used without modification, except for data balancing techniques applied during preprocessing.

### Dataset Source
- Kaggle: [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
- Download links:
  - [Training Set (Google Drive)](https://drive.google.com/file/d/1K3FRVeabwV7TxboRsrkFcT34tPV9DWqr/view?usp=sharing)
  - [Test Set (Google Drive)](https://drive.google.com/file/d/1GCzWxFrIbb4d1JSXSrt3kRb6uERVqXpx/view?usp=sharing)

## Approach
- Built a custom CNN with 6 convolutional layers and 3 fully connected layers.
- Applied data augmentation (e.g., random transformations) to enhance generalization.
- Used oversampling to address class imbalance issues.
- Implemented techniques like Batch Normalization, Dropout, and L2 Regularization to prevent overfitting.
- EarlyStopping was applied to stop training when validation loss did not improve.

**Note:** Google Colab was used for training and experimentation.

## Model Architecture
The CNN architecture is as follows:

```python
keras.Sequential([
    # Convolution Blocks
    Conv2D → BatchNormalization → MaxPooling
    Conv2D → BatchNormalization → MaxPooling
    Conv2D → BatchNormalization → MaxPooling
    Conv2D → BatchNormalization → MaxPooling
    Conv2D → BatchNormalization → MaxPooling
    Conv2D → BatchNormalization → MaxPooling

    # Dense Layers
    Flatten
    Dense → BatchNormalization → Dropout
    Dense → BatchNormalization → Dropout
    Dense (Softmax)
])
```

This architecture aims to capture complex features through multiple convolutional blocks, while regularization techniques like BatchNormalization and Dropout help enhance generalization.

## Model Performance
- Final test accuracy: **around 46.6%**
- Despite the modest accuracy, significant improvements were made from the initial model (which achieved only around 5% accuracy).
- Focus was placed on improving data handling, model structure, and hyperparameter tuning.

### Training and Validation Curves

The following figures show the model's training and validation accuracy and loss over epochs:

#### Accuracy Curve
![Training and Validation Accuracy](/model_accuracy.png)

#### Loss Curve
![Training and Validation Loss](/model_loss.png)

## How to Run
1. Clone the repository.
2. Open the provided Colab notebook (`main.ipynb`) or run on your local environment with necessary packages installed (TensorFlow, NumPy, etc.).
3. Download the dataset and mount your Google Drive if using Colab.




