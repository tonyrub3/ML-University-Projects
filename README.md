# Machine Learning - Academic Projects (Sapienza University of Rome)

Welcome to the repository of projects for the **Machine Learning** course (M.Sc. in *Engineering in Computer Science* / *Artificial Intelligence*) at **Sapienza University of Rome**, Department of Computer, Control and Management Engineering (DIAG) "Antonio Ruberti".

**Instructors:** Prof. Federico Fusco, Prof. Fabio Patrizi  
**Students:** Antonio Rubino (ID: 2235484), Nicola Moscufo (ID: 2254216)  
**Academic Year:** 2025/2026

This repository gathers two main assignments dedicated to the practical application of supervised and unsupervised Machine Learning models, dimensionality reduction techniques, and Deep Learning experiments.

---

## Table of Contents
1. [Assignment 1: Comparative Study of Classification Algorithms](#assignment-1-comparative-study-of-classification-algorithms)
2. [Assignment 2: Regression, Clustering, and Classification with Dimensionality Reduction](#assignment-2-regression-clustering-and-classification-with-dimensionality-reduction)
3. [Project Structure](#project-structure)
4. [Installation and Usage](#installation-and-usage)

---

## Assignment 1: Comparative Study of Classification Algorithms
* **Notebook:** [ML_Assignment_1_Generi_Musicali.ipynb](ML_Assignment_1_Generi_Musicali.ipynb)  
* **Reference Document:** [First Assignment.pdf](First%20Assignment.pdf)

### Objective
Develop a systematic comparative study of supervised classification algorithms applied to music genre recognition using audio features extracted from the **GTZAN** dataset.

### Preprocessing and Data Leakage Prevention
The GTZAN dataset contains audio files split into 3-second segments. Since different segments belong to the same original track (song), a random split would cause a severe **data leakage** phenomenon (overestimating the test accuracy due to parts of the same song being present in both training and test sets).
* **Resolution:** The data was grouped by `song_id` (a unique ID of the original song extracted from the filename).
* **Split:** `StratifiedGroupKFold` was applied to guarantee that segments of a specific song were strictly placed in either the training set or the test set, while maintaining a balanced class distribution.
* **Scaling:** Numerical features were standardized using `StandardScaler`.

### Tabular Models Results
The models were optimized using hyperparameter tuning (`GridSearchCV` with 3-fold cross-validation on macro F1-score).

| Model | Optimal Hyperparameters | Accuracy | Precision | Recall | Macro F1 | CV Score (F1) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Kernel RBF)** | `C: 10`, `gamma: 0.01` | **0.7365** | 0.7388 | 0.7364 | **0.7360** | 0.6894 | 81.81 |
| **Random Forest** | `max_depth: 25`, `min_samples_leaf: 3`, `n_estimators: 200` | 0.7097 | 0.7063 | 0.7091 | 0.7047 | 0.6639 | 165.35 |
| **Softmax Regression** | `C: 0.1` | 0.6840 | 0.6809 | 0.6831 | 0.6808 | 0.6455 | 7.14 |
| **SVM (Linear)** | `C: 1` | 0.6720 | 0.6624 | 0.6712 | 0.6639 | 0.6258 | 5.51 |
| **Decision Tree** | `criterion: 'entropy'`, `max_depth: 20`, `min_samples_leaf: 5` | 0.5455 | 0.5440 | 0.5453 | 0.5423 | 0.4872 | 29.87 |
| **Naïve Bayes (Gaussian)**| `var_smoothing: 1e-09` | 0.5178 | 0.5213 | 0.5161 | 0.4995 | 0.4748 | **0.36** |

#### Computational Efficiency
Ratios of $\text{Accuracy} / \text{Time}$ (seconds for training and tuning):
1. **Naïve Bayes:** $1.4469$ acc/s (Extremely fast)
2. **SVM (Linear):** $0.1220$ acc/s
3. **Softmax Regression:** $0.0958$ acc/s
4. **Decision Tree:** $0.0183$ acc/s
5. **SVM (Kernel RBF):** $0.0090$ acc/s (Best trade-off of accuracy and time)
6. **Random Forest:** $0.0043$ acc/s (Most computationally expensive)

### Deep Learning: Convolutional Neural Network (CNN)
In addition to tabular features, a **CNN** was designed to classify GTZAN audio spectrogram images directly (resolution of $224 \times 224 \times 3$).
* **Data Augmentation:** Random Flip, Rotation, Zoom, and Contrast to mitigate overfitting.
* **Architecture:** Input -> Data Augmentation -> 4 conv blocks (Conv2D + BatchNormalization + MaxPooling2D) -> GlobalAveragePooling2D -> Dense(256) with Dropout(0.5) -> Softmax (10 classes).
* **Optimization:** Adam ($lr = 0.0001$), *sparse categorical crossentropy* loss, and Early Stopping with a patience of 7 epochs.

---

## Assignment 2: Regression, Clustering, and Classification with Dimensionality Reduction
* **Notebook:** [second_assignment.ipynb](second_assignment.ipynb)  
* **Reference Document:** [Second_Assignment_Machine_Learning (1).pdf](Second_Assignment_Machine_Learning%20%281%29.pdf)

### Objective
Analyze the effectiveness of two different dimensionality reduction techniques: a linear one (**PCA**) and a non-linear one (neural dense **Autoencoder**), applied to Regression, Clustering, and Classification tasks respectively.

### Part 1: Regression (Diamonds Dataset)
Predict diamond prices starting from physical and qualitative features (9 features in total, reduced to **3 latent dimensions**).
* **Autoencoder Architecture:** Input (9) -> Dense Relu (3) -> Dense Linear (9).
* **Predictive Model:** Feedforward Neural Network (FNN) (64 relu -> 32 relu -> 1 linear).

| Feature Set | Dimensionality | RMSE ($) | R² Score |
| :--- | :---: | :---: | :---: |
| **Original** | 9 | **592.01** | **0.9780** |
| **PCA** | 3 | 1578.82 | 0.8432 |
| **Autoencoder** | 3 | 1613.60 | 0.8362 |

*Reducing to only 3 features leads to a natural performance drop, but PCA and the Autoencoder perform very closely ($R^2 \approx 84\%$), with a minor edge for PCA.*

### Part 2: Clustering (Wine Quality Dataset)
Group white wines into $K = 6$ clusters based on chemical-physical properties (11 features in total, reduced to **2 dimensions** for visualization).
* **Algorithm:** K-Means ($K=6$).
* **Metrics:** Evaluated via Silhouette Score.

| Feature Set | Dimensionality | Silhouette Score |
| :--- | :---: | :---: |
| **Original** | 11 | 0.1478 |
| **PCA** | 2 | 0.3352 |
| **Autoencoder** | 2 | **0.3445** |

*Removing noise and compressing the space using dimensionality reduction significantly improves cluster compactness, raising the Silhouette Score from 0.14 to over 0.33. The dense Autoencoder yields the best results.*

### Part 3: Classification (GTZAN Spectrograms)
Classify spectrogram images (resized to $64 \times 64 = 4096$ pixels) into 10 music genres.
* **Reduction:** Features compressed from 4096 to **128 dimensions**.
* **Compared Models:**
  1. **Original CNN:** Convolutional network applied directly to original images ($64 \times 64 \times 1$).
  2. **Dense PCA:** Feedforward neural network (FNN) (128 -> Dropout 0.3 -> 64 -> 10) on features compressed using PCA.
  3. **Dense Autoencoder:** Feedforward neural network (FNN) (128 -> Dropout 0.3 -> 64 -> 10) on features compressed using a dense Autoencoder (4096 -> 128 -> 4096).

| Model | Compression Method | Test Set Accuracy |
| :--- | :--- | :---: |
| **CNN** | None (Original $64 \times 64$ images) | **0.5400** |
| **FNN + PCA** | PCA (128 components) | 0.4300 |
| **FNN + Autoencoder**| Autoencoder (128 latent units) | 0.2200 |

*The CNN achieves the best performance by preserving spatial features in the spectrograms. In the reduced space, the linear compression of PCA is much more effective than the dense Autoencoder for maintaining class separability (43% vs 22%).*

---

## Project Structure
```text
.
├── ML_Assignment_1_Generi_Musicali.ipynb  # First Assignment Notebook
├── second_assignment.ipynb                # Second Assignment Notebook
├── First Assignment.pdf                   # First Assignment PDF Prompt
├── Second_Assignment_Machine_Learning (1).pdf # Second Assignment PDF Prompt
└── README.md                              # Project Documentation
```

---

## Installation and Usage

### Requirements
To run the notebooks, install Python 3.8+ and the required packages via pip:

```bash
pip install pandas numpy scikit-learn tensorflow opencv-python seaborn matplotlib pypdf
```

### Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/tonyrub3/ML-University-Projects.git
   cd ML-University-Projects
   ```
2. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open and run the notebooks `ML_Assignment_1_Generi_Musicali.ipynb` or `second_assignment.ipynb`.

*(Note: Certain deep learning training blocks were originally run in Google Colab with GPU acceleration due to compute requirements).*
