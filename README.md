# Machine Learning – Academic Projects & Statistical Model Evaluations

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![University](https://img.shields.io/badge/University-Sapienza_Rome-E2001A?logo=sapienza-university-of-rome&logoColor=white)](https://www.uniroma1.it)

Welcome to the academic project repository for the **Machine Learning** course (M.Sc. in *Engineering in Computer Science* / *Artificial Intelligence*) at **Sapienza University of Rome**, Department of Computer, Control and Management Engineering (DIAG) "Antonio Ruberti".

This repository showcases the implementation, optimization, and evaluation of supervised and unsupervised machine learning pipelines, deep learning architectures, and dimensionality reduction techniques across diverse real-world datasets.

---

## 💼 Professional Value & Skills Demonstrated

This repository is designed to demonstrate core competencies in **Machine Learning Engineering**, **Data Science Pipelines**, and **Statistical Model Benchmarking**:

### 1. Robust Pipeline Design & Data Leakage Prevention
* **Preventing Group Leakage**: Solved crucial data leakage issues in audio processing (GTZAN dataset) by implementing custom grouping pipelines. Used `StratifiedGroupKFold` on a unique `song_id` key, ensuring segments from the same audio track were strictly separated between training and test sets.
* **Feature Engineering & Scalers**: Structured pipeline preprocessing including numerical standardization using `StandardScaler` to prevent feature scaling biases.

### 2. Multi-Model Optimization & Tuning
* **Hyperparameter Search**: Configured automated tuning using cross-validated Grid Search (`GridSearchCV` with 3-fold CV) targeting macro F1-score optimization.
* **Supervised Classifiers**: Implemented and benchmarked Support Vector Machines (Linear and RBF Kernels), Random Forests, Decision Trees, Softmax Regression, and Gaussian Naïve Bayes.

### 3. Deep Learning & Image Classification
* **Convolutional Neural Networks (CNNs)**: Designed convolutional architectures processing spectrogram image inputs ($224 \times 224 \times 3$), incorporating Data Augmentation layers (flips, rotations, zooms) to combat overfitting, combined with Early Stopping.
* **Feedforward Neural Networks (FNNs)**: Developed multi-layer neural networks for dense feature prediction in latent and compressed spaces.

### 4. Dimensionality Reduction & Latent Space Analysis
* **Linear vs. Non-Linear Compression**: Benchmarked Principal Component Analysis (**PCA**) against dense **Autoencoders** (neural bottleneck structures) across three task categories: Regression (Diamonds), Clustering (Wine Quality), and Classification (GTZAN Spectrograms).
* **Downstream Task Impact**: Evaluated how information loss in 2D and 3D latent spaces affects model accuracy ($R^2$, RMSE, Silhouette scores, and accuracy percentages).

---

## 📝 Project Summary & Key Results

### Assignment 1: Music Genre Classification (GTZAN Dataset)
* **Notebook**: [ML_Assignment_1_Generi_Musicali.ipynb](file:///C:/Users/Anton/Desktop/ML-University-Projects/ML_Assignment_1_Generi_Musicali.ipynb)
* **Reference Specification**: [First Assignment.pdf](file:///C:/Users/Anton/Desktop/ML-University-Projects/First%20Assignment.pdf)

This assignment focuses on a comparative evaluation of supervised classification models applied to music genre recognition using structural audio descriptors.

#### Model Comparison Results
All tabular models were tuned to maximize macro F1-score. RBF Kernel SVM achieved the highest accuracy, while Gaussian Naïve Bayes was the most computationally efficient:

| Model | Optimal Hyperparameters | Accuracy | Precision | Recall | Macro F1 | CV Score (F1) | Training Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Kernel RBF)** | `C: 10`, `gamma: 0.01` | **0.7365** | 0.7388 | 0.7364 | **0.7360** | 0.6894 | 81.81 |
| **Random Forest** | `max_depth: 25`, `min_samples_leaf: 3`, `n_estimators: 200` | 0.7097 | 0.7063 | 0.7091 | 0.7047 | 0.6639 | 165.35 |
| **Softmax Regression** | `C: 0.1` | 0.6840 | 0.6809 | 0.6831 | 0.6808 | 0.6455 | 7.14 |
| **SVM (Linear)** | `C: 1` | 0.6720 | 0.6624 | 0.6712 | 0.6639 | 0.6258 | 5.51 |
| **Decision Tree** | `criterion: 'entropy'`, `max_depth: 20`, `min_samples_leaf: 5` | 0.5455 | 0.5440 | 0.5453 | 0.5423 | 0.4872 | 29.87 |
| **Naïve Bayes (Gaussian)**| `var_smoothing: 1e-09` | 0.5178 | 0.5213 | 0.5161 | 0.4995 | 0.4748 | **0.36** |

#### Computational Efficiency (Accuracy / Time Ratio)
1. **Naïve Bayes**: $1.4469$ acc/s (Fastest)
2. **SVM (Linear)**: $0.1220$ acc/s
3. **Softmax Regression**: $0.0958$ acc/s
4. **Decision Tree**: $0.0183$ acc/s
5. **SVM (Kernel RBF)**: $0.0090$ acc/s (Best trade-off of accuracy and performance)
6. **Random Forest**: $0.0043$ acc/s (Highest computational cost)

---

### Assignment 2: Dimensionality Reduction Analysis (PCA vs. Autoencoders)
* **Notebook**: [second_assignment.ipynb](file:///C:/Users/Anton/Desktop/ML-University-Projects/second_assignment.ipynb)
* **Reference Specification**: [Second_Assignment_Machine_Learning.pdf](file:///C:/Users/Anton/Desktop/ML-University-Projects/Second_Assignment_Machine_Learning.pdf)

This assignment examines how linear (PCA) and non-linear (dense Autoencoders) dimensionality reduction affect regression, clustering, and classification performance.

#### Part 1: Regression (Diamonds Price Prediction)
Predicting diamond prices by compressing 9 features into **3 latent dimensions**. PCA and Autoencoder configurations perform similarly, with PCA retaining a slight edge:

| Feature Set | Dimensionality | RMSE ($) | R² Score |
| :--- | :---: | :---: | :---: |
| **Original** | 9 | **592.01** | **0.9780** |
| **PCA** | 3 | 1578.82 | 0.8432 |
| **Autoencoder** | 3 | 1613.60 | 0.8362 |

#### Part 2: Clustering (Wine Quality Clustering)
Grouping white wines into $K = 6$ clusters. Reducing noise by compressing 11 features into **2 dimensions** improved clustering quality (Silhouette score):

| Feature Set | Dimensionality | Silhouette Score |
| :--- | :---: | :---: |
| **Original** | 11 | 0.1478 |
| **PCA** | 2 | 0.3352 |
| **Autoencoder** | 2 | **0.3445** |

#### Part 3: Image Classification (GTZAN Spectrogram Images)
Classifying 10 music genres from raw spectrograms ($64 \times 64 = 4096$ features) compressed down to **128 dimensions**. A spatial CNN outperforms both reduced FNN models, and linear PCA preserves class boundaries better than the dense Autoencoder:

| Model | Compression Method | Test Set Accuracy |
| :--- | :--- | :---: |
| **CNN** | None (Original $64 \times 64$ images) | **0.5400** |
| **FNN + PCA** | PCA (128 components) | 0.4300 |
| **FNN + Autoencoder**| Autoencoder (128 latent units) | 0.2200 |

---

## 📂 Project Structure

```text
.
├── ML_Assignment_1_Generi_Musicali.ipynb       # Tabular & CNN genre classification
├── second_assignment.ipynb                     # PCA vs. Autoencoder analysis
├── First Assignment.pdf                        # Assignment 1 spec doc
├── Second_Assignment_Machine_Learning.pdf      # Assignment 2 spec doc
└── README.md                                   # Project Documentation
```

---

## 💻 Installation and Usage

### Prerequisites
Install Python 3.8+ and install the dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow opencv-python seaborn matplotlib pypdf
```

### Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/tonyrub3/ML-University-Projects.git
   cd ML-University-Projects
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and execute `ML_Assignment_1_Generi_Musicali.ipynb` or `second_assignment.ipynb`.

*(Note: Heavy Deep Learning training stages were run with CUDA/GPU acceleration in Google Colab).*

---

## 👥 The Team

* **Antonio Rubino** ([tonyrub3](https://github.com/tonyrub3)) – DIAG Department, Sapienza University
* **Nicola Moscufo** – DIAG Department, Sapienza University

---

*This project was developed for academic evaluation under DIAG, Sapienza University of Rome.*
