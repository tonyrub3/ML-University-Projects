# Machine Learning - Progetti Accademici (Sapienza Università di Roma)

Benvenuto nel repository dei progetti per il corso di **Machine Learning** (Laurea Magistrale in *Engineering in Computer Science* / *Artificial Intelligence*) presso la **Sapienza Università di Roma**, Dipartimento di Ingegneria Informatica, Automatica e Gestionale (DIAG) "Antonio Ruberti".

**Docenti:** Prof. Federico Fusco, Prof. Fabio Patrizi  
**Studenti:** Antonio Rubino (Matr. 2235484), Nicola Moscufo (Matr. 2254216)  
**Anno Accademico:** 2025/2026

Questo repository raccoglie due assignment principali dedicati all'applicazione pratica di modelli di Machine Learning supervisionato e non supervisionato, riduzione della dimensionalità ed esperimenti di Deep Learning.

---

## Indice
1. [Assignment 1: Studio Comparativo di Algoritmi di Classificazione](#assignment-1-studio-comparativo-di-algoritmi-di-classificazione)
2. [Assignment 2: Regressione, Clustering e Classificazione con Riduzione Dimensionalità](#assignment-2-regressione-clustering-e-classificazione-con-riduzione-dimensionalità)
3. [Struttura del Progetto](#struttura-del-progetto)
4. [Installazione e Utilizzo](#installazione-e-utilizzo)

---

## Assignment 1: Studio Comparativo di Algoritmi di Classificazione
* **Notebook:** [ML_Assignment_1_Generi_Musicali.ipynb](ML_Assignment_1_Generi_Musicali.ipynb)  
* **Documento di Riferimento:** [First Assignment.pdf](First%20Assignment.pdf)

### Obiettivo
Sviluppare uno studio comparativo sistematico di algoritmi di classificazione supervisionata applicati al riconoscimento dei generi musicali a partire da feature audio estratte dal dataset **GTZAN**.

### Preprocessing e Prevenzione del Data Leakage
Il dataset GTZAN contiene file audio suddivisi in segmenti da 3 secondi. Poiché segmenti diversi appartengono alla stessa traccia originale (canzone), uno split casuale provocherebbe un grave fenomeno di **data leakage** (sovrastima dell'accuratezza dovuta alla presenza di parti della stessa canzone sia in train che in test).
* **Risoluzione:** I dati sono stati raggruppati per `song_id` (ID univoco della canzone originale estratto dal nome del file).
* **Split:** È stato applicato `StratifiedGroupKFold` per garantire che i segmenti di una specifica canzone fossero inseriti esclusivamente nel training set o nel test set, mantenendo la distribuzione delle classi bilanciata.
* **Scaling:** Le feature numeriche sono state standardizzate tramite `StandardScaler`.

### Risultati dei Modelli Tabulari
I modelli sono stati ottimizzati tramite ricerca degli iperparametri (`GridSearchCV` con 3-fold cross-validation su macro F1 score). 

| Modello | Iperparametri Ottimali | Accuratezza | Precision | Recall | Macro F1 | CV Score (F1) | Tempo (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Kernel RBF)** | `C: 10`, `gamma: 0.01` | **0.7365** | 0.7388 | 0.7364 | **0.7360** | 0.6894 | 81.81 |
| **Random Forest** | `max_depth: 25`, `min_samples_leaf: 3`, `n_estimators: 200` | 0.7097 | 0.7063 | 0.7091 | 0.7047 | 0.6639 | 165.35 |
| **Softmax Regression** | `C: 0.1` | 0.6840 | 0.6809 | 0.6831 | 0.6808 | 0.6455 | 7.14 |
| **SVM (Lineare)** | `C: 1` | 0.6720 | 0.6624 | 0.6712 | 0.6639 | 0.6258 | 5.51 |
| **Decision Tree** | `criterion: 'entropy'`, `max_depth: 20`, `min_samples_leaf: 5` | 0.5455 | 0.5440 | 0.5453 | 0.5423 | 0.4872 | 29.87 |
| **Naïve Bayes (Gaussian)**| `var_smoothing: 1e-09` | 0.5178 | 0.5213 | 0.5161 | 0.4995 | 0.4748 | **0.36** |

#### Efficienza Computazionale
Rapporto $\text{Accuratezza} / \text{Tempo}$ (secondi per l'addestramento ed ottimizzazione):
1. **Naïve Bayes:** $1.4469$ acc/s (Estremamente veloce)
2. **SVM (Lineare):** $0.1220$ acc/s
3. **Softmax Regression:** $0.0958$ acc/s
4. **Decision Tree:** $0.0183$ acc/s
5. **SVM (Kernel RBF):** $0.0090$ acc/s (Miglior compromesso prestazioni/accuratezza)
6. **Random Forest:** $0.0043$ acc/s (Il più oneroso)

### Deep Learning: Rete Neurale Convoluzionale (CNN)
In aggiunta alle feature tabulari, è stata progettata una **CNN** per classificare direttamente le immagini degli spettrogrammi audio di GTZAN (risoluzione $224 \times 224 \times 3$).
* **Data Augmentation:** Random Flip, Rotation, Zoom e Contrast per mitigare l'overfitting.
* **Architettura:** Input -> Data Augmentation -> 4 blocchi conv (Conv2D + BatchNormalization + MaxPooling2D) -> GlobalAveragePooling2D -> Dense(256) con Dropout(0.5) -> Softmax (10 classi).
* **Ottimizzazione:** Adam ($lr = 0.0001$), loss *sparse categorical crossentropy*, arresto anticipato (Early Stopping) con patience di 7 epoche.

---

## Assignment 2: Regressione, Clustering e Classificazione con Riduzione Dimensionalità
* **Notebook:** [second_assignment.ipynb](second_assignment.ipynb)  
* **Documento di Riferimento:** [Second_Assignment_Machine_Learning (1).pdf](Second_Assignment_Machine_Learning%20%281%29.pdf)

### Obiettivo
Analizzare l'efficacia di due diverse tecniche di riduzione della dimensionalità: una lineare (**PCA**) e una non lineare (**Autoencoder** neurale denso), applicate rispettivamente a problemi di Regressione, Clustering e Classificazione.

### Parte 1: Regressione (Diamonds Dataset)
Prevedere il prezzo dei diamanti a partire dalle feature fisiche e qualitative (9 feature in totale, ridotte a **3 dimensioni** latenti).
* **Architettura Autoencoder:** Input (9) -> Dense Relu (3) -> Dense Linear (9).
* **Modello predittivo:** Rete neurale FNN (64 relu -> 32 relu -> 1 lineare).

| Feature Set | Dimensionalità | RMSE ($) | R² Score |
| :--- | :---: | :---: | :---: |
| **Originale** | 9 | **592.01** | **0.9780** |
| **PCA** | 3 | 1578.82 | 0.8432 |
| **Autoencoder** | 3 | 1613.60 | 0.8362 |

*La riduzione a sole 3 feature comporta una fisiologica perdita di performance, ma PCA e Autoencoder offrono risultati estremamente vicini e confrontabili ($R^2 \approx 84\%$), con una leggera superiorità per la PCA.*

### Parte 2: Clustering (Wine Quality Dataset)
Raggruppare i vini bianchi in $K = 6$ cluster basandosi sulle caratteristiche chimico-fisiche (11 feature totali, ridotte a **2 dimensioni** per la visualizzazione).
* **Algoritmo:** K-Means ($K=6$).
* **Metriche:** Valutazione tramite Silhouette Score.

| Feature Set | Dimensionalità | Silhouette Score |
| :--- | :---: | :---: |
| **Originale** | 11 | 0.1478 |
| **PCA** | 2 | 0.3352 |
| **Autoencoder** | 2 | **0.3445** |

*La rimozione del rumore e la compressione operata dalle tecniche di riduzione aumentano significativamente la compattezza dei cluster, portando il Silhouette Score da 0.14 ad oltre 0.33. L'Autoencoder denso ottiene il risultato migliore.*

### Parte 3: Classificazione (GTZAN Spectrograms)
Classificare le immagini degli spettrogrammi (risoluzione ridotta a $64 \times 64 = 4096$ pixel) in 10 generi musicali.
* **Riduzione:** Feature compresse da 4096 a **128 dimensioni**.
* **Modelli confrontati:**
  1. **CNN Originale:** Rete convoluzionale applicata direttamente sulle immagini originali ($64 \times 64 \times 1$).
  2. **Dense PCA:** Rete densa FNN (128 -> Dropout 0.3 -> 64 -> 10) su feature compresse con PCA.
  3. **Dense Autoencoder:** Rete densa FNN (128 -> Dropout 0.3 -> 64 -> 10) su feature compresse con un Autoencoder denso (4096 -> 128 -> 4096).

| Modello | Metodo di Compressione | Accuratezza sul Test Set |
| :--- | :--- | :---: |
| **CNN** | Nessuno (Immagini $64 \times 64$) | **0.5400** |
| **FNN + PCA** | PCA (128 componenti) | 0.4300 |
| **FNN + Autoencoder**| Autoencoder (128 unità latenti) | 0.2200 |

*La CNN ottiene le prestazioni migliori catturando le relazioni spaziali intrinseche degli spettrogrammi. Nello spazio ridotto, la compressione lineare della PCA si rivela molto più efficace rispetto a quella dell'Autoencoder denso per la separabilità delle classi (43% vs 22%).*

---

## Struttura del Progetto
```text
.
├── ML_Assignment_1_Generi_Musicali.ipynb  # Notebook Primo Assignment
├── second_assignment.ipynb                # Notebook Secondo Assignment
├── First Assignment.pdf                   # Traccia del Primo Assignment
├── Second_Assignment_Machine_Learning (1).pdf # Traccia del Secondo Assignment
└── README.md                              # Documentazione del Progetto
```

---

## Installazione e Utilizzo

### Requisiti
Per eseguire i notebook è necessario installare Python 3.8+ e le dipendenze richieste. Puoi installarle tramite pip:

```bash
pip install pandas numpy scikit-learn tensorflow opencv-python seaborn matplotlib pypdf
```

### Esecuzione
1. Clona il repository:
   ```bash
   git clone https://github.com/tonyrub3/ML-University-Projects.git
   cd ML-University-Projects
   ```
2. Avvia Jupyter Notebook o JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Apri ed esegui i notebook `ML_Assignment_1_Generi_Musicali.ipynb` o `second_assignment.ipynb`.

*(Nota: Alcune sezioni relative a modelli di Deep Learning pesanti sono state originariamente configurate per l'esecuzione in ambiente Google Colab con accelerazione GPU).*
