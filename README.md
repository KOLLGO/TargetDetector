# TargetDetector

Das Programm trainiert ein Modell, um die Zielgruppe von Texten zwischen Individual, Group und Public zu klassifizieren.  

# Übersicht

## Besonderheiten

- Entstehung
  
  Das Projekt TargetDetector wurde im Rahmen des Softwareprojekts des Studiengangs "Allgemeine und digitale Forensik" als Shared Task programmiert. Im Projekt wurde ein Datensatz mit Twitter-Beiträgen (Tweets) verwendet. In diesem werden Erwähnungen und Nennungen von Namen von Gruppen (inkl. Unternehmen) und Individuen anonymisiert:
  
  | Type       | Replacement |
  | ---------- | ----------- |
  | Individual | `@IND`      |
  | Group      | `@GRP`      |
  | Police     | `@POL`      |
  | Press      | `@PRE`      |

- Paper
  
  Zum Projekt wurde ein [Paper](Paper_Target_Detection.pdf) verfasst, welches die wissenschaftlichen Hintergründe erklärt, auf denen die Software aufbaut.

- Version
  
  geschrieben und getestet in Python 3.13

## Systemskizze

![Systemskizze](https://github.com/KOLLGO/TargetDetector/blob/e946c86b889a79b7f23a8469c48d9c6126fb40cd/system_sketch.png)

# Setup

```bash
git clone https://github.com/KOLLGO/TargetDetector.git
```

```bash
pip install -r requirements.txt
```

# Nutzung

- Modell trainieren
  
  ```bash
  python main.py -h
  ```
  ```text
  usage: main.py [-h] -i INPUT -m MODEL [-s SEED] [-c CORES]

  options:
    -h, --help         show this help message and exit
    -i, --input INPUT  Path to the CSV data file
    -m, --model MODEL  Folder to save the model components and results
    -s, --seed SEED    Seed for reproducibibility (default: None for random seed)
    -c, --cores CORES  Number of CPU cores to use (default: -1 for all available cores)
  ```
  
  - initialisiert den Zufallsgenerator mit `<SEED>` für Reproduzierbarkeit der Ergebnisse
  
  - falls kein Seed angegeben wird, erfolgt Initialisierung zufällig (`None`)
  
  - nutzt die `<INPUT>.csv` zum Trainieren des Modells (Separator ist `;`, kann in [preprocessing.py](preprocessing.py) `data_handling()` angepasst werden)
  
  - speichert den Vectorizer unter `<MODEL>/tfidf_vectorizer.pkl`
  
  - evaluiert das Modell und speichert die Ergebnisse unter `<MODEL>/training_results.txt`

- Das Modell zu speichern, um es dann einzusetzen wäre im Sinne einer Shared Task, entfällt aber aufgrund der Aufgabenstellung.

# Testergebnisse
## Finaler Test
Der beste Test des Modells wies folgende Werte auf:
| Komponente | Wert | Ergebnis |
| --- | --- | --- |
| Notizen | ohne | ich, mir, mein, meine, meiner, meines, meinen, meinem |
| Größe Datensatz | full | 14470 |
| TF-IDF Vectorizer | Dimensionalität | 10.000 (10k) → top 10k |
|  | min. Vorkommen eines Wortes | 1 |
|  | sublinear | logarithmische Skalierung |
|  | Normalisierung | L2 |
| Seed | 42 |  |
| Grad Oversampling | kein Oversampling |  |
| Inner k (für k-Fold-CV) | 3 |  |
| Outer k (für k-Fold-CV) | 5 |  |
| ———————— | Ergebnisse | ——————————————— |
| Fold 1 | SVC | Precision: 0.6636<br>Recall: 0.6635<br>F1 Score: 0.6615 |
|  | Logistische Regression | Precision: 0.6879<br>Recall: 0.6743<br>F1 Score: 0.6785 |
|  | Naive Bayes | Precision: 0.7130<br>Recall: 0.6247<br>F1 Score: 0.6523 |
|  | Random Forest | Precision: 0.6951<br>Recall: 0.6567<br>F1 Score: 0.6727 |
|  | Stacking CLF | Precision: 0.7450<br>Recall: 0.6512<br>F1 Score: 0.6810 |
| Fold 2 | SVC | Precision: 0.6523<br>Recall: 0.6523<br>F1 Score: 0.6480 |
|  | Logistische Regression | Precision: 0.6656<br>Recall: 0.6498<br>F1 Score: 0.6553 |
|  | Naive Bayes | Precision: 0.6941<br>Recall: 0.6131<br>F1 Score: 0.6376 |
|  | Random Forest | Precision: 0.6960<br>Recall: 0.6480<br>F1 Score: 0.6663 |
|  | Stacking CLF | Precision: 0.7269<br>Recall: 0.6313<br>F1 Score: 0.6596 |
| Fold 3 | SVC | Precision: 0.6453<br>Recall: 0.6421<br>F1 Score: 0.6389 |
|  | Logistische Regression | Precision: 0.6749<br>Recall: 0.6545<br>F1 Score: 0.6625 |
|  | Naive Bayes | Precision: 0.6998<br>Recall: 0.6198<br>F1 Score: 0.6448 |
|  | Random Forest | Precision: 0.6911<br>Recall: 0.6517<br>F1 Score: 0.6673 |
|  | Stacking CLF | Precision: 0.7337<br>Recall: 0.6411<br>F1 Score: 0.6699 |
| Fold 4 | SVC | Precision: 0.6518<br>Recall: 0.6549<br>F1 Score: 0.6487 |
|  | Logistische Regression | Precision: 0.6781<br>Recall: 0.6582<br>F1 Score: 0.6655 |
|  | Naive Bayes | Precision: 0.6970<br>Recall: 0.6110<br>F1 Score: 0.6368 |
|  | Random Forest | Precision: 0.6912<br>Recall: 0.6424<br>F1 Score: 0.6618 |
|  | Stacking CLF | Precision: 0.7267<br>Recall: 0.6387<br>F1 Score: 0.6664 |
| Fold 5 | SVC | Precision: 0.6584<br>Recall: 0.6696<br>F1 Score: 0.6581 |
|  | Logistische Regression | Precision: 0.6763<br>Recall: 0.6603<br>F1 Score: 0.6657 |
|  | Naive Bayes | Precision: 0.6960<br>Recall: 0.6128<br>F1 Score: 0.6383 |
|  | Random Forest | Precision: 0.6891<br>Recall: 0.6480<br>F1 Score: 0.6642 |
|  | Stacking CLF | Precision: 0.7332<br>Recall: 0.6421<br>F1 Score: 0.6699 |
| Makro-AVG | SVC | Average precision: 0.6543 (+/- 0.0062)<br>Average recall: 0.6565 (+/- 0.0095)<br>Average f1 score: 0.6510 (+/- 0.0080) |
|  | Logistische Regression | Average precision: 0.6765 (+/- 0.0071)<br>Average recall: 0.6594 (+/- 0.0083)<br>Average f1 score: 0.6655 (+/- 0.0075) |
|  | Naive Bayes | Average precision: 0.7000 (+/- 0.0068)<br>Average recall: 0.6163 (+/- 0.0052)<br>Average f1 score: 0.6420 (+/- 0.0059) |
|  | Random Forest | Average precision: 0.6925 (+/- 0.0026)<br>Average recall: 0.6494 (+/- 0.0047)<br>Average f1 score: 0.6665 (+/- 0.0037) |
|  | Stacking CLF | Average precision: 0.7331 (+/- 0.0067)<br>Average recall: 0.6409 (+/- 0.0064)<br>Average f1 score: 0.6694 (+/- 0.0069) |

## Test ohne Preprocessing
| Komponente | Wert | Ergebnis |
| --- | --- | --- |
| Notizen | Ohne | ich, mir, mein, meine, meiner, meines, meinen, meinem<br>Preprocessing|
| Größe Datensatz | full | 14470 |
| TF-IDF Vectorizer | Dimensionalität | 10.000 (10k) → top 10k |
|  | min. Vorkommen eines Wortes | 1 |
|  | sublinear | logarithmische Skalierung |
|  | Normalisierung | L2 |
| Seed | 42 |  |
| Grad Oversampling | kein Oversampling |  |
| Inner k (für k-Fold-CV) | 3 |  |
| Outer k (für k-Fold-CV) | 5 |  |
| **————————** | **Ergebnisse** | **———————————————** |
| Fold 1 | SVC | Precision: 0.6851<br>Recall: 0.6490<br>F1 Score: 0.6642 |
|  | Logistische Regression | Precision: 0.6646<br>Recall: 0.6756<br>F1 Score: 0.6675 |
|  | Naive Bayes | Precision: 0.7216<br>Recall: 0.6270<br>F1 Score: 0.6569 |
|  | Random Forest | Precision: 0.6803<br>Recall: 0.6384<br>F1 Score: 0.6535 |
|  | Stacking CLF | Precision: 0.7413<br>Recall: 0.6336<br>F1 Score: 0.6634 |
| Fold 2 | SVC | Precision: 0.6706<br>Recall: 0.6541<br>F1 Score: 0.6575 |
|  | Logistische Regression | Precision: 0.6507<br>Recall: 0.6540<br>F1 Score: 0.6513 |
|  | Naive Bayes | Precision: 0.6943<br>Recall: 0.6105<br>F1 Score: 0.6369 |
|  | Random Forest | Precision: 0.6814<br>Recall: 0.6342<br>F1 Score: 0.6518 |
|  | Stacking CLF | Precision: 0.7369<br>Recall: 0.6302<br>F1 Score: 0.6613 |
| Fold 3 | SVC | Precision: 0.6415<br>Recall: 0.6479<br>F1 Score: 0.6400 |
|  | Logistische Regression | Precision: 0.6519<br>Recall: 0.6648<br>F1 Score: 0.6554 |
|  | Naive Bayes | Precision: 0.6938<br>Recall: 0.6119<br>F1 Score: 0.6385 |
|  | Random Forest | Precision: 0.6546<br>Recall: 0.6270<br>F1 Score: 0.6352 |
|  | Stacking CLF | Precision: 0.7322<br>Recall: 0.6219<br>F1 Score: 0.6529 |
| Fold 4 | SVC | Precision: 0.6544<br>Recall: 0.6622<br>F1 Score: 0.6536 |
|  | Logistische Regression | Precision: 0.6611<br>Recall: 0.6665<br>F1 Score: 0.6614 |
|  | Naive Bayes | Precision: 0.7032<br>Recall: 0.6216<br>F1 Score: 0.6487 |
|  | Random Forest | Precision: 0.6700<br>Recall: 0.6280<br>F1 Score: 0.6398 |
|  | Stacking CLF | Precision: 0.7263<br>Recall: 0.6263<br>F1 Score: 0.6574 |
| Fold 5 | SVC | Precision: 0.6621<br>Recall: 0.6321<br>F1 Score: 0.6449 |
|  | Logistische Regression | Precision: 0.6510<br>Recall: 0.6569<br>F1 Score: 0.6530 |
|  | Naive Bayes | Precision: 0.7064<br>Recall: 0.6280<br>F1 Score: 0.6536 |
|  | Random Forest | Precision: 0.6792<br>Recall: 0.6291<br>F1 Score: 0.6479 |
|  | Stacking CLF | Precision: 0.7256<br>Recall: 0.6295<br>F1 Score: 0.6584 |
| Makro-AVG | SVC | Average precision: 0.6627 (+/- 0.0147)<br>Average recall: 0.6491 (+/- 0.0099)<br>Average f1 score: 0.6520 (+/- 0.0087) |
|  | Logistische Regression | Average precision: 0.6559 (+/- 0.0058)<br>Average recall: 0.6636 (+/- 0.0076)<br>Average f1 score: 0.6577 (+/- 0.0060) |
|  | Naive Bayes | Average precision: 0.7039 (+/- 0.0101)<br>Average recall: 0.6198 (+/- 0.0073)<br>Average f1 score: 0.6469 (+/- 0.0080) |
|  | Random Forest | Average precision: 0.6731 (+/- 0.0101)<br>Average recall: 0.6313 (+/- 0.0043)<br>Average f1 score: 0.6456 (+/- 0.0070) |
|  | Stacking CLF | Average precision: 0.7325 (+/- 0.0061)<br>Average recall: 0.6283 (+/- 0.0040)<br>Average f1 score: 0.6587 (+/- 0.0036) |
