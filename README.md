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
  
  Zum Projekt wurde ein Paper verfasst, welches die wissenschaftlichen Hintergründe erklärt, auf denen die Software aufbaut.

## Systemskizze![Systemskizze](https://github.com/KOLLGO/TargetDetector/blob/e946c86b889a79b7f23a8469c48d9c6126fb40cd/system_sketch.png)

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
  
  - nutzt die `<INPUT>.csv` zum Trainieren des Modells (Separator ist `;`, kann in [preprocessing.py](https://github.com/KOLLGO/TargetDetector/blob/0bda2151fd0eb0ffeaa5cd800153949593e59257/preprocessing.py) `data_handling()` angepasst werden)
  
  - speichert den Vectorizer unter `<MODEL>/tfidf_vectorizer.pkl`
  
  - speichert die Feature-Struktur unter `<MODEL>/feature_names.pkl`
  
  - evaluiert das Modell und speichert die Ergebnisse unter `<MODEL>/training_results.txt`

- Das Modell zu speichern, um es dann einzusetzen wäre im Sinne einer Shared Task, entfällt aber aufgrund der Aufgabenstellung.

# Testergebnisse
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
