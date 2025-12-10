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
  python training.py <path_to_csv>.csv <path_to_model_folder> <seed>
  ```
  
  - initialisiert den Zufallsgenerator mit `<seed>` für Reproduzierbarkeit der Ergebnisse
  
  - falls kein Seed angegeben wird, erfolgt Initialisierung zufällig (`-1`)
  
  - nutzt die `<path_to_csv>.csv` zum Trainieren des Modells (Separator ist `;`, kann in [preprocessing.py](https://github.com/KOLLGO/TargetDetector/blob/0bda2151fd0eb0ffeaa5cd800153949593e59257/preprocessing.py) `data_handling()` angepasst werden)
  
  - speichert den Vectorizer unter `<path_to_model_folder>/tfidf_vectorizer.pkl`
  
  - speichert die Feature-Struktur unter `<path_to_model_folder>/feature_names.pkl`
  
  - evaluiert das Modell und speichert die Ergebnisse unter `<path_to_model_folder>/training_results.txt`

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
| Fold 1 | SVC | Precision: 0.6636
Recall: 0.6635
F1 Score: 0.6615 |
|  | Logistische Regression | Precision: 0.6879
Recall: 0.6743
F1 Score: 0.6785 |
|  | Naive Bayes | Precision: 0.7130
Recall: 0.6247
F1 Score: 0.6523 |
|  | Random Forest | Precision: 0.6951
Recall: 0.6567
F1 Score: 0.6727 |
|  | Stacking CLF | Precision: 0.7450
Recall: 0.6512
F1 Score: 0.6810 |
| Fold 2 | SVC | Precision: 0.6523
Recall: 0.6523
F1 Score: 0.6480 |
|  | Logistische Regression | Precision: 0.6656
Recall: 0.6498
F1 Score: 0.6553 |
|  | Naive Bayes | Precision: 0.6941
Recall: 0.6131
F1 Score: 0.6376 |
|  | Random Forest | Precision: 0.6960
Recall: 0.6480
F1 Score: 0.6663 |
|  | Stacking CLF | Precision: 0.7269
Recall: 0.6313
F1 Score: 0.6596 |
| Fold 3 | SVC | Macro values for svc:
Precision: 0.6453
Recall: 0.6421
F1 Score: 0.6389 |
|  | Logistische Regression | Precision: 0.6749
Recall: 0.6545
F1 Score: 0.6625 |
|  | Naive Bayes | Precision: 0.6998
Recall: 0.6198
F1 Score: 0.6448 |
|  | Random Forest | Precision: 0.6911
Recall: 0.6517
F1 Score: 0.6673 |
|  | Stacking CLF | Precision: 0.7337
Recall: 0.6411
F1 Score: 0.6699 |
| Fold 4 | SVC | Precision: 0.6518
Recall: 0.6549
F1 Score: 0.6487 |
|  | Logistische Regression | Precision: 0.6781
Recall: 0.6582
F1 Score: 0.6655 |
|  | Naive Bayes | Precision: 0.6970
Recall: 0.6110
F1 Score: 0.6368 |
|  | Random Forest | Precision: 0.6912
Recall: 0.6424
F1 Score: 0.6618 |
|  | Stacking CLF | Precision: 0.7267
Recall: 0.6387
F1 Score: 0.6664 |
| Fold 5 | SVC | Precision: 0.6584
Recall: 0.6696
F1 Score: 0.6581 |
|  | Logistische Regression | Precision: 0.6763
Recall: 0.6603
F1 Score: 0.6657 |
|  | Naive Bayes | Precision: 0.6960
Recall: 0.6128
F1 Score: 0.6383 |
|  | Random Forest | Precision: 0.6891
Recall: 0.6480
F1 Score: 0.6642 |
|  | Stacking CLF | Precision: 0.7332
Recall: 0.6421
F1 Score: 0.6699 |
| Makro-AVG | SVC | Average precision: 0.6543 (+/- 0.0062)
Average recall: 0.6565 (+/- 0.0095)
Average f1 score: 0.6510 (+/- 0.0080) |
|  | Logistische Regression | Average precision: 0.6765 (+/- 0.0071)
Average recall: 0.6594 (+/- 0.0083)
Average f1 score: 0.6655 (+/- 0.0075) |
|  | Naive Bayes | Average precision: 0.7000 (+/- 0.0068)
Average recall: 0.6163 (+/- 0.0052)
Average f1 score: 0.6420 (+/- 0.0059) |
|  | Random Forest | Average precision: 0.6925 (+/- 0.0026)
Average recall: 0.6494 (+/- 0.0047)
Average f1 score: 0.6665 (+/- 0.0037) |
|  | Stacking CLF | Average precision: 0.7331 (+/- 0.0067)
Average recall: 0.6409 (+/- 0.0064)
Average f1 score: 0.6694 (+/- 0.0069) |
