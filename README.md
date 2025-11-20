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

## Systemskizze![Systemskizze](https://github.com/KOLLGO/TargetDetector/blob/main/system_sketch.svg)

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
  python training.py <path_to_csv>.csv <path_to_model_folder>
  ```
  
  - nutzt die `<path_to_csv>.csv` zum Trainieren des Modells (Separator ist `;`, kann in [preprocessing.py](https://github.com/KOLLGO/TargetDetector/blob/0bda2151fd0eb0ffeaa5cd800153949593e59257/preprocessing.py) `data_handling()` angepasst werden)
  
  - teilt die Daten im Verhältnis 80%-20% in Training- und Testdaten (anpassbar in  [preprocessing.py](https://github.com/KOLLGO/TargetDetector/blob/0bda2151fd0eb0ffeaa5cd800153949593e59257/preprocessing.py) `split_data()`)
  
  - speichert das Modell unter `<path_to_model_folder>/model.joblib`
  
  - speichert den Vectorizer unter `<path_to_model_folder>/tfidf_vectorizer.pkl`
  
  - speichert die Feature-Struktur unter `<path_to_model_folder>/feature_names.pkl`
  
  - evaluiert das Modell und speichert die Ergebnisse unter `<path_to_model_folder>/evaluation.txt`

- Modell einsetzen
  
  ```bash
  python main.py <path_to_model_folder> <path_to_test_data>.csv
  ```
  
  - nutzt das vorher trainierte Modell mit  Vectorizer und Feature-Struktur, um die Daten in `path_to_test_data.csv` zu klassifizieren (Separator ist `;`, kann in [preprocessing.py](https://github.com/KOLLGO/TargetDetector/blob/0bda2151fd0eb0ffeaa5cd800153949593e59257/preprocessing.py) `data_handling()` angepasst werden)
  
  - speichert das Ergebnis in `TargetDetector/outputs/<datetime>.csv`
