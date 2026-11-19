# Emotion Detection in Moroccan Darija

Course project: fine-tune DarijaBERT to detect anger, fear, joy, and sadness in Moroccan Darija text, compared against lightweight baselines.

## Team roles

| Member | Responsibility |
|--------|----------------|
| Soulaymane Boulaich | `src/models/darija_bert/` — DarijaBERT fine-tuning, hyperparameter search, threshold tuning |
| Fares Ilham | `src/data_pipeline/` — HF load, `ary` filtering, normalization, splits; `src/models/second_encoder/` — AraBERT/XLM-R reference |
| Tasnim Touil | `src/models/baselines/` — TF-IDF, BiLSTM, majority baseline; `src/evaluation/` — metrics, charts, error analysis |
| All | `ui/app.py` — shared Streamlit interface |

## Repository structure

```
data/              # Fares Ilham — raw downloads, processed splits (gitignored)
src/data_pipeline/ # Fares Ilham
src/models/darija_bert/   # Soulaymane Boulaich
src/models/second_encoder/# Fares Ilham
src/models/baselines/     # Tasnim Touil
src/evaluation/           # Tasnim Touil
src/utils.py              # Shared
ui/app.py                 # Shared (merge last)
notebooks/                # Personal sandboxes
configs/experiment.yaml   # Shared — raw vs. normalized toggle, seeds, hyperparams
tests/                    # Shared
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Data

The dataset is **not committed** (large files). Download from Hugging Face:

https://huggingface.co/datasets/vgaraujov/semeval-2025-task11-track-a

Run the pipeline to keep only Moroccan examples (ids starting with `ary`) and build train/dev/test splits:

```bash
python -m src.data_pipeline.build_splits
```

## Training

DarijaBERT (Soulaymane):
```bash
python -m src.models.darija_bert.train
```

Second encoder (Fares Ilham):
```bash
python -m src.models.second_encoder.train
```

Baselines (Tasnim):
```bash
python -m src.models.baselines.tfidf_logreg
python -m src.models.baselines.bilstm
```

## Evaluation

```bash
python -m src.evaluation.metrics
python -m src.evaluation.charts
```

## UI

```bash
streamlit run ui/app.py
```

## Git workflow

- Work on feature branches: `feature/soulaymane-darija-bert`, `feature/fares-pipeline`, `feature/tasnim-eval`
- Open PRs to `main`; review before merging
- `ui/app.py` is merged last once at least one model is trained
- Do not edit inside another teammate's folder without asking