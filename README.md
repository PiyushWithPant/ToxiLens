# ToxiLens 🔎: AI-powered Toxicity Detector and Categorizer

A dual-head neural network that detects whether a comment is toxic and identifies the specific toxic categories. Model available at: [ToxiLens 🔎](https://toxilens.streamlit.app/)

<img alt="Model not optimized" src="https://img.shields.io/badge/Model Status-Not Optimized-red">

---

### Dataset 📦

The dataset used in ToxiLens is the Jigsaw Toxic Comment Classification dataset, which is available on Kaggle. This dataset contains a large collection of comments from various online platforms, labeled for toxicity. It is designed to help in the development of models that can detect and categorize toxic comments effectively.

You can access the dataset [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). 🧾

- **Total Comments**: Over 200,000
- **Labels**: Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate
- **Usage**: Ideal for training machine learning models for natural language processing tasks related to comment moderation and toxicity detection.

### Structure 📁

1. `/data`

- `/data/preprocessed data`: contains processed features derived from the Jigsaw Toxic Comment dataset (Kaggle, CC BY 4.0).

2. `src`

- `data.ipynb`: Data Preprocessing & TF-IDF Vectorization, saves the data
- `model.py`: Contains the main model
- `train.py`: Model Training
- `app.py`: Streamlit app

---

## Highlights ✨

- Dual-head architecture: one head for binary toxicity detection and another for multi-label category classification.
- Uses TF-IDF features (sparse) and a compact feed-forward network.
- Lightweight, reproducible training and a Streamlit demo for inference.

## Model (brief) 🧠

The network in `ToxicANN` (defined in `src/model.py`) uses a shared MLP backbone and two heads:

- Binary head: outputs probability of toxicity (1 output)
- Multi-label head: outputs probabilities for 6 categories

Activation: sigmoid on both heads. Loss: binary cross-entropy for both targets.

## Usage ▶️

1. Create and activate your Python environment:

```sh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Preprocess data

   - Run the notebook `src/data.ipynb` or your preprocessing script to generate TF-IDF features and label CSVs.
   - Ensure outputs are saved to `data/preprocessed data/`:
     - `X_tfidf_sparse.pkl`
     - `y_binary.csv`
     - `y_multi.csv`

3. Train
   - The training loop and configuration live in `src/model.py`. You can run training directly:

```sh
python src/train.py
```

4. Inspect/save model

   - Model checkpoint is saved to `model/toxilens.pth` after training.

5. Run demo
   - Launch the Streamlit app:

```sh
streamlit run src/app.py
```

- The app uses the saved TF-IDF vectorizer and model weights to make predictions on input text.

## Notes & Tips 🛠️

- Configurable parameters (e.g., INPUT_DIM, BATCH_SIZE, LR, EPOCHS) are set near the top of `src/model.py`.
- The dataset wrapper (`SparseDataset`) converts sparse TF-IDF rows to dense tensors in the DataLoader.
- For faster iteration on CPU, reduce batch size or INPUT_DIM.
- Save and version your `model/toxilens.pth` and preprocessing artifacts (`data/preprocessed data/`) to reproduce results.

## Files & Quick Links 🔗

- `src/model.py` — contains `ToxicANN`, `SparseDataset`
- `src/data.ipynb` — preprocessing & TF-IDF steps
- `src/train.py` — training helper (if present) and the main training/validation loop
- `src/app.py` — Streamlit demo
- `data/preprocessed data/` — saved features and labels
- `model/toxilens.pth` — saved model weights

## License 📜

This project is MIT licensed. See `LICENSE`.

---

By Piyush Pant (पियुष पंत)
