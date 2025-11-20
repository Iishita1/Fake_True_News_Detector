# 📰 Fake News Detector

A robust NLP + ML pipeline to classify news articles as Fake or True using a hybrid approach:
- TF-IDF features (uni/bi/tri-grams)
- Handcrafted features (lexical, stylistic, sentiment via TextBlob)
- Logistic Regression classifier
- Interactive CLI for quick testing

> Built to help analyze and mitigate misinformation by combining semantic and stylistic cues.

---

## 📂 Project Structure

The project directory is organized as follows:

```text
News_Detector/
├─ Datasets/
│  ├─ Fake.csv
│  └─ True.csv
├─ FakeNewsPredictor.ipynb
├─ fake_news_dataset.csv
├─ README.md
└─ .gitattributes
```

* **`Datasets/Fake.csv`**, **`Datasets/True.csv`**: Core labeled datasets.
* **`FakeNewsPredictor.ipynb`**: End-to-end workflow (**EDA → features → model → evaluation → interactive predictor**).
  
---

## 🚀 Features

- Hybrid feature engineering:
  - TF-IDF with n-grams up to 3
  - Manual features: word count, avg word length, caps ratio, exclamations, subjectivity, polarity
- Careful text cleaning to avoid source leakage (e.g., removing patterns like “(Reuters) - ...”)
- Train/test split with reproducibility
- Confusion matrix and detailed classification report
- Interactive predictor with confidence bands and contextual reasoning

---

## 🧠 Model & Approach

- Vectorization: `TfidfVectorizer(max_features=8000, ngram_range=(1,3), stop_words='english', sublinear_tf=True)`
- Manual features: via a custom `ManualFeatureExtractor`
- Feature union: `FeatureUnion([tfidf, manual])`
- Scaling: `StandardScaler(with_mean=False)` to support sparse matrices
- Classifier: `LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')`

---

## 📊 Results

- Test Accuracy: ~99.47%
- Evaluation: Accuracy, Precision/Recall/F1 (per class), Confusion Matrix
- Insight: Hybrid features capture both semantic content and stylistic/sentiment signal that helps distinguish fake from true articles.

Note: Reported metrics are based on the given dataset and current preprocessing choices.

---

## 🛠️ Setup

### 1) Create environment
```bash
# Option A: venv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# Option B: conda
conda create -n news-detector python=3.10 -y
conda activate news-detector
```
### 2) Install dependencies
```bash
pip install -U pip
pip install pandas numpy requests beautifulsoup4 matplotlib seaborn scikit-learn nltk textblob
```
### 3) NLTK & TextBlob data (first run may auto-download)
The notebook already attempts:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
```

  ## ▶️ Usage

### Option A: Run the notebook

1.  Open **`FakeNewsPredictor.ipynb`** in Jupyter or VS Code.
2.  Run cells top-to-bottom.
3.  The notebook will:
    * Load datasets (by default via URLs; it also works with local CSVs)
    * Perform **EDA** (Exploratory Data Analysis)
    * Train the pipeline
    * Evaluate the model
    * Launch an interactive predictor

### Option B: Interactive predictor (from notebook)

At the end of the notebook, use the prompt shown:

```text
📰 --- FAKE NEWS PREDICTOR (ENHANCED) --- 📰
Paste the BODY text of the article below.
Paste text or type exit to end.
```
  
  Output includes **prediction band** and **confidence** with brief reasoning.

---

## 📁 Data

**Provided:**

* `Datasets/Fake.csv`
* `Datasets/True.csv`

**Columns used:**

* **`text`** (main body), **`title`** (fallback), and **`label`** (0 = True, 1 = Fake; assigned in notebook)

**Preprocessing:**

* Fill missing `text`/`title` with empty strings
* Shuffle data for unbiased splits
* Boilerplate/source removal to reduce leakage

**Note:** If using alternative datasets, ensure a similar schema with a **`text`** field.

---

## 📈 EDA Highlights

* **Class distribution visualization** (True vs Fake classes)
* **Article length** (word count) histograms by class
* Observations on **stylistic patterns** (e.g., capitalization, punctuation), **sentiment tendencies**

---

## 🧩 Extending the Project

Here are several ways to extend and improve the project:

* **Swap Classifier:** Experiment with different models (e.g., Linear SVM, Logistic Regression with different $C$/penalty).
* **Hyperparameter Search:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize current model parameters.
* **Add Features:**
    * **Readability indices** (e.g., Flesch–Kincaid)
    * **POS tag distributions** (Part-of-Speech tags)
    * **Named entity ratios**
* **Robustness Checks:**
    * **Cross-domain validation**
    * **Train/dev/test split with time-awareness** (critical for time-series data)
* **Deployment:**
    * Export the trained pipeline via **`joblib`**.
    * Wrap with a minimal API (**FastAPI**) or a UI (**Streamlit/Gradio**).

---

## ⚠️ Notes & Limitations

* High accuracy can reflect dataset characteristics and **potential residual biases**.
* Real-world performance depends on **domain shift**, writing styles, and **adversarial content**.
* Use as a **decision aid**, not a sole arbiter of truth.

---

## 📦 Requirements

* **Python** 3.9–3.11
* **Packages:**
    * `pandas`, `numpy`, `requests`, `beautifulsoup4`
    * `matplotlib`, `seaborn`
    * `scikit-learn`
    * `nltk`, `textblob`

---

## 🙌 Acknowledgments

* Dataset sources derived from public fake/true news corpora.
* Open-source libraries by the Python & ML community.
