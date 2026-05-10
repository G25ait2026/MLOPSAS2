# MLOps Assignment 2: Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment

## Project Description

This project implements a complete MLOps workflow for fine-tuning a DistilBERT model on the UCSD Goodreads dataset to classify book reviews into seven genres: poetry, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, and young adult. The workflow covers modular Python scripts, experiment tracking with Weights & Biases, model publishing to the Hugging Face Hub, and reproducible results via GitHub.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the root directory:
```text
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
```

### 4. Run the pipeline

**Download and prepare data:**
```bash
python data.py
```

**Train the model (logs to W&B automatically):**
```bash
python train.py
```

**Evaluate and upload results to W&B:**
```bash
python eval.py
```

**Push model to Hugging Face Hub:**
```python
from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

login(token="your_hf_token")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-goodreads-genres")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-goodreads-genres")
model.push_to_hub("your-username/distilbert-goodreads-genres")
tokenizer.push_to_hub("your-username/distilbert-goodreads-genres")
```

---

## Project Structure

```
.
├── data.py          # Data loading, sampling, train/test split, encoding
├── train.py         # Model loading, Trainer setup, training loop
├── eval.py          # Evaluation, metrics, saving results
├── utils.py         # Shared helpers: label maps, dataset class, compute_metrics
├── requirements.txt # Python dependencies
└── README.md
```

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.54  |
| F1 Score  | 0.54  |
| Eval Loss | 1.24  |

> Fill in the scores after running training and evaluation.

---

## Links

- Hugging Face model: https://huggingface.co/charantejpeteti/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/g25ait2026-charantej/distilbert-goodreads-genres
