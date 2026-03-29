# AI-RESEARCH-PAPER-LITERATURE-ANALIYZER

## 📌 Project Overview
This project focuses on extracting and analyzing **healthcare-related research papers** from the ArXiv dataset and refining them using **BioBERT-based classification**.

The pipeline combines:
- Rule-based filtering (`q-bio`)
- Transformer-based classification (BioBERT)

---

## 🧠 What is BioBERT?
BioBERT is a pre-trained biomedical language model based on BERT, designed for:
- Biomedical text mining
- Scientific literature understanding
- Healthcare NLP tasks

---

## 📂 Dataset
- Source: ArXiv Metadata Dataset  
- File: `arxiv-metadata-oai-snapshot.json`  
- Format: JSON (line-by-line)

---

## ⚙️ Tech Stack
- Python 🐍  
- Pandas 📊  
- PyTorch 🔥  
- HuggingFace Transformers 🤗  
- Jupyter Notebook  

---

## 🚀 Features

### 🔹 1. Large Dataset Handling
- Process massive JSON data using chunking
- Memory-efficient approach

### 🔹 2. Rule-Based Filtering
- Extract papers using:
```python
df['categories'].str.contains('q-bio')
```
## 🔹 3. BioBERT Classification (Advanced)

- Classify abstracts into **healthcare / non-healthcare**
- Improves filtering accuracy beyond keyword-based methods

---

## 🔄 Workflow

### Step 1: Load Data
```python
chunks = pd.read_json(path, lines=True, chunksize=100000)
```
### Step 2: Filter Healthcare Category
```python
filtered = chunk[chunk['categories'].str.contains('q-bio', na=False)]
```
### Step 3: BioBERT Model Setup
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
```
### Step 4: Classification Function
```python
def classify_healthcare(text):
    inputs = tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction
```
### Step 5: Apply Model
```python
labels = []

for abstract in medical_df["abstract"].fillna(""):
    label = classify_healthcare(abstract)
    labels.append(label)

medical_df["healthcare_label"] = labels
```
### Step 6: Filter Final Healthcare Dataset
```python
healthcare_df = medical_df[
    medical_df["healthcare_label"] == 1
]
```
### Step 7: Save Final Dataset
```python
healthcare_df.to_csv("healthcare_papers.csv", index=False)
```
## 📊 Output

Final dataset contains:

- id
- title
- abstract
- author
- authors_parsed
- categories
- update_date
- journal-ref
- doi
- year
- healthcare_label
  
---
## 📈 Key Insight
- q-bio → Coarse filtering
- BioBERT → Semantic understanding
---
