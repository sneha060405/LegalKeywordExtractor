# Legal Document Analyzer

## Project Overview
The Legal Document Analyzer is a Streamlit-based Natural Language Processing (NLP) application designed to process legal documents such as contracts, NDAs, and agreements. It extracts meaningful insights including keywords, summaries, clause detection, and risk analysis to assist in quick legal document understanding.

The system combines traditional NLP techniques with TF-IDF-based keyword extraction, sentence scoring, and rule-based legal analysis to provide structured insights from unstructured legal text.

---

## Features

### 1. Document Input
- Paste legal text directly
- Upload PDF files for automatic text extraction

### 2. Keyword Extraction
- TF-IDF based keyword extraction
- Supports unigram and bigram detection
- Filters irrelevant terms and noise

### 3. Smart Summary Generation
- Sentence scoring based on:
  - Keyword relevance
  - Legal action words
  - Sentence position
  - Text uniqueness
- Diversity filtering to avoid redundant sentences

### 4. Clause Detection
Automatically identifies key legal clauses such as:
- Termination clauses
- Payment terms
- Liability clauses
- Confidentiality clauses
- Obligations
- Dispute resolution clauses

### 5. Risk Analysis
Classifies risk-related phrases into:
- High risk
- Medium risk
- Low risk

Based on predefined legal risk patterns.

### 6. Visual Analytics
- Word cloud generation for document overview
- Keyword frequency bar chart
- Document statistics (words, sentences, unique words, etc.)

---

## Tech Stack

- Python
- Streamlit
- NLTK (Natural Language Toolkit)
- Scikit-learn (TF-IDF Vectorization)
- PyPDF2 (PDF text extraction)
- Matplotlib (visualizations)
- WordCloud
- Pandas
- Regular Expressions (Regex)

---

## Project Structure


LegalKeywordExtractor/
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## Installation and Setup

### 1. Clone the repository

git clone https://github.com/your-username/LegalKeywordExtractor.git

cd LegalKeywordExtractor


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the application

streamlit run app.py


---

## Requirements

Ensure the following dependencies are installed:


streamlit
nltk
scikit-learn
matplotlib
wordcloud
pandas
PyPDF2


---

## How It Works

1. The user uploads or pastes a legal document
2. The system cleans and preprocesses the text
3. TF-IDF is used to extract important keywords
4. Sentences are scored based on relevance and legal importance
5. Key clauses and risk phrases are identified using rule-based matching
6. Results are displayed with visualizations and structured summaries

---

## Future Improvements

- Integration with advanced NLP models (BERT-based summarization)
- Named Entity Recognition (NER) for legal entities
- Clause classification using machine learning
- Export results as PDF reports
- Multi-document comparison feature

---

## Author

Sneha Mahajan  
B.Tech ECE AIML Student  
Project focused on NLP-based legal document analysis

---

## License

This project is for educational purposes. You may modify and extend it for academic or research use.
