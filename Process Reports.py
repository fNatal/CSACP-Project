# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 02:30:47 2025

@author: dinit
"""

import spacy
import en_core_web_sm
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize

# Step 1: Load the Reports from External Files
def load_reports(file_path):
    with open(file_path, 'r') as file:
        report = file.read()
    return report

# Step 2: Preprocess the Report Text
def preprocess_report(report):
    # Tokenize text
    tokens = word_tokenize(report.lower())
    # Remove stopwords and punctuations
    stopwords = set(['the', 'is', 'a', 'an', 'and', 'in', 'to', 'of'])
    tokens = [word for word in tokens if word not in stopwords]
    # Lemmatize text (using spacy for simplicity)
    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc]
    return ' '.join(tokens)

# Step 3: Coreference Resolution
def resolve_coreferences(report):
    # Load pre-trained coreference resolution model
    coref_model = pipeline('coreference-resolution', model="neuralmind/bert-large-uncased-finetuned-squad")
    resolved_report = coref_model(report)
    return resolved_report

# Step 4: Vector Space Representation (TF-IDF)
def vector_space_representation(report):
    tfidf_vectorizer = TfidfVectorizer()
    vector = tfidf_vectorizer.fit_transform([report])
    return vector

# Step 5: Generate Word Embeddings using Word2Vec
def generate_word_embeddings(report):
    # Tokenize text
    tokens = word_tokenize(report.lower())
    # Use pre-trained Word2Vec or GloVe embeddings (here we use Gensim's Word2Vec model for simplicity)
    model = Word2Vec([tokens], min_count=1)
    embeddings = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    return embeddings

# Step 6: Analyse and Extract Information (NER and Relationships)
def analyze_report(report):
    # Named Entity Recognition using spaCy
    nlp = en_core_web_sm.load()
    doc = nlp(report)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract relationships (e.g., victim, weapon, location)
    relationships = extract_relationships(report)
    
    return entities, relationships

def extract_relationships(report):
    # This can be done by dependency parsing or rule-based methods.
    # For simplicity, return dummy relationships (to be replaced with a real method)
    relationships = [('Suspect', 'was at', 'Location')]
    return relationships

# Step 7: Summarization
def summarize_report(report):
    # Here you can use extractive or abstractive summarization
    # Using simple extractive approach for demo (Gensim Summarizer or HuggingFace transformer-based models)
    summarizer = pipeline("summarization")
    summary = summarizer(report, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Main Function to Execute the Process
def process_report(file_path):
    # Load the report from the external file
    report = load_reports(file_path)
    
    # Preprocess the report
    processed_report = preprocess_report(report)
    
    # Resolve coreferences
    resolved_report = resolve_coreferences(processed_report)
    
    # Convert report to vector space model representation
    vector_rep = vector_space_representation(resolved_report)
    
    # Generate word embeddings
    word_embeddings = generate_word_embeddings(resolved_report)
    
    # Analyse report for entities and relationships
    entities, relationships = analyze_report(resolved_report)
    
    # Generate summary of the report
    summary = summarize_report(resolved_report)
    
    # Output the results
    print("Entities:", entities)
    print("Relationships:", relationships)
    print("Summary:", summary)

# Example Usage: Process a report from a text file
file_path = "example_report.txt"  # Path to your report file
process_report(file_path)
