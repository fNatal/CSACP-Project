# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:03:34 2025

@author: dinit
"""

import spacy
import en_core_web_sm
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Reports from External Files
def load_reports(file_path):
    with open(file_path, 'r') as file:
        report = file.read()
    return report

# Step 2: Preprocess the Report Text
def preprocess_report(report):
    tokens = word_tokenize(report.lower())
    stopwords = set(['the', 'is', 'a', 'an', 'and', 'in', 'to', 'of'])
    tokens = [word for word in tokens if word not in stopwords]
    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc]
    return ' '.join(tokens)

# Step 3: Coreference Resolution
def resolve_coreferences(report):
    coref_model = pipeline('coreference-resolution', model="neuralmind/bert-large-uncased-finetuned-squad")
    resolved_report = coref_model(report)
    return resolved_report

# Step 4: NER and Entity Encoding
def extract_entities(report):
    nlp = en_core_web_sm.load()
    doc = nlp(report)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Example: Encode entities (e.g., suspects and locations) into structured data
    entity_dict = {}
    for ent in entities:
        if ent[1] == "PERSON":
            entity_dict["suspect"] = ent[0]
        elif ent[1] == "GPE":
            entity_dict["location"] = ent[0]
        elif ent[1] == "DATE":
            entity_dict["date"] = ent[0]
    
    return entity_dict

# Step 5: Vector Space Representation (TF-IDF)
def vector_space_representation(report):
    tfidf_vectorizer = TfidfVectorizer()
    vector = tfidf_vectorizer.fit_transform([report])
    return vector

# Step 6: Word Embeddings using Word2Vec
def generate_word_embeddings(report):
    tokens = word_tokenize(report.lower())
    model = Word2Vec([tokens], min_count=1)
    embeddings = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    return embeddings

# Step 7: Combine Features for ML Model Input
def prepare_data_for_model(report):
    # Preprocess report
    preprocessed_report = preprocess_report(report)
    
    # Coreference resolution
    resolved_report = resolve_coreferences(preprocessed_report)
    
    # Extract entities
    entities = extract_entities(resolved_report)
    
    # Vectorization (TF-IDF and Word Embeddings)
    vector_rep = vector_space_representation(resolved_report)
    word_embeddings = generate_word_embeddings(resolved_report)
    
    # Combine features into a final feature vector
    feature_vector = np.concatenate([vector_rep.toarray().flatten(), word_embeddings])
    
    # Convert entities into numerical features (simple encoding)
    entity_features = [LabelEncoder().fit_transform([value])[0] for key, value in entities.items()]
    
    # Combine text and entity features
    final_feature_vector = np.concatenate([feature_vector, entity_features])
    
    return final_feature_vector

# Main function to process report for ML model
def process_report_for_ml(file_path):
    # Load report
    report = load_reports(file_path)
    
    # Prepare the report for ML model processing
    feature_vector = prepare_data_for_model(report)
    
    # Return the processed feature vector that can be used in the ML model
    return feature_vector

# Example Usage
file_path = "example_report.txt"  # Path to your report file
feature_vector = process_report_for_ml(file_path)
print("Processed Feature Vector for ML Model:", feature_vector)
