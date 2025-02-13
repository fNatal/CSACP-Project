# -*- coding: utf-8 -*-
"""
Created on Sun jan  29 20:19:28 2025

@author: dinit
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load the dataset (replace this with your own dataset file path if needed)
file_path = 'BPD_Part_1_Victim_Based_Crime_Data_preprocessed.csv'
data = pd.read_csv(file_path)

# Separate text and structured columns
text_columns = ['Description']  # Columns that will be used for BERT
structured_columns = ['CrimeCode', 'Location', 'Weapon', 'District', 'Neighborhood', 'Latitude', 'Longitude']  # Structured data columns

# Split dataset into text data and structured data
text_data = data[text_columns]
structured_data = data[structured_columns]

# Encode the target variable (replace with your own target)
target = 'District'
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Split the dataset into train and test
X_train_text, X_test_text, X_train_structured, X_test_structured, y_train, y_test = train_test_split(
    text_data, structured_data, data[target], test_size=0.2, random_state=42
)

# Initialize DistilBERT Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Function to tokenize and encode text data using DistilBERT
def encode_text(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Tokenize and encode the text data
X_train_text_encoded = encode_text(X_train_text['Description'].tolist())
X_test_text_encoded = encode_text(X_test_text['Description'].tolist())

# Extract embeddings from DistilBERT
def get_distilbert_embeddings(encoded_texts):
    with torch.no_grad():
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        outputs = model(**encoded_texts)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Taking mean of token embeddings
    return embeddings.numpy()

# Get embeddings for training and testing text data
X_train_text_embeddings = get_distilbert_embeddings(X_train_text_encoded)
X_test_text_embeddings = get_distilbert_embeddings(X_test_text_encoded)

# Preprocessing for structured data: Impute missing values and scale numerical features
numeric_features = ['Latitude', 'Longitude']
categorical_features = ['CrimeCode', 'Location', 'Weapon', 'District', 'Neighborhood']

# Imputation and scaling for numeric features, encoding for categorical features
numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), LabelEncoder())

# Full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess structured data
X_train_structured_preprocessed = preprocessor.fit_transform(X_train_structured)
X_test_structured_preprocessed = preprocessor.transform(X_test_structured)

# Combine text and structured features
X_train_combined = np.hstack((X_train_text_embeddings, X_train_structured_preprocessed))
X_test_combined = np.hstack((X_test_text_embeddings, X_test_structured_preprocessed))

# Train Random Forest model on combined features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_combined, y_train)

# Evaluate the model
accuracy = rf_model.score(X_test_combined, y_test)
print(f'Model Accuracy: {accuracy}')


