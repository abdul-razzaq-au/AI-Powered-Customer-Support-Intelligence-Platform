import joblib
import torch
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# =====================
# DistilBERT
# =====================

distilbert_path = "../../models/distilbert"


distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(
    distilbert_path
)


distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    distilbert_path
)


distilbert_model.eval()



ticket_type_encoder = joblib.load(
    "../../models/distilbert/label_encoder.pkl"
)



# =====================
# Priority Model
# =====================

priority_model = joblib.load(
    "../../models/priority/lgb_priority.pkl"
)


priority_label_encoder = joblib.load(
    "../../models/encoders/ticket_priority_label_encoder.pkl"
)



# =====================
# Regression Model (for time resolution)
# =====================

regression_model = joblib.load(
    "../../models/regression/xgboost_optuna_baseline.pkl"
)


# Feature encoders

le_gender = joblib.load(
    "../../models/encoders/le_gender.pkl"
)


le_product = joblib.load(
    "../../models/encoders/le_product.pkl"
)


le_channel = joblib.load(
    "../../models/encoders/le_channel.pkl"
)

# TF-IDF vectorizer
tfidf_vectorizer = joblib.load(
    "../../models/tfidf_vectorizer.pkl"
)



# Reference date
REFERENCE_DATE = pd.Timestamp(
    "2026-04-12"
)
print("All models loaded")