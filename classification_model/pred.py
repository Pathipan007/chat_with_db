import json
import re
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# ---------- STEP 1: Load Model and Preprocessors ----------
model = joblib.load("resource/lgbm_smote_class2_model.joblib")
scaler = joblib.load("resource/scaler.joblib")
svd_question = joblib.load("resource/svd_question.joblib")
svd_sql = joblib.load("resource/svd_sql.joblib")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- STEP 2: Handcrafted Feature Extractor ----------
def extract_sql_features(sql):
    sql = sql.upper()
    features = {}
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
        'WITH', 'RANK', 'ROW_NUMBER', 'PARTITION BY'
    ]
    for keyword in keywords:
        features[f'count_{keyword.lower().replace(" ", "_")}'] = len(re.findall(r'\b' + keyword + r'\b', sql))
    features['has_subquery'] = 1 if re.search(r'\(.*?SELECT.*?\)', sql, flags=re.DOTALL) else 0
    subqueries = re.findall(r'\(([^()]*(SELECT)[^()]*)\)', sql)
    features['nested_query_depth'] = len(subqueries)
    features['word_count'] = len(sql.split())
    features['char_count'] = len(sql)
    table_matches = re.findall(r'\bFROM\s+(\w+)|JOIN\s+(\w+)', sql)
    tables = set([match[0] or match[1] for match in table_matches if match[0] or match[1]])
    features['table_count'] = len(tables)
    select_clause = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_clause:
        columns = select_clause.group(1).split(',')
        features['column_count'] = len([col for col in columns if col.strip()])
    else:
        features['column_count'] = 0
    features['has_distinct'] = 1 if 'DISTINCT' in sql else 0
    agg_funcs = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
    features['aggregation_count'] = sum(len(re.findall(r'\b' + func + r'\b', sql)) for func in agg_funcs)
    features['condition_count'] = len(re.findall(r'\b(AND|OR)\b', sql))
    features['has_case_when'] = 1 if re.search(r'CASE\s+WHEN', sql) else 0
    return features

# ---------- STEP 3: Load JSON File ----------
input_file = "../spider/data/train/train_spider.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

if not {'query', 'question'}.issubset(df.columns):
    raise ValueError("JSON ต้องมีคอลัมน์ 'SQL' และ 'question'")

# ---------- STEP 4: Feature Extraction ----------
print("Extracting handcrafted SQL features...")
feature_list = df["query"].apply(extract_sql_features)
feature_df = pd.DataFrame(feature_list.tolist())
feature_scaled = scaler.transform(feature_df)

print("Encoding and reducing question embeddings...")
question_embeddings = embedder.encode(df["question"].tolist(), normalize_embeddings=True)
question_reduced = svd_question.transform(question_embeddings)

print("Encoding and reducing SQL embeddings...")
sql_embeddings = embedder.encode(df["query"].tolist(), normalize_embeddings=True)
sql_reduced = svd_sql.transform(sql_embeddings)

X_combined = np.hstack([feature_scaled, question_reduced, sql_reduced])

# ---------- STEP 5: Predict ----------
print("Predicting difficulty levels...")
predictions = model.predict(X_combined)
label_map = {0: "simple", 1: "moderate", 2: "challenging"}
df["difficulty"] = [label_map[p] for p in predictions]

# ---------- STEP 5.1: Count difficulty distribution ----------
difficulty_counts = df["difficulty"].value_counts()
total = difficulty_counts.sum()

print("\nDifficulty distribution:")
for level in ["simple", "moderate", "challenging"]:
    count = difficulty_counts.get(level, 0)
    percent = (count / total) * 100
    print(f"{level}: {count} ({percent:.2f}%)")

# ---------- STEP 6: Save Results as JSON ----------
output_file = "test_smote_class2.json"
df.to_json(output_file, orient="records", force_ascii=False, indent=2)
print(f"\nPrediction complete! Results saved to: {output_file}")