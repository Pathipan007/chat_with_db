import json
from sentence_transformers import SentenceTransformer
import chromadb
import ast

def batch_add_to_collection(collection, ids, embeddings, documents, metadatas, batch_size=1000):
    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        print(f"[BATCH] Adding records {start} to {min(end, total)} / {total}")
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

# โหลด Train set
train_file = 'bird/data/train/train_bird_th.json'
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# โหลดโมเดล embedding (BAAI/bge-m3)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# กำหนด path
persist_directory = "vector_store/"

# ใช้ PersistentClient แบบใหม่
client = chromadb.PersistentClient(path=persist_directory)

collection_name = "bird_train_rag_para_v2"

# ลบ collection เดิมก่อน ถ้ามี
if collection_name in [col.name for col in client.list_collections()]:
    client.delete_collection(collection_name)
    print(f"[INFO] Deleted existing collection: {collection_name}")

collection = client.create_collection(collection_name)

all_ids = []
all_embeddings = []
all_documents = []
all_metadatas = []

for i, item in enumerate(train_data, start=1):
    print(f"Process: {i}/{len(train_data)}")
    question_id = item['question_id']
    question = item['question']
    question_th = item.get('question_th', '')
    sql = item.get('SQL', '')
    tables_raw = item.get('table', [])
    tables = []
    # print(f"[DEBUG] q{item.get('question_id')}: raw table = {tables_raw} | type = {type(tables_raw)}")

    if isinstance(tables_raw, str):
        try:
            parsed = ast.literal_eval(tables_raw)
            if isinstance(parsed, list):
                tables = [str(t) for t in parsed]
            else:
                tables = [str(parsed)]
        except (ValueError, SyntaxError):
            print(f"[WARNING] Failed to parse table for q{item.get('question_id')}: {tables_raw}")
            tables = [tables_raw]  # fallback เป็น string เดียว
    elif isinstance(tables_raw, list):
        tables = [str(t) for t in tables_raw]
    else:
        tables = [str(tables_raw)]
    # print(f"[DEBUG] parsed tables = {tables}")

    if question:
        all_ids.append(f"q{question_id}_en")
        all_embeddings.append(embedding_model.encode(question).tolist())
        all_documents.append(question)
        all_metadatas.append({
            "type": "question",
            "language": "en",
            "text": question,
            "question_th": question_th,
            "sql": sql,
            "table": ', '.join(tables) if tables else ''
        })

    if question_th:
        all_ids.append(f"q{question_id}_th")
        all_embeddings.append(embedding_model.encode(question_th).tolist())
        all_documents.append(question_th)
        all_metadatas.append({
            "type": "question_th",
            "language": "th",
            "text": question_th,
            "question": question,
            "sql": sql,
            "table": ', '.join(tables) if tables else ''
        })

batch_add_to_collection(collection, all_ids, all_embeddings, all_documents, all_metadatas)

print(f"Processed {len(train_data)} questions and added to Vector Store in collection '{collection_name}' at '{persist_directory}'")

# ตรวจสอบข้อมูลบางส่วนของ all_embeddings หลังจากที่ทำการ encode ข้อความ
print(f"First 5 embeddings: {all_embeddings[:5]}")

# ต่อมา คุณสามารถใช้ query เพื่อทดสอบว่าเวกเตอร์ที่เพิ่มเข้าไปถูกต้องหรือไม่
query_embedding = all_embeddings[0]  # เลือกเวกเตอร์ตัวแรกจาก all_embeddings

# ใช้ query เพื่อค้นหาเอกสารที่ตรงกับ query_embedding
results = collection.query(query_embeddings=[query_embedding], n_results=5)

# แสดงผลเอกสารที่ตรงกับ query
for result in results['documents']:
    print(result)