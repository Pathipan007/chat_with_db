import json
import requests
import time
import chromadb
from sentence_transformers import SentenceTransformer

# ฟังก์ชันแปลงเวลาจากวินาทีเป็นนาที+วินาที
def format_time(seconds):
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f} วินาที"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"

# ฟังก์ชันเรียก Ollama API
def query_ollama(prompt, error_log, question_id, question):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "stream": False
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        generation_time = end_time - start_time
        return response.json()['response'].strip(), generation_time
    except Exception as e:
        print(f"Error for Question ID {question_id}: {e}")
        error_log.append({"question_id": question_id, "question": question, "error": str(e)})
        return "", -1

# โหลดโมเดล embedding
embedding_model = SentenceTransformer('BAAI/bge-m3')

# กำหนด path สำหรับ Vector Store
persist_directory = "vector_store/"
client = chromadb.PersistentClient(path=persist_directory)

# ทดสอบการเชื่อมต่อกับ Ollama API
response, generation_time = query_ollama("Test prompt", [], "test_question_id", "Test question")
print("Ollama Response:", response)
print("Generation Time:", format_time(generation_time))

# รับ Train set collection
train_collection_name = "bird_train_rag_bge_m3_v2"
train_collection = client.get_collection(train_collection_name)

# ฟังก์ชัน Similarity Search (Top N)
def perform_similarity_search(query_text, top_n=10):
    query_embedding = embedding_model.encode([query_text]).tolist()[0]
    results = train_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    languages = [meta["language"] for meta in results["metadatas"][0]]  # ดึง language จาก metadata
    return results["documents"][0], results["metadatas"][0], results["distances"][0], languages

# ฟังก์ชันคัดกรองด้วย LLM เพื่อเลือก Top K
def rerank_with_llm(original_query, documents, metadatas, top_k=3):
    candidates = "\n".join([
        f"Candidate {i+1}:\nEnglish: {meta['question_eng']}\nThai: {meta['question_th']}\nSQL: {meta['SQL']}\nTable: {meta['table']}"
        for i, meta in enumerate(metadatas)
    ])
    
    prompt = f"""You are an expert in evaluating the semantic similarity between natural language questions. 
The original question may be in either English or Thai. Your task is to rank the candidate questions based on their semantic similarity to the original question and select the top {top_k} most similar ones. 
Focus on the meaning and intent of the questions, considering both English and Thai versions if available.

Original Question: {original_query}

Candidate Questions:
{candidates}

### Instructions:
- Rank the candidates based on semantic similarity to the original question.
- Output only the indices of the top {top_k} candidates (0-based index), separated by commas.
- Example output: 0, 1, 2
- Do not include explanations or additional text."""

    error_log = []
    response, _ = query_ollama(prompt, error_log, "rerank", original_query)
    
    if not response or error_log:
        print("Error in LLM reranking, falling back to original ranking.")
        return list(range(top_k))
    
    try:
        top_indices = [int(idx.strip()) for idx in response.split(",") if idx.strip().isdigit()]
        
        # กรอง index ที่อยู่นอกช่วงของ metadatas
        top_indices = [idx for idx in top_indices if 0 <= idx < len(metadatas)]

        # ถ้าได้ไม่ครบ top_k ให้เติม index อื่น ๆ ที่ยังไม่ถูกเลือก
        if len(top_indices) < top_k:
            fallback_indices = [i for i in range(len(metadatas)) if i not in top_indices]
            top_indices += fallback_indices[:top_k - len(top_indices)]

        top_indices = top_indices[:top_k]
    except:
        print("Invalid LLM response, falling back to original ranking.")
        top_indices = list(range(min(top_k, len(metadatas))))
    
    return top_indices

# โหลด Dev set
dev_file = 'bird/data/dev/dev_j2c2j.json'
with open(dev_file, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# ประมวลผล Dev set
overall_start_time = time.time()
results = []

for i, item in enumerate(dev_data[:3]):  # ทดสอบ 3 คำถามแรก
    question_id = item['question_id']
    question = item['question']
    question_th = item.get('question_th', '')
    difficulty = item.get('difficulty', 'N/A')

    # เลือกคำถามที่ใช้ (ถ้ามี question_th ใช้ก่อน ถ้าไม่มีใช้ question)
    query_text = question_th if question_th else question

    # Similarity Search (Top N)
    print(f"\n=== Processing Question ID: {question_id} ===")
    print(f"Original Query: {query_text}")
    documents, metadatas, distances, languages = perform_similarity_search(query_text, top_n=10)

    print("\nInitial Top 10 Results (Before LLM Reranking):")
    for j, (doc, meta, dist, lang) in enumerate(zip(documents, metadatas, distances, languages)):
        print(f"Result {j+1}:")
        print(f"Similarity Score: {1 - dist:.4f}")
        print(f"Language: {lang}")
        print(f"English: {meta['question_eng']}")
        print(f"Thai: {meta['question_th']}")
        print(f"SQL: {meta['SQL']}")
        print(f"Table: {meta['table']}")
        print("---")

    # คัดกรองด้วย LLM เพื่อเลือก Top K
    top_k = 3
    top_indices = rerank_with_llm(query_text, documents, metadatas, top_k=top_k)
    
    print(f"\n\n\ncheck: {top_indices}\n\n\n")
    print(f"\nTop {top_k} Results (After LLM Reranking):")
    for j, idx in enumerate(top_indices):
        meta = metadatas[idx]
        lang = languages[idx]
        print(f"Result {j+1}:")
        print(f"Language: {lang}")
        print(f"English: {meta['question_eng']}")
        print(f"Thai: {meta['question_th']}")
        print(f"SQL: {meta['SQL']}")
        print(f"Table: {meta['table']}")
        print("---")

    results.append({
        "question_id": question_id,
        "original_query": query_text,
        "top_k_results": [
            {
                "language": languages[idx],
                "english": metadatas[idx]['question_eng'],
                "thai": metadatas[idx]['question_th'],
                "sql": metadatas[idx]['SQL'],
                "table": metadatas[idx]['table']
            }
            for idx in top_indices
        ]
    })


# บันทึกผลลัพธ์
with open('bird/exp_result/reranked_test.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\n=== Overall Processing Time: {format_time(overall_time)} ===")