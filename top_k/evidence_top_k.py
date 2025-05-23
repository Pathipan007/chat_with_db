import json
import requests
import re
import time
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import math

# ใช้ tokenizer ของ google/gemma-3-12b-it
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

# สร้าง list สำหรับเก็บ log คำถามที่เกินโทเค็น
token_exceed_log = []

# สร้าง list สำหรับเก็บ log ข้อมูลโทเค็นทั้งหมด
token_log = []

# สร้าง dict สำหรับเก็บผลการเช็ค id
id_match_log = {"correct": 0, "incorrect": 0}

# ฟังก์ชันคำนวณ num_ctx แบบ dynamic
def calculate_dynamic_num_ctx(token_count, min_ctx=2048, max_ctx=8192, buffer=1.2):
    num_ctx = token_count * buffer
    if num_ctx <= min_ctx:
        return min_ctx
    n = math.ceil(math.log2(num_ctx))
    num_ctx = 2 ** n
    return max(min_ctx, min(max_ctx, num_ctx))

# ฟังก์ชันนับโทเค็น
def count_tokens(text, question_id, query, max_tokens=8192):
    token_count = len(tokenizer.encode(text))
    if token_count > max_tokens:
        token_exceed_log.append({
            "question_id": question_id,
            "query": query,
            "token_count": token_count,
            "max_tokens": max_tokens
        })
    return token_count

# ฟังก์ชันแปลงเวลาจากวินาทีเป็นชั่วโมง+นาที+วินาที
def format_time(seconds):
    if seconds < 0:  
        return "N/A"
    if seconds < 60:  
        return f"{seconds:.2f} วินาที"
    if seconds < 3600:  
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    remaining_seconds = remaining_seconds % 60
    return f"{hours} ชั่วโมง {minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"

# ฟังก์ชันเรียก Ollama API
def query_ollama(prompt, error_log, question_id, query, num_ctx):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx}
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if 'response' not in data:
            print(f"Ollama returned: {json.dumps(data, indent=2)}")
            raise ValueError("Invalid response format: 'response' key missing")
        end_time = time.time()
        generation_time = end_time - start_time
        return data['response'].strip(), generation_time
    except (requests.RequestException, ValueError) as e:
        print(f"Error for Question ID {question_id}: {e}")
        error_log.append({"question_id": question_id, "query": query, "error": str(e)})
        return "", -1

# โหลดโมเดล embedding
embedding_model = SentenceTransformer('BAAI/bge-m3')

# กำหนด path สำหรับ Vector Store
persist_directory = "../embed_and_vector_store/vector_store/"
client = chromadb.PersistentClient(path=persist_directory)

# ทดสอบการเชื่อมต่อกับ Ollama API
response, generation_time = query_ollama("Test prompt", [], "test_question_id", "Test query", num_ctx=2048)
print("Ollama Response:", response)
print("Generation Time:", format_time(generation_time))

# รับ Train set collection
train_collection_name = "bird_train_set_rag_evidence_bge_m3_cosine"
train_collection = client.get_collection(train_collection_name)

# ฟังก์ชัน Similarity Search (Top N)
def perform_similarity_search(query_text, top_n=10):
    query_embedding = embedding_model.encode([query_text]).tolist()[0]
    results = train_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    ids = results["ids"][0]  # ดึง ID จากผลลัพธ์
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return ids, documents, metadatas, distances

# ฟังก์ชันคัดกรองด้วย LLM เพื่อเลือก Top K
def rerank_with_llm(original_query, question_id, ids, documents, metadatas, top_k=3):
    candidates = "\n".join([
        f"Candidate {i+1}:\nEvidence: {meta['evidence']}\nID: {ids[i]}"
        for i, meta in enumerate(metadatas)
    ])
    
    prompt = f"""You are an expert in evaluating the semantic similarity between natural language contexts. 
The original query may be in either English or Thai. Your task is to rank the candidate evidence based on their semantic similarity to the original query and select the top {top_k} most similar ones. 
Focus on the meaning and intent of the evidence in relation to the query.

Original Query: {original_query}

Candidate Evidence:
{candidates}

### Instructions:
- Rank the candidates based on semantic similarity to the original query.
- Output only the indices of the top {top_k} candidates (0-based index), separated by commas.
- Example output: 0, 1, 2
- Do not include explanations or additional text."""

    # นับโทเค็นของ prompt
    token_count = count_tokens(prompt, question_id, original_query, max_tokens=8192)
    num_ctx = calculate_dynamic_num_ctx(token_count)
    
    if token_count > 8192:
        print(f"⚠️ Reranking prompt นี้ยาวเกิน 8192 tokens ({token_count} tokens) อาจถูกตัดโดย Ollama")
    elif token_count > 6553:
        print(f"⚠️ Reranking prompt ใกล้ขีดจำกัด ({token_count} tokens)")
    else:
        print(f"\nToken of reranking prompt: {token_count}")
    print(f"Dynamic num_ctx for reranking: {num_ctx}")

    error_log = []
    response, rerank_time = query_ollama(prompt, error_log, f"rerank_q{question_id}", original_query, num_ctx)
    print(f"\n\nCheck response: {response}\n\n")
    
    # บันทึกข้อมูลโทเค็นของ rerank
    token_log.append({
        "stage": "rerank",
        "question_id": question_id,
        "prompt": prompt,
        "token_count": token_count,
        "generation_time": rerank_time,
        "num_ctx": num_ctx
    })

    if not response or error_log:
        print("Error in LLM reranking, falling back to original ranking.")
        return list(range(top_k)), rerank_time
    
    try:
        top_indices = [int(idx.strip()) for idx in response.split(",") if idx.strip().isdigit()]
        top_indices = [idx for idx in top_indices if 0 <= idx < len(metadatas)]
        if len(top_indices) < top_k:
            fallback_indices = [i for i in range(len(metadatas)) if i not in top_indices]
            top_indices += fallback_indices[:top_k - len(top_indices)]
        top_indices = top_indices[:top_k]
    except:
        print("Invalid LLM response, falling back to original ranking.")
        top_indices = list(range(min(top_k, len(metadatas))))
    
    return top_indices, rerank_time

# โหลด Dev set
test_file = '../bird/data/train/test_split_bird_20.json'
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# ประมวลผล Dev set
overall_start_time = time.time()

for i, item in enumerate(test_data[:30]):  # ทดสอบ 30 คำถามแรก
    print(f"Processed Test question {i+1}/{len(test_data)}")
    question_id = item['question_id']
    question = item['question']
    question_th = item.get('question_th')

    # เลือกคำถามที่ใช้ (ถ้ามี question_th ใช้ก่อน ถ้าไม่มีใช้ question)
    query_text = question_th if question_th else question

    # Similarity Search (Top N)
    print(f"\n=== Processing Question ID: {question_id} ===")
    print(f"Original Query: {query_text}")
    ids, documents, metadatas, distances = perform_similarity_search(query_text, top_n=10)

    print("\nInitial Top 10 Results (Before LLM Reranking):")
    for j, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        print(f"Result {j+1}:")
        print(f"Similarity Score: {1 - dist:.4f}")
        print(f"ID: {doc_id}")
        print(f"Evidence: {meta['evidence']}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

    # คัดกรองด้วย LLM เพื่อเลือก Top K
    top_k = 3
    top_indices, rerank_time = rerank_with_llm(query_text, question_id, ids, documents, metadatas, top_k=top_k)
    
    print(f"\nTop {top_k} Results (After LLM Reranking):")
    has_matching_id = False
    for j, idx in enumerate(top_indices):
        meta = metadatas[idx]
        doc_id = ids[idx]
        # สกัดเลขจาก doc_id (เช่น q11_evidence_th → 11)
        evidence_number = int(re.search(r'q(\d+)_', doc_id).group(1)) if re.search(r'q(\d+)_', doc_id) else None
        print(f"Result {j+1}:")
        print(f"ID: {doc_id}")
        print(f"Extracted Number: {evidence_number}")
        print(f"Evidence: {meta['evidence']}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")
        # เช็คว่าเลขจาก evidence ตรงกับ question_id (ตัวเลข) หรือไม่
        if evidence_number is not None and str(evidence_number) == str(question_id):
            has_matching_id = True
    
    # อัปเดต id_match_log
    if has_matching_id:
        id_match_log["correct"] += 1
    else:
        id_match_log["incorrect"] += 1

overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\n=== Overall Processing Time: {format_time(overall_time)} ===")

# แสดง Token Exceed Summary
print("\n=== Token Exceed Summary ===")
if token_exceed_log:
    print(f"Total queries exceeding token limit: {len(token_exceed_log)}")
    for entry in token_exceed_log:
        print(f"Question ID: {entry['question_id']}")
        print(f"Query: {entry['query']}")
        print(f"Token Count: {entry['token_count']} (Max: {entry['max_tokens']})")
        print("--------------------")
else:
    print("No queries exceeded the token limit.")

# แสดง ID Match Summary
print("\n=== ID Match Summary ===")
print(f"Number of questions with matching ID in Top K: {id_match_log['correct']}")
print(f"Number of questions with no matching ID in Top K: {id_match_log['incorrect']}")

# สร้าง id_match_log.json
#with open('../bird/exp_result/gemma3_test_split_output/id_match_log.json', 'w', encoding='utf-8') as f:
#    json.dump(id_match_log, f, ensure_ascii=False, indent=4)

# สร้าง token_log.json
#with open('../bird/exp_result/gemma3_test_split_output/token_log.json', 'w', encoding='utf-8') as f:
#    json.dump(token_log, f, ensure_ascii=False, indent=4)