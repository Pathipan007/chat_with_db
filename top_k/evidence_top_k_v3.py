import json
import requests
import re
import time
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import math
import fasttext

# โหลดโมเดล fastText language detection
lang_model = fasttext.load_model("../lang_detect_model/lid.176.bin")

# ใช้ tokenizer ของ google/gemma-3-12b-it
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

# สร้าง list สำหรับเก็บ log คำถามที่เกินโทเค็น
token_exceed_log = []

# สร้าง list สำหรับเก็บ log ข้อมูลโทเค็นทั้งหมด
token_log = []

# สร้าง dict สำหรับเก็บผลการเช็ค id
id_match_log_top_n = {"correct": 0, "incorrect": 0}
id_match_log_top_k = {"correct": 0, "incorrect": 0}

lang_count_log = {"en": 0, "th": 0, "other": 0}

def detect_language(text):
    prediction = lang_model.predict(text.strip().replace('\n', ' '))[0][0]
    lang_code = prediction.replace('__label__', '')
    return lang_code

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

# รับ Train set collection
train_collection_name = "bird_train_set_evidence_bge_m3"
train_collection = client.get_collection(train_collection_name)

# ฟังก์ชัน Similarity Search (Top N) พร้อม filter ภาษา
def perform_similarity_search(query_text, lang="en", top_n=10):
    query_embedding = embedding_model.encode([query_text]).tolist()[0]
    results = train_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        where={"language": lang}
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return ids, documents, metadatas, distances

# ฟังก์ชันคัดกรองด้วย LLM เพื่อเลือก Top K
def rerank_with_llm(original_query, question_id, ids, documents, metadatas, distances, top_k=3):
    candidates = "\n".join([
        f"Candidate {i}: Similarity Score: {1 - dist:.4f}, Evidence: {meta['evidence']}."
        for i, (meta, dist) in enumerate(zip(metadatas, distances))
    ])
    
    lang_code = detect_language(original_query)
    lang_display = {
        "th": "Thai",
        "en": "English"
    }.get(lang_code, lang_code.capitalize())

    prompt = f"""You are an expert in evaluating semantic similarity between natural language contexts.
The original query is in {lang_display}, and all candidate evidence is in the same language.
Each candidate includes a similarity score (from 0.00 to 1.00) based on embedding similarity. Your task is to select the top {top_k} most semantically relevant candidates.

You should carefully consider both:
1. The **semantic content** of the evidence compared to the query
2. The **similarity score**, which reflects how close the candidate is based on vector embedding

Original Query:
{original_query}

Candidate Evidence (with similarity scores):
{candidates}

### Instructions:
- Rank the candidates based on both their semantic similarity and similarity score.
- Use your judgment to resolve conflicts between score and meaning (e.g., high score but off-topic).
- Output only the indices of the top {top_k} candidates (0-based index), separated by commas.
- Example output: 0, 2, 4
- Do not include any explanations or additional text."""
    
    # นับโทเค็นของ prompt
    token_count = count_tokens(prompt, question_id, original_query, max_tokens=8192)
    num_ctx = calculate_dynamic_num_ctx(token_count)

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
            print(f"⚠️ Only {len(top_indices)} valid indices found, falling back for Question ID {question_id}")
            fallback_indices = [i for i in range(len(metadatas)) if i not in top_indices]
            top_indices += fallback_indices[:top_k - len(top_indices)]
        top_indices = top_indices[:top_k]
    except:
        print("Invalid LLM response, falling back to original ranking.")
        top_indices = list(range(min(top_k, len(metadatas))))
    
    return top_indices, rerank_time

def main():
    # ทดสอบการเชื่อมต่อกับ Ollama API
    response, generation_time = query_ollama("Test prompt", [], "test_question_id", "Test query", num_ctx=2048)
    print("Ollama Response:", response)
    print(f"Generation Time: {format_time(generation_time)}\n\n")

    test_file = '../bird/data/train/test_sample_10.json'
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    data = test_data
    total_data = len(data)

    top_n = 15
    top_k = 5
    
    overall_start_time = time.time()

    for i, item in enumerate(data):

        print(f"Processed Test question {i+1}/{total_data}")
        question_id = item['question_id']
        question = item['question']
        question_th = item.get('question_th')

        # เลือกคำถามที่ใช้
        query_text = question_th

        # ตรวจจับภาษา
        lang = detect_language(query_text)

        if lang in ["en", "th"]:
            lang_count_log[lang] += 1
        else:
            lang_count_log["other"] += 1

        # fallback หากไม่พบภาษาที่ต้องการ
        if lang not in ['en', 'th']:
            lang = 'en'

        # Similarity Search (Top N)
        print(f"\n=== Processing Question ID: {question_id} ===")
        print(f"Original Query: {query_text}")
        ids, documents, metadatas, distances = perform_similarity_search(query_text, lang=lang, top_n=top_n)

        print(f"\nInitial Top {top_n} Results (Before LLM Reranking):")
        has_matching_id_top_n = False
        for j, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            evidence_number_top_n = int(re.search(r'q(\d+)_', doc_id).group(1)) if re.search(r'q(\d+)_', doc_id) else None
            print(f"Result {j+1}:")
            print(f"Similarity Score: {1 - dist:.4f}")
            print(f"ID: {doc_id}")
            print(f"Extracted Number: {evidence_number_top_n}")
            print(f"Evidence: {meta['evidence']}")
            print("-----------------------------------------------------------------------------------------------------------\n\n")
            # เช็คว่าเลขจาก evidence ตรงกับ question_id (ตัวเลข) หรือไม่
            if evidence_number_top_n is not None and str(evidence_number_top_n) == str(question_id):
                has_matching_id_top_n = True

        print(f"Check match: {has_matching_id_top_n}")
        # อัปเดต id_match_log
        if has_matching_id_top_n:
            id_match_log_top_n["correct"] += 1
        else:
            id_match_log_top_n["incorrect"] += 1
        
        # คัดกรองด้วย LLM เพื่อเลือก Top K
        top_indices, rerank_time = rerank_with_llm(query_text, question_id, ids, documents, metadatas, distances, top_k=top_k)
        
        print(f"\nTop {top_k} Results (After LLM Reranking):")
        has_matching_id_top_k = False
        for j, idx in enumerate(top_indices):
            meta = metadatas[idx]
            doc_id = ids[idx]
            evidence_number_top_k = int(re.search(r'q(\d+)_', doc_id).group(1)) if re.search(r'q(\d+)_', doc_id) else None
            print(f"Result {j+1}:")
            print(f"ID: {doc_id}")
            print(f"Extracted Number: {evidence_number_top_k}")
            print(f"Evidence: {meta['evidence']}")
            print("-----------------------------------------------------------------------------------------------------------\n\n")
            # เช็คว่าเลขจาก evidence ตรงกับ question_id (ตัวเลข) หรือไม่
            if evidence_number_top_k is not None and str(evidence_number_top_k) == str(question_id):
                has_matching_id_top_k = True
        
        print(f"Check match: {has_matching_id_top_k}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

        # อัปเดต id_match_log
        if has_matching_id_top_k:
            id_match_log_top_k["correct"] += 1
        else:
            id_match_log_top_k["incorrect"] += 1
        

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    print(f"\n=== Overall Processing Time: {format_time(overall_time)} of {total_data} record ===")

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
    print(f"\n=== ID Match Summary for Top-N = {top_n} ===")
    print(f"Number of questions with matching ID in Top N: {id_match_log_top_n['correct']}")
    print(f"Number of questions with no matching ID in Top N: {id_match_log_top_n['incorrect']}")

    print(f"\n=== ID Match Summary for Top-K = {top_k} ===")
    print(f"Number of questions with matching ID in Top K: {id_match_log_top_k['correct']}")
    print(f"Number of questions with no matching ID in Top K: {id_match_log_top_k['incorrect']}")

    print("\n=== Language Detection Summary ===")
    for k, v in lang_count_log.items():
        print(f"Language '{k}': {v} questions")

    # สร้าง token_log.json
    with open('../bird/exp_result/gemma3_test_split_output/token_log.json', 'w', encoding='utf-8') as f:
        json.dump(token_log, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()