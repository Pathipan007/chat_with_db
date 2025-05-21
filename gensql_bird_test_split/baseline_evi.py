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
def query_ollama(prompt, question_id, query, num_ctx):
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
        return "", -1

# โหลดโมเดล embedding
embedding_model = SentenceTransformer('BAAI/bge-m3')

# กำหนด path สำหรับ Vector Store
persist_directory = "./embed_and_vector_store/vector_store/"
client = chromadb.PersistentClient(path=persist_directory)

# รับ Train set collection
train_collection_name = "bird_train_set_rag_evidence_bge_m3_cosine"
train_collection = client.get_collection(train_collection_name)

# ฟังก์ชัน Similarity Search (Top N = 10)
def perform_similarity_search(query_text, top_n=10):
    query_embedding = embedding_model.encode([query_text]).tolist()[0]
    results = train_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return ids, documents, metadatas, distances

# ฟังก์ชันคัดกรองด้วย LLM เพื่อเลือก Top K = 3
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

    token_count = count_tokens(prompt, question_id, original_query, max_tokens=8192)
    num_ctx = calculate_dynamic_num_ctx(token_count)
    
    if token_count > 8192:
        print(f"⚠️ Reranking prompt นี้ยาวเกิน 8192 tokens ({token_count} tokens) อาจถูกตัดโดย Ollama")
    elif token_count > 6553:
        print(f"⚠️ Reranking prompt ใกล้ขีดจำกัด ({token_count} tokens)")
    else:
        print(f"\nToken of reranking prompt: {token_count}")
    print(f"Dynamic num_ctx for reranking: {num_ctx}")
    print(f"\nReranking for Top K = {top_k}...")

    response, rerank_time = query_ollama(prompt, f"rerank_q{question_id}", original_query, num_ctx)
    
    # บันทึกข้อมูลโทเค็นของ rerank
    token_log.append({
        "stage": "rerank",
        "question_id": question_id,
        "prompt": prompt,
        "token_count": token_count,
        "generation_time": rerank_time,
        "num_ctx": num_ctx
    })

    if not response:
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

# ฟังก์ชันดึง schema จาก train_tables.json (เพิ่ม backticks รอบชื่อคอลัมน์)
def get_schema(db_id, tables_file='./bird/data/train/train_tables.json'):
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    for db in tables:
        if db['db_id'] == db_id:
            schema = []
            for table_idx, table_name in enumerate(db['table_names_original']):
                columns = [f"`{col[1]}`" for col in db['column_names_original'] if col[0] == table_idx]
                if columns:
                    schema.append(f"Table: {table_name}, Columns: {', '.join(columns)}")
            return '\n'.join(schema)
    return ""

# ฟังก์ชันล้าง Markdown และแปลง SQL เป็นบรรทัดเดียว
def clean_sql(sql):
    sql = re.sub(r'```sql\n|```', '', sql, flags=re.MULTILINE)
    sql = ' '.join(sql.split())
    if not sql.endswith(';'):
        sql += ';'
    return sql.strip()



def main():
    # ทดสอบการเชื่อมต่อกับ Ollama API
    response, generation_time = query_ollama("Test prompt", "test_question_id", "Test query", num_ctx=2048)
    print("Ollama Response:", response)
    print("Generation Time:", format_time(generation_time))

    # โหลด Test set
    test_file = './bird/data/train/test_split_bird_20.json'
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # เตรียม dictionary สำหรับ predict_test.json
    predict_test_json = {}

    # วัดเวลาเริ่มต้นทั้งหมด
    overall_start_time = time.time()

    # ประมวลผล Test set
    for i, item in enumerate(test_data):
        print(f"Processed Test question {i+1}/{len(test_data)}")
        question_id = item['question_id']
        question = item['question']
        question_th = item.get('question_th')
        db_id = item['db_id']
        
        # เลือกคำถามที่ใช้ (ถ้ามี question_th ใช้ก่อน ถ้าไม่มีใช้ question)
        query_text = question_th if question_th else question

        # ดึง Top K Evidence
        print(f"\n=== Fetching Evidence for Question ID: {question_id} ===")
        print(f"Original Query: {query_text}")
        ids, documents, metadatas, distances = perform_similarity_search(query_text, top_n=10)
        
        #print("\nInitial Top 10 Results (Before LLM Reranking):")
        #for j, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        #    print(f"Result {j+1}:")
        #    print(f"Similarity Score: {1 - dist:.4f}")
        #    print(f"ID: {doc_id}")
        #    print(f"Evidence: {meta['evidence']}")
        #    print("-----------------------------------------------------------------------------------------------------------\n\n")

        top_k = 3
        top_indices, rerank_time = rerank_with_llm(query_text, question_id, ids, documents, metadatas, top_k=top_k)
        
        print(f"\nTop {top_k} Results (After LLM Reranking):")
        selected_evidence = []
        for j, idx in enumerate(top_indices):
            meta = metadatas[idx]
            doc_id = ids[idx]
            print(f"Result {j+1}:")
            print(f"ID: {doc_id}")
            print(f"Evidence: {meta['evidence']}")
            print("-----------------------------------------------------------------------------------------------------------\n\n")
            selected_evidence.append(meta['evidence'])

        # แปลง evidence เป็น string สำหรับ prompt
        evidence_text = "\n".join([f"- {ev}" for ev in selected_evidence]) if selected_evidence else "No relevant evidence found."

        # ดึง schema
        schema = get_schema(db_id)
        
        # สร้าง prompt สำหรับเจน SQL โดยเพิ่ม evidence
        prompt = f"""
    You are an expert in translating natural language questions into SQL queries. 
    The questions may be in either English or Thai, and you must handle both languages correctly. 
    Use the provided database schema and relevant evidence to accurately generate a syntactically and semantically correct SQL query.

    ### Database Schema:
    {schema}

    ### Evidence:
    {evidence_text}

    ### Task:
    Translate the following natural language question into a valid SQL query:

    "{question}"

    ### Output Format:
    - Output only the SQL query in a single line.
    - Do NOT include markdown (e.g., ```sql), explanations, or any additional text.
    - Do NOT use aliases or table joins unless necessary based on the schema and evidence.
    - Use the exact column and table names from the schema.
    """
        
        # นับโทเค็น
        token_count = count_tokens(prompt, question_id, question, max_tokens=8192)
        num_ctx = calculate_dynamic_num_ctx(token_count)
        
        if token_count > 8192:
            print(f"⚠️ Prompt นี้ยาวเกิน 8192 tokens ({token_count} tokens) อาจถูกตัดโดย Ollama")
        elif token_count > 6553:
            print(f"⚠️ Prompt ใกล้ขีดจำกัด ({token_count} tokens)")
        else:
            print(f"Token of SQL generation prompt: {token_count}")
        print(f"Dynamic num_ctx: {num_ctx}")
        print(f"\nSQL Generating...")

        # เรียก API และเก็บเวลาการ Generate
        sql, sql_gen_time = query_ollama(prompt, question_id, question, num_ctx)
        cleaned_sql = clean_sql(sql)
        
        # รวมเวลาการเรียก LLM (rerank + SQL generation)
        total_llm_time = rerank_time + sql_gen_time if rerank_time >= 0 and sql_gen_time >= 0 else -1
        
        # บันทึกข้อมูลโทเค็นของ SQL generation
        token_log.append({
            "stage": "sql_generation",
            "question_id": question_id,
            "prompt": prompt,
            "token_count": token_count,
            "generation_time": sql_gen_time,
            "num_ctx": num_ctx,
            "total_llm_time": total_llm_time
        })
        
        # แปลงเวลาเป็นรูปแบบที่เหมาะสมสำหรับการแสดงผล
        formatted_rerank_time = format_time(rerank_time)
        formatted_sql_gen_time = format_time(sql_gen_time)
        formatted_total_llm_time = format_time(total_llm_time)
        
        # รูปแบบสำหรับ predict_test.json: SQL \t----- bird -----\t db_id
        json_line = f"{cleaned_sql}\t----- bird -----\t{db_id}"
        predict_test_json[str(question_id)] = json_line
        
        print(f"Question ID: {question_id}")
        print(f"Question: {question}")
        print(f"SQL query: {cleaned_sql}")
        print(f"Rerank Time: {formatted_rerank_time}")
        print(f"SQL Generation Time: {formatted_sql_gen_time}")
        print(f"Total LLM Time: {formatted_total_llm_time}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

    # สร้าง predict_test.json
    with open('./bird/exp_result/gemma3_test_split_output/eng_baseline_with_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(predict_test_json, f, ensure_ascii=False, indent=4)

    # สร้าง token_log.json
    with open('./bird/exp_result/gemma3_test_split_output/eng_baseline_with_evidence_token_log.json', 'w', encoding='utf-8') as f:
        json.dump(token_log, f, ensure_ascii=False, indent=4)

    print("=== Generated successful!!! ===")

    # วัดเวลาทั้งหมด
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    print(f"\n=== Overall Processing Time: {format_time(overall_time)} ===")

    # แสดง Token Exceed Summary
    print("\n=== Token Exceed Summary ===")
    if token_exceed_log:
        print(f"Total questions exceeding token limit: {len(token_exceed_log)}")
        for entry in token_exceed_log:
            print(f"Question ID: {entry['question_id']}")
            print(f"Query: {entry['query']}")
            print(f"Token Count: {entry['token_count']} (Max: {entry['max_tokens']})")
            print("--------------------")
    else:
        print("No questions exceeded the token limit.")

if __name__ == "__main__":
    main()