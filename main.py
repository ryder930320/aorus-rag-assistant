import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import time
import os
import pynvml

# ==========================================
# 1. 資源監控初始化 (NVIDIA VRAM)
# ==========================================
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 抓取第 0 張顯卡

def get_vram_usage():
    """取得當前 VRAM 使用量 (回傳 MB)"""
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return info.used / (1024 * 1024)

# ==========================================
# 2. 模型載入與資料解析
# ==========================================
print("正在載入 Embedding 模型...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("正在解析規格資料...")
chunks = []
with open("specs.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if len(line) > 5:
            chunks.append(line)

doc_embeddings = embedder.encode(chunks)

def search_vectors(query: str, top_k: int = 7):
    """純 Numpy 實作的 Cosine Similarity 檢索"""
    query_emb = embedder.encode([query])[0]
    dot_product = np.dot(doc_embeddings, query_emb)
    norm_docs = np.linalg.norm(doc_embeddings, axis=1)
    norm_query = np.linalg.norm(query_emb)
    similarities = dot_product / (norm_docs * norm_query)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

print("正在載入 LLM 推論引擎...")
# 初始化 Llama.cpp (GPU 加速模式)
llm = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1, # 全面啟用 GPU
    n_ctx=1024,      # 限制上下文長度以節省 VRAM
    verbose=False    
)

# ==========================================
# 3. RAG 核心邏輯與回答生成
# ==========================================
def ask_assistant(user_query: str):
    print(f"\n[使用者提問]: {user_query}")
    print("-" * 40)
    
    # 檢索
    retrieved_info = search_vectors(user_query)
    context_str = "\n".join(retrieved_info)
    #print(f"(內部檢索到的參考資訊: \n{context_str})\n")
    print("-" * 40)
    
    # 防幻覺版 Prompt
    prompt = f"""<|im_start|>system
你是一個專業的 AORUS 規格助手。請依照以下優先級回答問題：

1. 【確認不支援】：若規格明確寫出「不支援」或「非...」，請回答使用者「不支援」。
2. 【規格存在】：若規格存在，請精確回答。問到GPU或是顯示卡時，如果沒有指定型號，請完全列出 BZH/BYH/BXH規格。
3. 【逃生出口】：若規格中「完全找不到」正反面線索（例如價格、上市日期、顏色），請「只能」回答：「根據目前提供的規格表，沒有提到這項資訊。」

請統一使用繁體中文或英文回答，不要編造任何數據。

【規格資訊】
{context_str}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    print("[AI 回答]: ", end="", flush=True)
    
    stream = llm(prompt, max_tokens=256, stream=True, stop=["<|im_end|>"])
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start_time
        
        text = chunk['choices'][0]['text']
        print(text, end="", flush=True)
        token_count += 1
        
    end_time = time.time()
    tps = token_count / (end_time - first_token_time) if first_token_time else 0
    
    # 抓取生成結束當下的 VRAM 佔用量
    current_vram = get_vram_usage()
    
    print("\n\n" + "=" * 40)
    print(f"📊 效能與資源指標 (System Evaluation):")
    print(f" - 首字延遲 (TTFT): {ttft:.3f} 秒")
    print(f" - 生成速度 (TPS):  {tps:.2f} tokens/秒")
    print(f" - VRAM 佔用量:     {current_vram:.2f} MB  (限制: 4096 MB)")
    print("=" * 40)

# ==========================================
# 4. 測試執行區
# ==========================================
if __name__ == "__main__":
    ask_assistant("BZH這台筆電的 GPU 是哪一張？有幾 GB？")
    ask_assistant("Please tell me about the cooling system.") 
    ask_assistant("這台筆電支援觸控螢幕嗎？")
