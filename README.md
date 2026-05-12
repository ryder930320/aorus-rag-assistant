# AORUS RAG AI Assistant (4GB VRAM 硬體限制版)

本專案為 AI 硬體小幫手的純 Python RAG 實作。嚴格遵循「無高階框架 (No Frameworks)」、「4GB VRAM 限制」與「uv 環境管理」等規範。

## 🚀 快速啟動 (Quick Start)

1. 安裝環境管理器 `uv` (若未安裝)
2. 複製本專案後，於終端機執行：`uv sync`
3. 安裝支援 CUDA 的 llama-cpp 套件：
   `uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`
4. 執行 RAG 系統：`uv run main.py`

## 🧠 架構與模型選型理由

* **推論引擎**: `llama.cpp` (Python binding)。捨棄會預先佔用大量記憶體的 vLLM，改用資源控制極佳的 `llama.cpp`，確保不發生 OOM。
* **LLM 模型**: `Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)`。量化後檔案僅 ~1.1GB，具備優異的中英混搭指令遵循能力。
* **Embedding 模型**: `paraphrase-multilingual-MiniLM-L12-v2`。體積極小，精準對齊中英雙語規格。
* **向量檢索**: 純 Python 與 `Numpy` 實作 Cosine Similarity 進行 Top-K 檢索，零依賴向量資料庫。

## 📊 系統評測 (System Evaluation)

本系統內建 `pynvml` 資源監控，以下為實測效能數據：

### 1. 定量指標 (Quantitative)
* **平均生成速度 (TPS):** 134 - 149 tokens/sec (CUDA 加速全開)
* **首字延遲 (TTFT):** 首次啟動約 0.27s，後續生成均小於 0.05s。
* **資源佔用:** 實測 VRAM 佔用峰值僅約 **1559 MB**，完美符合 4GB 限制。

### 2. 定性分析 (Qualitative)
* **事實萃取**：精準回答 RTX 5090 與 24GB，未被上下文數據干擾。
* **跨語系問答**：準確將英文提問對應至中文規格表，並以英文流暢回應。
* **抗幻覺機制**：透過嚴格的 System Prompt 與 Fallback 機制，成功攔截「觸控螢幕」等規格表未提及之陷阱題。