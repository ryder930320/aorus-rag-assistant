#GIGABYTE AORUS RAG Assistant (Zero-Framework MVP)
本專案為針對 GIGABYTE AORUS MASTER 16 AM6H 消費級筆電規格所開發的硬體 AI 小幫手。
系統嚴格遵守資源限制，在無依賴任何高階 RAG 框架的前提下，達成高併發、極低 VRAM 佔用的即時推論。

🚀 系統亮點與硬性指標達成
4GB VRAM 限制通關：實測最高 VRAM 佔用僅 ~1.58 GB。

Zero-Framework 實作：捨棄 LangChain / LlamaIndex，採用純 Python Numpy 實作 Cosine Similarity 向量檢索與手動 Chunking。

輕量級推論引擎：底層採用 llama.cpp Python Binding 搭配 CUDA 加速，支援 Streaming 串流輸出。

現代化環境管理：全面採用 uv 進行極速套件依賴與虛擬環境管理。

🛠️ 安裝與啟動步驟
本專案使用 uv 進行環境管理，確保依賴的絕對一致性。

1. 建立並啟用環境

Bash
uv venv
Windows 啟用方式:
.venv\Scripts\activate

2. 安裝相依套件
Bash
uv sync
(註：若遇 GPU 監控報錯，請確認已安裝 nvidia-ml-py，避免使用已棄用的 pynvml)

3. 下載核心模型
請在根目錄建立 models 資料夾，並下載 qwen2.5-1.5b-instruct-q4_k_m.gguf 置入其中。

4. 啟動系統
Bash
uv run main.py

🧠 系統評測與定性分析 (System Evaluation & Benchmark)
1. 定量指標 (Quantitative Metrics)
在 NVIDIA RTX 環境下實測，本系統展現出極高的推論效率：

首字延遲 (TTFT): 穩定落在 0.03 ~ 0.3 秒 區間（支援串流即時吐字）。

生成速度 (TPS): 平均達 120 ~ 150 Tokens/sec。

VRAM 佔用: 檢索與生成全程控制在 1,580 MB 左右，完美符合消費級筆電 4GB VRAM 之嚴苛限制。

2. 定性分析：資料解析管線與 SLM 邊界測試 (Qualitative Benchmark)
在開發「資料解析 (Data Parsing)」模組時，我針對 1.5B 級別的小型語言模型 (SLM) 進行了深入的邊界測試，並得出以下架構決策：

觀察 (Observation)：
在處理技嘉官網複雜的 HTML 對比表（如 BZH/BYH/BXH 多型號混排，且單一屬性跨多行排版）時，若採用「線上即時爬蟲 + LLM 動態清洗」，1.5B 模型極易發生「注意力衰減 (Attention Drop)」與「對齊失效 (Alignment Failure)」。例如：模型會將 RTX 5090 的 175W 誤植為儲存裝置的規格，或將 GDDR7 與系統記憶體混淆。

架構決策 (Architectural Decision)：
為確保 RAG 系統 100% 的事實準確性，並徹底貫徹 Edge AI 設備的極低資源消耗，我放棄了線上動態爬蟲，轉而採用 「解耦資料管線 (Decoupled Data Pipeline)」 架構。
系統底層改為讀取經人工驗證的 黃金資料集 (Human-verified Golden Dataset, specs.txt)。這種純粹的 Key-Value 結構不僅消除了模型檢索時的幻覺風險，更將資料處理的 VRAM 負擔降至 0，將所有的算力保留給終端使用者的問答生成。

📂 專案結構
pyproject.toml / uv.lock: 環境與套件依賴鎖定檔。

main.py: RAG 核心邏輯（包含手寫 Vector Search 與 llama.cpp 推論）。

specs.txt: AORUS MASTER 16 結構化黃金資料集。

models/: (被 .gitignore 忽略) 存放 GGUF 量化模型。
