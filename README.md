# GIGABYTE AORUS RAG Assistant 💻🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![uv](https://img.shields.io/badge/Package_Manager-uv-purple)
![llama.cpp](https://img.shields.io/badge/Engine-llama.cpp-orange)
![Zero_Framework](https://img.shields.io/badge/Framework-Zero%20(Pure%20Python)-success)

本專案為針對 **GIGABYTE AORUS MASTER 16 AM6H** 消費級筆電規格所開發的硬體 AI 小幫手。
系統嚴格遵守資源限制，在無依賴任何高階 RAG 框架的前提下，透過純手工編寫核心演算法，達成高併發、極低 VRAM 佔用的即時推論。

---

## 🎯 核心任務與硬性規範達成度 (Hard Constraints Checklist)

| 規範項目 | 達成方式與技術選型 | 狀態 |
| :--- | :--- | :---: |
| **4GB VRAM 限制** | 選用 `Qwen2.5-1.5B-Instruct` 搭配 `Q4_K_M` 量化[cite: 3]。 | ✅ 通過 |
| **No Frameworks** | 捨棄 LangChain / LlamaIndex。純手工使用 `numpy` 實作 Cosine Similarity 向量檢索[cite: 3]。 | ✅ 通過 |
| **環境管理** | 全面採用 `uv` 進行極速套件依賴與虛擬環境鎖定 (`pyproject.toml`)[cite: 3]。 | ✅ 通過 |
| **推論引擎** | 底層採用 `llama.cpp` (Python binding) 搭配 CUDA 運算，支援即時 Streaming 輸出[cite: 3]。 | ✅ 通過 |
| **中英雙語支援** | Embedding 模型採用 `paraphrase-multilingual-MiniLM-L12-v2`，完美支援混合語系檢索[cite: 3]。 | ✅ 通過 |

---

## 🧠 模型選擇理由 (Model Selection Rationale)

為了在 **4GB VRAM** 的嚴苛限制下，達成「精準事實對齊」與「極致生成速度」的平衡，本專案的核心決策如下：

### 1. 為何選擇 Qwen2.5-1.5B-Instruct？
*   **記憶體預算分配 (VRAM Budget)**：
    *   **模型權重**：使用 `Q4_K_M` 量化後，模型權重僅佔約 **900 MB**[cite: 3]。
    *   **KV Cache**：設定 `n_ctx=2048`，預留約 **200 - 300 MB** 空間處理長文本上下文[cite: 3]。
    *   **系統/Embedding 預算**：預留約 **400 MB** 供向量運算與系統背景佔用[cite: 3]。
    *   **實測峰值**：總 VRAM 消耗穩定控制於 **1.58 GB** 左右，遠低於 4GB 限制，確保了系統的極高穩定性與擴充餘裕[cite: 1, 3]。
*   **指令遵循能力 (Instruction Following)**：在小型語言模型 (SLM) 領域中，Qwen2.5 系列在處理 Key-Value 結構化資料提取上，展現了比同量級模型更強的邏輯對齊與 Prompt 遵循能力[cite: 3, 4]。

### 2. 推論引擎優化
*   **llama.cpp 整合**：透過 `n_gpu_layers=-1` 實作全 GPU 層加速，確保在低顯存環境下仍能達成毫秒級響應[cite: 3]。

---

## 🏗️ 系統架構與 RAG 管線 (System Architecture)

1. **資料解析與索引 (Data Parsing & Indexing)**
   * **結構化語意綁定**：將非結構化的網頁文本，透過人工驗證轉化為 `specs.txt` (Golden Dataset)[cite: 3]。
   * **Chunking 策略**：採用「Key-Value 屬性強綁定」切分法，確保 GPU 型號與功耗等關鍵數據不因固定字數切分而遺失上下文[cite: 3, 4]。

2. **向量檢索 (Vector Retrieval)**
   * 將使用者提問進行 Embedding，透過 `numpy` 執行矩陣內積與正規化 (L2 Norm)，精準抓取 Top-K 最相關的硬體規格[cite: 3]。

3. **生成與防幻覺機制 (Generation & Anti-Hallucination)**
   * **三級判斷架構 (Three-tier Logic)**：在 Prompt 中導入「確認不支援」、「規格存在」、「逃生出口」三個邏輯層級[cite: 3]。
   * **否定語義優化**：針對「觸控螢幕」等否定規格，將其轉化為結論性陳述（如：標註為非觸控螢幕），解決 SLM 難以處理否定句的痛點[cite: 1, 3]。

---

## 📊 評測結果分析 (System Evaluation & Benchmark)

### 1. 定量指標 (Quantitative Metrics)
| 評測項目 | 表現數值 | 評語 |
| :--- | :--- | :--- |
| **首字延遲 (TTFT)** | **0.056 - 0.344 秒**[cite: 1] | 極致響應，使用者體感幾乎無延遲[cite: 1]。 |
| **生成速度 (TPS)** | **125 - 156 tokens/s**[cite: 1] | 遠超一般閱讀速度，推論效率極高[cite: 1]。 |
| **VRAM 佔用量** | **~1580 MB**[cite: 1] | 穩定運行，符合並大幅優於 4GB 限制[cite: 1]。 |

### 2. 定性分析：SLM 邊界測試與優化 (Qualitative Analysis)
*   **型號對齊優化 (Model Alignment)**：
    *   **挑戰**：1.5B 級別模型處理 BZH/BYH/BXH 多型號混排時容易發生注意力偏移或數據混淆[cite: 1, 3]。
    *   **對策**：透過 **「強語義對齊 (Hard Semantic Alignment)」** 技術，在 `specs.txt` 中重複型號關鍵標籤，成功達成 100% 的事實準確度，精確區分 140W (BXH) 與 175W (BZH) 的功耗差異[cite: 1, 4]。
*   **防幻覺邊界設定**：
    *   對於「售價」或「上市日期」等未提供資訊，系統能精確觸發「根據目前提供的規格表，沒有提到這項資訊」之逃生出口，避免產生隨機數值幻覺[cite: 1, 3]。
*   **讀卡機語意判定**：
    *   系統能正確理解 `MicroSD` 插槽等同於具備 `SD 讀卡機` 功能，克服了小型模型常發生的詞彙理解偏差[cite: 1, 4]。

---

## 🚀 安裝與執行指引 (Getting Started)

### Step 1: 建立並啟用虛擬環境
~~~bash
uv venv
# Windows 系統:
.venv\Scripts\activate
# Mac/Linux 系統:
source .venv/bin/activate
~~~

### Step 2: 安裝相依套件
~~~bash
# 使用windows powershell:
$env:CMAKE_ARGS="-DGGML_CUDA=on"
uv sync
~~~
*(內含 `nvidia-ml-py` 用於精準監控 VRAM 佔用[cite: 1])*

### Step 3: 下載核心模型
請在專案根目錄建立 `models` 資料夾，並將 `qwen2.5-1.5b-instruct-q4_k_m.gguf` 置入其中。

~~~text
📦 aorus-rag-assistant
 ┣ 📂 models
 ┃ ┗ 📜 qwen2.5-1.5b-instruct-q4_k_m.gguf
 ┣ 📜 main.py
 ┣ 📜 specs.txt
 ┗ 📜 pyproject.toml
~~~

### Step 4: 啟動 AI 助手
~~~bash
uv run main.py
~~~
