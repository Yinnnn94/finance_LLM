# finance_LLM
# 金融與保險文件檢索系統

本專案實現了一個專門用於金融和保險相關查詢的文件檢索系統，運用了多種自然語言處理技術和嵌入模型。

## 專案結構
```
.
├── data/
│   ├── finance/        # 金融文件語料庫
│   └── faq/           # FAQ文件語料庫
├── src/
│   ├── finance_retrieval.py    # 金融文件檢索實現
│   ├── faq_retrieval.py        # FAQ檢索實現
│   ├── pdf_extractor.py        # PDF文字提取工具
│   └── insurance_query.py      # 保險專用查詢處理
├── requirements.txt
└── README.md
```

## 檔案說明

### 源碼檔案

- `finance_retrieval.py`: 實現基於關鍵字和嵌入的金融文件檢索
- `faq_retrieval.py`: 使用E5嵌入實現FAQ文件的語義檢索
- `pdf_extractor.py`: 提供PDF文件文字提取工具，包括OCR功能
- `insurance_query.py`: 實現保險專用查詢處理，包含同義詞和BM25檢索

## 安裝與設置

1. 安裝所需依賴：
```bash
pip install -r requirements.txt
```

2. 安裝Tesseract OCR：
- Windows：從 https://github.com/UB-Mannheim/tesseract/wiki 下載並安裝
- Linux：`sudo apt-get install tesseract-ocr`
- macOS：`brew install tesseract`

3. 設置環境變數：
```bash
export TESSERACT_PATH="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Windows
# 或
export TESSERACT_PATH="/usr/bin/tesseract"  # Linux/macOS
```

## 模型配置

### E5嵌入模型
- 模型：`intfloat/multilingual-e5-large`
- 最大序列長度：512
- 運行設備：優先使用CUDA，否則使用CPU
- 嵌入維度：1024

### 檢索參數
- 文本分塊大小：200字符
- 嵌入批次大小：8
- 查詢-關鍵詞權重比：0.5:0.5

## 資源需求

### 硬體需求
- 內存：建議最少16GB
- GPU：建議使用配備至少8GB顯存的NVIDIA GPU以獲得最佳性能
- 儲存空間：模型權重和文件緩存至少需要10GB空間

### 軟體需求
- Python 3.8或更高版本
- CUDA 11.0或更高版本（用於GPU支持）
- Tesseract OCR 5.0或更高版本

## 使用示例

```python
# 金融文件檢索
python src/finance_retrieval.py \
    --question_path 問題檔案路徑/questions.json \
    --source_path 源文件路徑/source/docs \
    --output_path 輸出路徑/output.json

# FAQ檢索
python src/faq_retrieval.py \
    --question_path 問題檔案路徑/questions.json \
    --source_path 源文件路徑/source/docs \
    --output_path 輸出路徑/output.json
```

## 性能考量

- E5模型載入時間：GPU約30秒，CPU約1分鐘
- 嵌入生成速度：GPU每分鐘約100個文件，CPU每分鐘約20個文件
- 記憶體使用：基礎約4GB + 每1000個文件約2GB（使用嵌入時）

## 緩存管理

系統實現了以下內容的緩存：
- FAQ嵌入（faq_embeddings.pkl）
- 提取的PDF文字
- 處理後的文件分塊

如果更新源文件或模型參數，請清除緩存檔案。