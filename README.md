# SAP 訂單整理工具

這是一個用於整理訂單 Excel 檔案的 Web 應用程式，可以自動標示不上傳 SAP 的訂單，並支援手動選擇要上傳的項目。

## 功能

- 上傳原始訂單 Excel 檔案
- 自動判斷並標示不上傳 SAP 的訂單
- 顯示不上傳訂單的判斷規則
- 手動選擇要上傳的訂單項目
- 下載整理後的 Excel 檔案

## 本地開發

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 執行應用

```bash
python app.py
```

應用程式會在 http://localhost:5223 啟動。

## Railway 部署

### 1. 準備 Git 儲存庫

確保你的程式碼已經推送到 GitHub。

### 2. 在 Railway 上建立專案

1. 前往 [Railway](https://railway.app)
2. 登入並點擊 "New Project"
3. 選擇 "Deploy from GitHub repo"
4. 選擇你的儲存庫

### 3. 設定環境變數（可選）

在 Railway 專案設定中，可以設定以下環境變數：

- `FLASK_SECRET_KEY`: Flask session 密鑰（建議設定）
- `PORT`: Railway 會自動設定，不需要手動設定

### 4. 部署

Railway 會自動偵測 Python 應用並開始部署。

## 技術棧

- Python 3.11
- Flask 3.0.3
- pandas 2.2.3
- openpyxl 3.1.5

