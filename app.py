import os
import uuid
from typing import Dict, List

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for

from processor.excel_processor import REMOVAL_RULES, ProcessResult, dataframe_to_bytes, process_workbook
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sap-excel-converter")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

RESULT_CACHE: Dict[str, ProcessResult] = {}
MAX_CACHE_ITEMS = 20
REMOVED_PREVIEW_LIMIT = 50


def _store_result(result: ProcessResult) -> str:
    job_id = str(uuid.uuid4())
    RESULT_CACHE[job_id] = result
    if len(RESULT_CACHE) > MAX_CACHE_ITEMS:
        # FIFO eviction to避免占用過多記憶體
        first_key = next(iter(RESULT_CACHE))
        RESULT_CACHE.pop(first_key, None)
    return job_id


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        upload = request.files.get("workbook")
        if not upload or upload.filename == "":
            flash("請選擇要上傳的 Excel 檔案。", "danger")
            return redirect(url_for("index"))

        try:
            result = process_workbook(upload)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("index"))
        except Exception as exc:
            flash(f"整理檔案時發生錯誤：{exc}", "danger")
            return redirect(url_for("index"))

        job_id = _store_result(result)
        original_preview = []
        for idx, row in result.original_df.iterrows():
            record = {col: row.get(col) for col in result.columns}
            record["_is_removed"] = idx in result.removed_row_indexes
            record["_is_kangxuan_exception"] = idx in result.kangxuan_exception_indexes
            record["_row_index"] = idx  # 儲存原始索引，用於後續處理
            original_preview.append(record)
        original_count = len(original_preview)
        return render_template(
            "index.html",
            result=result,
            job_id=job_id,
            rules=REMOVAL_RULES,
            removed_preview_limit=REMOVED_PREVIEW_LIMIT,
            original_preview=original_preview,
            original_count=original_count,
        )

    return render_template(
        "index.html",
        rules=REMOVAL_RULES,
        removed_preview_limit=REMOVED_PREVIEW_LIMIT,
        original_preview=None,
        original_count=0,
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    result = RESULT_CACHE.get(job_id)
    if not result:
        flash("找不到對應的檔案，請重新上傳一次。", "warning")
        return redirect(url_for("index"))

    filename = f"sap_excel_{job_id}.xlsx"
    return send_file(
        result.build_download_stream(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/download-custom/<job_id>", methods=["POST"])
def download_custom(job_id: str):
    """根據用戶選擇的項目生成 Excel"""
    result = RESULT_CACHE.get(job_id)
    if not result:
        return jsonify({"error": "找不到對應的檔案"}), 404

    data = request.get_json()
    selected_indices = data.get("selected_indices", [])
    
    if not selected_indices:
        return jsonify({"error": "請至少選擇一筆資料"}), 400

    try:
        # 將字串索引轉換為整數
        selected_indices = [int(idx) for idx in selected_indices]
        
        # 從保持原始欄位順序的 DataFrame 中選取用戶選擇的列
        selected_df = result.original_df_original_order.loc[selected_indices].copy()
        
        # 使用原始欄位順序（不重新排列）
        # 確保欄位順序與原始檔案一致
        selected_df = selected_df[result.original_columns]
        
        # 生成 Excel bytes
        workbook_bytes = dataframe_to_bytes(selected_df)
        
        # 建立臨時檔案或直接返回
        from io import BytesIO
        filename = f"sap_excel_custom_{job_id}.xlsx"
        return send_file(
            BytesIO(workbook_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        return jsonify({"error": f"處理檔案時發生錯誤：{str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5223))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)

