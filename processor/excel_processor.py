from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class ProcessedRow:
    row_index: int
    rule_id: str
    rule_name: str
    description: str
    row_data: Dict[str, Any]


@dataclass
class ProcessResult:
    original_df: pd.DataFrame
    original_df_original_order: pd.DataFrame  # 保持原始欄位順序的 DataFrame
    kept_df: pd.DataFrame
    removed_rows: List[ProcessedRow]
    removed_row_indexes: Set[int]
    kangxuan_exception_indexes: Set[int]
    columns: List[str]
    original_columns: List[str]  # 原始欄位順序
    stats: Dict[str, int]
    amount_column: Optional[str]
    workbook_bytes: bytes

    def build_download_stream(self) -> BytesIO:
        """Return a BytesIO stream ready to be sent as a file download."""
        return BytesIO(self.workbook_bytes)


@dataclass
class Rule:
    rule_id: str
    name: str
    description: str

    def applies(self, row: pd.Series, context: "DataContext") -> Tuple[bool, Optional[str]]:
        raise NotImplementedError


class ZeroAmountKeywordRule(Rule):
    def __init__(self, rule_id: str, name: str, description: str, keywords: List[str]):
        super().__init__(rule_id, name, description)
        self.keywords = [kw.lower() for kw in keywords]

    def applies(self, row: pd.Series, context: "DataContext") -> Tuple[bool, Optional[str]]:
        amount = context.extract_amount(row)
        if amount is None or abs(amount) > 0.0001:
            return False, None
        if context.contains_keywords(row, self.keywords):
            return True, f"金額為 0 並且品項包含：{', '.join(self.keywords)}"
        return False, None


class KeywordRule(Rule):
    def __init__(self, rule_id: str, name: str, description: str, keywords: List[str]):
        super().__init__(rule_id, name, description)
        self.keywords = [kw.lower() for kw in keywords]

    def applies(self, row: pd.Series, context: "DataContext") -> Tuple[bool, Optional[str]]:
        if context.contains_keywords(row, self.keywords):
            return True, f"品項包含：{', '.join(self.keywords)}"
        return False, None


class EmptyMaterialRule(Rule):
    """料號欄位為空字串或 None 時，不上傳 SAP。"""
    
    def __init__(self, rule_id: str, name: str, description: str):
        super().__init__(rule_id, name, description)

    def applies(self, row: pd.Series, context: "DataContext") -> Tuple[bool, Optional[str]]:
        material_col = getattr(context, "material_column", None)
        if not material_col or material_col not in row:
            return False, None
        value = row[material_col]
        if pd.isna(value):
            return True, f"料號欄位（{material_col}）為空，不上傳 SAP"
        if isinstance(value, str) and not value.strip():
            return True, f"料號欄位（{material_col}）為空字串，不上傳 SAP"
        return False, None


class DataContext:
    """Utility helpers derived from the uploaded DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.amount_column = self._detect_amount_column(df)
        self.text_columns = self._detect_text_columns(df)
        self.dlno_column = self._detect_dlno_column(df)
        self.material_column = self._detect_material_column(df)

    @staticmethod
    def _detect_amount_column(df: pd.DataFrame) -> Optional[str]:
        prioritized_keywords = [
            "原價金額",
            "總貨價",
            "應收金額",
            "總金額",
            "金額",
            "總計",
            "合計",
        ]
        lowered = {col: col.lower() for col in df.columns}
        for keyword in prioritized_keywords:
            for col in df.columns:
                if keyword.lower() in lowered[col]:
                    return col
        numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
        return numeric_candidates[0] if numeric_candidates else None

    @staticmethod
    def _detect_text_columns(df: pd.DataFrame) -> List[str]:
        object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        keyword_cols = [col for col in df.columns if any(token in col for token in ("名稱", "備註", "課程", "參考書", "商品", "內容"))]
        merged = object_cols + keyword_cols
        # Preserve column order while removing duplicates.
        seen: set[str] = set()
        ordered_unique: List[str] = []
        for col in merged:
            if col not in seen:
                seen.add(col)
                ordered_unique.append(col)
        return ordered_unique

    @staticmethod
    def _detect_dlno_column(df: pd.DataFrame) -> Optional[str]:
        """偵測平台號碼DLNO欄位"""
        keywords = ["平台號碼", "dlno", "訂單號碼", "訂單編號"]
        lowered = {col: col.lower() for col in df.columns}
        for keyword in keywords:
            for col in df.columns:
                if keyword.lower() in lowered[col]:
                    return col
        return None

    @staticmethod
    def _detect_material_column(df: pd.DataFrame) -> Optional[str]:
        """偵測料號欄位"""
        keywords = ["料號", "料件", "貨號", "商品代碼"]
        lowered = {col: col.lower() for col in df.columns}
        for keyword in keywords:
            for col in df.columns:
                if keyword.lower() in lowered[col]:
                    return col
        return None

    def extract_amount(self, row: pd.Series) -> Optional[float]:
        if not self.amount_column or self.amount_column not in row:
            return None
        value = row[self.amount_column]
        if pd.isna(value):
            return None
        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()
            if not cleaned:
                return None
            try:
                value = float(cleaned)
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def contains_keywords(self, row: pd.Series, keywords: List[str]) -> bool:
        lowered_keywords = [kw.lower() for kw in keywords]
        for col in self.text_columns:
            cell_value = row.get(col)
            if pd.isna(cell_value):
                continue
            cell_text = str(cell_value).lower()
            if any(keyword in cell_text for keyword in lowered_keywords):
                return True
        return False


REMOVAL_RULES: List[Rule] = [
    ZeroAmountKeywordRule(
        "rule_zero_open_code",
        "0 元訂單（開通碼）",
        "0 元且為開通碼的訂單不上傳 SAP",
        ["開通碼"],
    ),
    EmptyMaterialRule(
        "rule_empty_material",
        "料號為空",
        "料號欄位為空或空字串的訂單不上傳 SAP",
    ),
    KeywordRule(
        "rule_reference_jinan",
        "金安",
        "包含「金安」的訂單不上傳 SAP",
        ["金安"],
    ),
    ZeroAmountKeywordRule(
        "rule_zero_online_course",
        "0 元訂單 - 線上課程",
        "行銷手動塞課程的 0 元訂單不上傳 SAP",
        ["線上課程", "送段考卷", "送素養課"],
    ),
]


def is_kangxuan_exception(row: pd.Series, row_index: int, df: pd.DataFrame, context: DataContext) -> bool:
    """
    康軒參考書目 0 元訂單需要保留並上傳 SAP。
    當資料列的項次金額是0元時，如有其他資料列是相同的平台號碼DLNO並且項次金額不是0元時，
    則是參考書，不刪除（須上傳SAP）。
    """
    amount = context.extract_amount(row)
    # 只有當項次金額為0時才需要檢查
    if amount is None or abs(amount) >= 0.0001:
        return False
    # 若料號欄位為空，則不適用康軒例外規則，交由其他規則處理
    material_col = getattr(context, "material_column", None)
    if material_col and material_col in row:
        material_value = row[material_col]
        if pd.isna(material_value) or (isinstance(material_value, str) and not material_value.strip()):
            return False
    
    # 檢查是否有平台號碼DLNO欄位
    if not context.dlno_column or context.dlno_column not in row:
        return False
    
    dlno_value = row[context.dlno_column]
    if pd.isna(dlno_value):
        return False
    
    # 檢查相同平台號碼DLNO的其他列，是否有非0元的項目
    same_dlno_rows = df[df[context.dlno_column] == dlno_value]
    for other_idx, other_row in same_dlno_rows.iterrows():
        if other_idx == row_index:
            continue
        other_amount = context.extract_amount(other_row)
        if other_amount is not None and abs(other_amount) >= 0.0001:
            # 找到相同平台號碼DLNO且項次金額不為0的列，則此0元訂單應保留
            return True
    
    return False


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """重新排列欄位順序，將「發票項次品名」、「項次金額」和「料號」移到第3、第4和第5欄"""
    if df.empty:
        return df
    
    columns = df.columns.tolist()
    
    # 找出目標欄位
    product_name_col = None
    amount_col = None
    material_col = None
    
    for col in columns:
        if "發票項次品名" in str(col) or "品名" in str(col):
            product_name_col = col
        if "項次金額" in str(col) or ("金額" in str(col) and "項次" in str(col)):
            amount_col = col
        if "料號" in str(col):
            material_col = col
    
    if not product_name_col or not amount_col:
        # 如果找不到主要目標欄位，返回原始順序
        return df
    
    # 建立新的欄位順序
    new_columns = []
    used_cols = {product_name_col, amount_col}
    if material_col:
        used_cols.add(material_col)
    
    # 前兩欄保持原順序（排除目標欄位）
    for col in columns:
        if col not in used_cols:
            if len(new_columns) < 2:
                new_columns.append(col)
            else:
                break
    
    # 第3欄：發票項次品名
    if product_name_col:
        new_columns.append(product_name_col)
    
    # 第4欄：項次金額
    if amount_col:
        new_columns.append(amount_col)
    
    # 第5欄：料號
    if material_col:
        new_columns.append(material_col)
    
    # 其餘欄位按原順序加入（排除已使用的）
    for col in columns:
        if col not in new_columns:
            new_columns.append(col)
    
    return df[new_columns]


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def process_workbook(file_stream) -> ProcessResult:
    """Process an uploaded Excel file according to the business rules."""
    df = pd.read_excel(file_stream)
    if df.empty:
        raise ValueError("匯入的 Excel 沒有資料")

    context = DataContext(df)
    removed_rows: List[ProcessedRow] = []
    kept_rows: List[pd.Series] = []

    removed_indexes: Set[int] = set()
    kangxuan_exception_indexes: Set[int] = set()

    for idx, row in df.iterrows():
        if is_kangxuan_exception(row, idx, df, context):
            kept_rows.append(row)
            kangxuan_exception_indexes.add(idx)
            continue

        matched_rule: Optional[ProcessedRow] = None
        for rule in REMOVAL_RULES:
            matches, detail = rule.applies(row, context)
            if matches:
                matched_rule = ProcessedRow(
                    row_index=idx,
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    description=detail or rule.description,
                    row_data=row.to_dict(),
                )
                break

        if matched_rule:
            removed_rows.append(matched_rule)
            removed_indexes.add(idx)
        else:
            kept_rows.append(row)

    kept_df = pd.DataFrame(kept_rows, columns=df.columns) if kept_rows else pd.DataFrame(columns=df.columns)
    
    # 重新排列欄位順序（用於顯示）
    df_reordered = reorder_columns(df.copy())
    kept_df_reordered = reorder_columns(kept_df.copy())
    
    # 保持原始欄位順序的 DataFrame（用於下載）
    # 確保 kept_df 使用原始欄位順序
    kept_df_original_order = kept_df.copy()
    workbook_bytes = dataframe_to_bytes(kept_df_original_order)
    stats = {
        "original": len(df),
        "removed": len(removed_rows),
        "kept": len(kept_df),
    }

    return ProcessResult(
        original_df=df_reordered,  # 重新排列後的，用於顯示
        original_df_original_order=df.copy(),  # 保持原始順序，用於下載
        kept_df=kept_df_reordered,
        removed_rows=removed_rows,
        removed_row_indexes=removed_indexes,
        kangxuan_exception_indexes=kangxuan_exception_indexes,
        columns=df_reordered.columns.tolist(),  # 重新排列後的欄位順序，用於顯示
        original_columns=df.columns.tolist(),  # 原始欄位順序，用於下載
        stats=stats,
        amount_column=context.amount_column,
        workbook_bytes=workbook_bytes,
    )

