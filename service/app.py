"""
STEP 25-E
Local CSV Batch Matching Service V2

Changes:
1) Persist processed results in st.session_state so download clicks do not require
   running vessel identification again.
2) Show user-friendly Korean labels while preserving frozen backend status values.

Backend flow remains unchanged:
R_B100_V1 -> M6_FINAL_FROZEN -> CONFIDENCE_GATE_V4_NAME_FLAG_TOLERANCE
"""

from __future__ import annotations

import io
import re
import os
import hashlib
import html
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.batch_pipeline import BatchMatcher, PIPELINE_VERSION


APP_TITLE = "항만 선박 식별 CSV 서비스"
MAX_INPUT_ROWS = 5000

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "service.log"
MODEL_METADATA_FILE = ROOT / "results_step27" / "m6_final_frozen_service_metadata.json"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vessel_service")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
        )
        logger.addHandler(handler)

    return logger


def get_artifact_hash() -> str:
    try:
        with MODEL_METADATA_FILE.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("artifact_sha256", "UNKNOWN"))
    except Exception:
        return "UNKNOWN"


LOGGER = setup_logger()
ARTIFACT_HASH = get_artifact_hash()


REQUIRED_COLUMNS = [
    "vessel_name",
    "gross_tonnage",
    "vessel_type",
    "flag_name",
]

RESERVED_SERVICE_COLUMNS = {
    "vessel_name_raw",
    "gross_tonnage_raw",
    "vessel_type_raw",
    "flag_name_raw",
    "vessel_name_clean",
    "gross_tonnage_parsed",
    "gross_tonnage_status",
    "vessel_type_clean",
    "flag_name_clean",
    "name_missing",
    "gt_missing",
    "type_missing",
    "flag_missing",
    "input_status",
    "input_reason",
    "processable",
    "match_status",
}

STATUS_LABELS = {
    "AUTO_MATCH": "자동 식별",
    "REVIEW_REQUIRED": "검토 필요",
    "NOT_PROCESSED": "처리 불가",
    "NO_CANDIDATE": "후보 없음",
    "SYSTEM_ERROR": "시스템 오류",
}

REVIEW_CATEGORY_LABELS = {
    "INFORMATION_INSUFFICIENT": "정보 부족",
    "GENERAL_REVIEW": "검토 필요",
}

INPUT_STATUS_LABELS = {
    "VALID": "정상 입력",
    "MISSING_BUT_PROCESSABLE": "일부 값 누락 · 처리 가능",
    "INVALID": "입력 오류",
}


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "CSV 문자 인코딩을 읽지 못했습니다. "
        "UTF-8 또는 CP949 CSV로 저장한 뒤 다시 시도해주세요."
    ) from last_error


@st.cache_resource(show_spinner=False)
def get_matcher() -> BatchMatcher:
    return BatchMatcher()


STATUS_LABELS = {"AUTO_MATCH": "자동 식별", "REVIEW_REQUIRED": "검토 필요",
                 "NOT_PROCESSED": "처리 불가", "NO_CANDIDATE": "후보 없음",
                 "SYSTEM_ERROR": "시스템 오류"}
ERROR_TEXT = {
    "GT_NOT_NUMERIC": ("GT가 숫자가 아닙니다.", "GT를 숫자로 수정해주세요."),
    "GT_NOT_FINITE": ("GT가 유한한 숫자가 아닙니다.", "GT에 0보다 큰 유한한 숫자를 입력해주세요."),
    "GT_NON_POSITIVE": ("GT는 0보다 커야 합니다.", "GT를 0보다 큰 숫자로 수정해주세요."),
    "NAME_AND_GT_BOTH_MISSING": ("선박명과 GT가 모두 없습니다.", "선박명 또는 올바른 GT를 입력해주세요."),
}
INPUT_FIELDS = {"vessel_name": "입력_선박명", "gross_tonnage": "입력_GT",
                "vessel_type": "입력_선종", "flag_name": "입력_국적"}
CANDIDATE_FIELDS = {"candidate_vessel_id": "후보_선박ID", "catalog_vessel_name": "후보_선박명",
                    "catalog_gross_tonnage": "후보_GT", "catalog_vessel_type": "후보_선종",
                    "catalog_flag_name": "후보_국적"}
FEATURE_FIELDS = ["name_normalized_exact", "name_missing", "name_similarity", "gt_abs_diff",
                  "gt_missing", "type_comparison_status", "flag_comparison_status",
                  "korean_latin_rule_available", "korean_latin_rule_match"]


def error_details(row):
    codes = str(row.get("input_reason", "")).split("|")
    items = [ERROR_TEXT[c] for c in codes if c in ERROR_TEXT]
    if items:
        return " ".join(x[0] for x in items), " ".join(dict.fromkeys(x[1] for x in items))
    if row.get("match_status") == "NO_CANDIDATE":
        return "검색 가능한 후보를 찾지 못했습니다.", "입력 정보를 확인하고 Catalog 등록 여부를 확인해주세요."
    return "처리할 수 없는 입력입니다.", "원본 입력을 확인해주세요."


def result_message(row):
    status = row.get("match_status")
    if status == "AUTO_MATCH":
        return ("검색된 후보가 1개이므로 자동 식별했습니다." if row.get("candidate_count") == 1
                else "자동 식별 기준을 통과했습니다.")
    if status == "REVIEW_REQUIRED":
        if row.get("confidence_reason") == "NAME_FLAG_NEAR_TIE_REVIEW":
            return "1위 후보의 국적이 입력과 다르고, 이름 유사도가 비슷하면서 국적이 일치하는 상위 후보가 있어 검토가 필요합니다."
        if row.get("confidence_reason") == "NAME_FLAG_CONFLICT_REVIEW":
            return "1위 후보의 국적이 입력과 다르고, 이름이 더 비슷하면서 국적도 일치하는 상위 후보가 있어 검토가 필요합니다."
        if row.get("confidence_reason") == "CROSS_SCRIPT_SAFETY_REVIEW":
            return "한글·영문 이름 간 연결로, 동일 선박인지 추가 확인이 필요합니다."
        return "상위 후보들의 점수 차이가 작아 검토가 필요합니다."
    return error_details(row)[0]


def comparison_label(value):
    value = str(value).removeprefix("TYPE_")
    return {"MATCH": "일치", "MISMATCH": "불일치", "MISSING": "정보 없음",
            "UNMAPPED": "분류 불가", "AMBIGUOUS": "분류 모호"}.get(value, "확인 필요")


def name_label(row):
    if row.get("name_missing") == 1:
        return "정보 없음"
    if row.get("name_normalized_exact") == 1:
        return "동일(정규화 기준)"
    value = pd.to_numeric(row.get("name_similarity"), errors="coerce")
    return f"유사도 {value:.3f}" if pd.notna(value) else "정보 없음"



def _is_missing_value(value) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def _status_is(value, target: str) -> bool:
    return str(value).removeprefix("TYPE_").strip().upper() == target


def classify_review_record(result_row, candidate_group):
    """Display-only review reason/priority. Frozen ranking and confidence decisions are untouched."""
    if str(result_row.get("match_status", "")) != "REVIEW_REQUIRED":
        return "", "", 99, ""

    reason = str(result_row.get("confidence_reason", "") or "")
    missing_items = []
    for col, label in [
        ("gross_tonnage_raw", "GT"),
        ("flag_name_raw", "국적"),
        ("vessel_type_raw", "선종"),
        ("vessel_name_raw", "선박명"),
    ]:
        if col in result_row and _is_missing_value(result_row.get(col)):
            missing_items.append(label)

    g = candidate_group.sort_values("m6_rank", kind="mergesort").copy()
    top = g.iloc[0] if not g.empty else pd.Series(dtype=object)
    others = g.iloc[1:5] if len(g) > 1 else g.iloc[0:0]

    top_type_bad = _status_is(top.get("type_comparison_status"), "MISMATCH")
    top_flag_bad = _status_is(top.get("flag_comparison_status"), "MISMATCH")
    top_name = pd.to_numeric(top.get("name_similarity"), errors="coerce")
    top_name = float(top_name) if pd.notna(top_name) else 0.0

    better_alt = False
    for _, alt in others.iterrows():
        alt_name = pd.to_numeric(alt.get("name_similarity"), errors="coerce")
        alt_name = float(alt_name) if pd.notna(alt_name) else 0.0
        alt_type_ok = _status_is(alt.get("type_comparison_status"), "MATCH")
        alt_flag_ok = _status_is(alt.get("flag_comparison_status"), "MATCH")
        if alt_name >= max(0.80, top_name - 0.03) and (alt_type_ok or alt_flag_ok) and (top_type_bad or top_flag_bad):
            better_alt = True
            break

    margin = pd.to_numeric(result_row.get("score_margin"), errors="coerce")

    # Existing safety gates have highest operational priority.
    if reason in {"NAME_FLAG_NEAR_TIE_REVIEW", "NAME_FLAG_CONFLICT_REVIEW"}:
        return "🔴 높음", "후보 정보 충돌", 1, "1위 후보와 입력 정보가 충돌하고, 더 적합할 수 있는 상위 후보가 있습니다."
    if better_alt:
        return "🔴 높음", "더 적합한 후보 존재", 1, "1위 외 후보가 이름·선종·국적 기준에서 더 자연스러워 우선 확인이 필요합니다."
    if top_type_bad or top_flag_bad:
        return "🔴 높음", "후보 정보 충돌", 1, "1위 후보의 선종 또는 국적이 입력 정보와 충돌합니다."
    if reason == "CROSS_SCRIPT_SAFETY_REVIEW":
        return "🟠 중간", "이름 표기 확인", 2, "한글·영문 표기 차이가 있어 동일 선박인지 확인이 필요합니다."
    if missing_items:
        return "🟡 낮음", "정보 부족", 3, f"{', '.join(missing_items)} 정보가 없어 후보 구분 근거가 부족합니다."
    if pd.notna(margin):
        return "🟠 중간", "후보 경합", 2, f"1·2위 점수 차이가 {float(margin):.4f}로 자동 확정 기준을 통과하지 못했습니다."
    return "🟠 중간", "일반 검토", 2, "자동 식별 기준을 통과하지 못해 후보 확인이 필요합니다."



def build_user_tables(results, candidates):
    # V7 display-only status: keep backend match_status unchanged.
    results = results.copy()
    if "review_category" in results.columns:
        info_mask = (
            results["match_status"].eq("REVIEW_REQUIRED")
            & results["review_category"].eq("INFORMATION_INSUFFICIENT")
        )
        results["display_status"] = results["match_status"].map(STATUS_LABELS).fillna(
            results["match_status"]
        )
        results.loc[info_mask, "display_status"] = "정보 부족"
    else:
        results["display_status"] = results["match_status"].map(STATUS_LABELS).fillna(
            results["match_status"]
        )

    """Presentation only: retain backend ranks, scores and decisions exactly."""
    r = results.reset_index(drop=True).copy()
    base = pd.DataFrame(index=r.index)
    base["service_row_id"] = r["service_row_id"]
    base["행번호"] = range(1, len(r) + 1)
    base["입력상태"] = r["input_status"].map(INPUT_STATUS_LABELS).fillna("확인 필요")
    # V7: 사용자 화면/다운로드에는 REVIEW_REQUIRED를 두 종류로 표시한다.
    # 백엔드 match_status 자체는 변경하지 않는다.
    base["처리결과"] = r["display_status"].fillna("확인 필요")

    review_category = (
        r["review_category"]
        if "review_category" in r.columns
        else pd.Series("", index=r.index, dtype=object)
    )
    missing_info = (
        r["missing_info_needed"]
        if "missing_info_needed" in r.columns
        else pd.Series("", index=r.index, dtype=object)
    )
    review_message = (
        r["review_message"]
        if "review_message" in r.columns
        else pd.Series("", index=r.index, dtype=object)
    )

    base["검토구분"] = review_category.map({
        "INFORMATION_INSUFFICIENT": "정보 부족",
        "GENERAL_REVIEW": "일반 검토",
    }).fillna("")
    base["확인필요정보"] = missing_info.fillna("")
    base["안내"] = review_message.fillna("")
    base["처리메시지"] = [result_message(row) for row in r.to_dict("records")]

    # 정보 부족은 백엔드가 만든 구체적 안내 문구를 우선 사용한다.
    info_message_mask = base["검토구분"].eq("정보 부족") & base["안내"].astype(str).str.strip().ne("")
    base.loc[info_message_mask, "처리메시지"] = base.loc[info_message_mask, "안내"]
    for source, label in INPUT_FIELDS.items():
        base[label] = r[source + "_raw"] if source + "_raw" in r else r[source]
    base["검색후보수"] = r["candidate_count"]
    base["1·2위점수차이"] = r["score_margin"]
    c = candidates.copy()
    for col in ["service_row_id", "m6_rank", "m6_score", *CANDIDATE_FIELDS, *FEATURE_FIELDS]:
        if col not in c:
            c[col] = pd.Series(index=c.index, dtype=object)
    c = c.sort_values(["service_row_id", "m6_rank"], kind="mergesort")

    # Display-only review triage: classify each REVIEW_REQUIRED input once.
    review_meta = {}
    grouped_candidates = {key: grp for key, grp in c.groupby("service_row_id", sort=False)}
    for _, rr in r.iterrows():
        sid = rr["service_row_id"]
        grp = grouped_candidates.get(sid, c.iloc[0:0])
        review_meta[sid] = classify_review_record(rr, grp)

    base["위험도"] = base["service_row_id"].map(lambda sid: review_meta.get(sid, ("", "", 99, ""))[0])
    base["검토유형"] = base["service_row_id"].map(lambda sid: review_meta.get(sid, ("", "", 99, ""))[1])
    base["_위험도순서"] = base["service_row_id"].map(lambda sid: review_meta.get(sid, ("", "", 99, ""))[2])
    base["검토사유상세"] = base["service_row_id"].map(lambda sid: review_meta.get(sid, ("", "", 99, ""))[3])

    top = c.loc[c["m6_rank"].eq(1), ["service_row_id", *CANDIDATE_FIELDS]]
    top = top.rename(columns={k: v.replace("후보_", "1순위후보_") for k, v in CANDIDATE_FIELDS.items()})
    top = top.rename(columns={"1순위후보_선박ID": "1순위후보_ID"})
    all_rows = base.merge(top, on="service_row_id", how="left", validate="one_to_one", sort=False)
    all_rows["모델점수"] = r["rank1_score"].to_numpy()
    all_cols = ["행번호", "입력상태", "처리결과", "위험도", "검토유형", "검토사유상세", "검토구분", "확인필요정보", "안내", "처리메시지",
                *INPUT_FIELDS.values(),
                "1순위후보_ID", "1순위후보_선박명", "1순위후보_GT", "1순위후보_선종", "1순위후보_국적",
                "모델점수", "검색후보수", "1·2위점수차이"]
    all_rows = all_rows[all_cols]
    detail = c.merge(base, on="service_row_id", how="left", validate="many_to_one", sort=False)
    detail = detail.rename(columns={**CANDIDATE_FIELDS, "m6_rank": "후보순위", "m6_score": "모델점수"})
    detail["이름비교"] = [name_label(row) for row in detail.to_dict("records")]
    detail["GT차이"] = detail["gt_abs_diff"].where(detail["gt_missing"].ne(1))
    detail["선종비교"] = detail["type_comparison_status"].map(comparison_label)
    detail["국적비교"] = detail["flag_comparison_status"].map(comparison_label)
    detail["검토사유"] = detail["처리메시지"]
    review_cols = ["행번호", "처리결과", "위험도", "검토유형", "_위험도순서", "검토사유상세", "검토구분", "확인필요정보", "안내", "검토사유",
                   *INPUT_FIELDS.values(), "검색후보수", "1·2위점수차이",
                   "후보순위", *CANDIDATE_FIELDS.values(), "이름비교", "GT차이", "선종비교", "국적비교", "모델점수"]
    review = detail.loc[
        detail["처리결과"].isin(["검토 필요", "정보 부족"]) & detail["후보순위"].le(5),
        review_cols
    ].copy()
    review = review.sort_values(
        ["_위험도순서", "행번호", "후보순위"], kind="mergesort"
    ).reset_index(drop=True)
    for col in ["최종선택_선박ID", "검토결과", "검토메모"]:
        review[col] = ""
    analysis_cols = ["행번호", "후보순위", *INPUT_FIELDS.values(), *CANDIDATE_FIELDS.values(),
                     "모델점수", *FEATURE_FIELDS, "검색후보수", "1·2위점수차이",
                     "처리결과", "검토구분", "확인필요정보", "안내"]
    analysis = detail.loc[detail["후보순위"].le(10), analysis_cols].copy()
    mask = r["match_status"].isin(["NOT_PROCESSED", "NO_CANDIDATE", "SYSTEM_ERROR"]) | r["input_status"].eq("INVALID")
    errors = base.loc[mask, ["행번호", *INPUT_FIELDS.values(), "입력상태", "처리결과"]].copy()
    descriptions = [error_details(row) for row in r.loc[mask].to_dict("records")]
    errors["오류사유"] = [x[0] for x in descriptions]
    errors["수정안내"] = [x[1] for x in descriptions]
    return all_rows, review, errors, analysis


def review_report_bytes(results, review):
    """Standalone readable report. Escape all source text; no external assets."""
    def safe(value):
        return html.escape(str(value), quote=True)

    def number(value, digits=None):
        n = pd.to_numeric(value, errors="coerce")
        if pd.isna(n):
            return "정보 없음"
        if digits is not None:
            return f"{n:.{digits}f}"
        return f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"

    headings = ["순위", "후보 선박명", "후보 GT", "후보 선종", "후보 국적",
                "이름 비교", "GT 차이", "선종 비교", "국적 비교", "모델 점수"]
    parts = []
    indexed = results.set_index("행번호")
    for row_id, group in review.groupby("행번호", sort=False):
        row = indexed.loc[row_id]
        fields = [("선박명", row["입력_선박명"]), ("GT", number(row["입력_GT"])),
                  ("선종", row["입력_선종"]), ("국적", row["입력_국적"])]
        info = "".join(f"<dt>{safe(k)}</dt><dd>{safe(v) if str(v).strip() else '정보 없음'}</dd>" for k, v in fields)
        table_rows = []
        for _, candidate in group.iterrows():
            values = [candidate["후보순위"], candidate["후보_선박명"], number(candidate["후보_GT"]),
                      candidate["후보_선종"], candidate["후보_국적"], candidate["이름비교"],
                      number(candidate["GT차이"]), candidate["선종비교"], candidate["국적비교"],
                      number(candidate["모델점수"], 4)]
            table_rows.append("<tr>" + "".join(f"<td>{safe(v)}</td>" for v in values) + "</tr>")
        parts.append(
            f'<section><h2>입력 기록 #{safe(row_id)} <span>{safe(group.iloc[0].get("위험도", ""))} · {safe(group.iloc[0].get("검토유형", "검토 필요"))}</span></h2><dl>{info}</dl>'
            f'<div class="reason"><strong>검토 사유</strong><br>{safe(group.iloc[0].get("검토사유상세", row["처리메시지"]))}<br>'
            f'검색 후보 : {safe(row["검색후보수"])}척<br>1·2위 차이 : {number(row["1·2위점수차이"], 4)}</div>'
            '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headings) +
            '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div></section>'
        )
    document = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>검토 필요 후보 비교 보고서</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111318;color:#eee;font-family:Arial,'Malgun Gothic',sans-serif;line-height:1.65}
main{max-width:1450px;margin:auto;padding:28px}h1{font-size:26px}h2{font-size:20px;margin-top:0}
section{background:#1b1e25;border:1px solid #383d48;border-radius:12px;padding:26px;margin:24px 0}
span{font-size:14px;color:#f5cb76;margin-left:18px}dl{display:grid;grid-template-columns:90px 1fr;gap:4px 16px;margin:24px 0}
dt{color:#bbc2ce}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.reason{margin:24px 0 30px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #414650;white-space:nowrap}
th{color:#bcc6d7}td:last-child{font-variant-numeric:tabular-nums}footer,.note{color:#bbc2ce;font-size:13px}
@media print{body{background:white;color:#111}main{padding:0}section{background:white;border:0;border-radius:0;padding:8px;break-before:page}section:first-of-type{break-before:auto}th,dt,footer,.note{color:#444}span{color:#555}table{font-size:10px}th,td{padding:6px 4px;white-space:normal}.scroll{overflow:visible}}
</style></head><body><main><h1>검토 필요 후보 비교</h1>"""
    document += f'<p class="note">검토 대상 {review["행번호"].nunique()}건 · 입력 기록별 최대 5개 후보</p>'
    document += ''.join(parts)
    document += '<footer>1순위 후보는 확정 선박이 아닙니다. 모델 점수는 정답 확률이 아닙니다. 이름 동일 및 선종·국적 비교는 정규화·표준화 기준입니다. 표시 숫자는 반올림되어 있으며 CSV에는 원래 정밀도가 유지됩니다.</footer></main></body></html>'
    return document.encode("utf-8")


def summary_report_bytes(frame, *, errors=False):
    """Reuse the review report style for readable all-results/error reports."""
    title = "입력 오류·미처리 보고서" if errors else "전체 처리결과 보고서"
    empty_results = pd.DataFrame(columns=["행번호"])
    shell = review_report_bytes(empty_results, empty_results).decode("utf-8")
    head = shell.split("<body>", 1)[0].replace("<title>검토 필요 후보 비교 보고서</title>", f"<title>{title}</title>")
    def value(x):
        return "정보 없음" if pd.isna(x) or str(x).strip() == "" else html.escape(str(x), quote=True)
    def fmt(x, digits=0):
        n = pd.to_numeric(x, errors="coerce")
        if pd.isna(n):
            return "정보 없음"
        if digits:
            return f"{n:.{digits}f}"
        return f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"
    counts = " · ".join(f"{value(k)} {v:,}건" for k, v in frame["처리결과"].value_counts().items())
    parts = [head, f'<body><main><h1>{title}</h1><p>총 {len(frame):,}건 · {counts}</p>']
    if frame.empty:
        parts.append('<p>해당 기록이 없습니다.</p>')
    for _, row in frame.iterrows():
        parts.append(f'<section><h2>입력 기록 #{value(row["행번호"])} <span>{value(row["처리결과"])}</span></h2><dl>')
        for label, col in [("선박명", "입력_선박명"), ("GT", "입력_GT"), ("선종", "입력_선종"), ("국적", "입력_국적")]:
            text = fmt(row[col]) if col == "입력_GT" and pd.notna(pd.to_numeric(row[col], errors="coerce")) else value(row[col])
            parts.append(f'<dt>{label}</dt><dd>{text}</dd>')
        parts.append('</dl>')
        if errors:
            parts.append(f'<div class="reason"><strong>오류 사유</strong><p>{value(row["오류사유"])}</p><strong>수정 안내</strong><p>{value(row["수정안내"])}</p></div>')
        else:
            parts.append(f'<p>{value(row["처리메시지"])}</p>')
            if str(row["1순위후보_ID"]).strip() and pd.notna(row["1순위후보_ID"]):
                cols = ["1순위후보_ID", "1순위후보_선박명", "1순위후보_GT", "1순위후보_선종", "1순위후보_국적", "모델점수"]
                headers = ["후보 ID", "후보 선박명", "후보 GT", "후보 선종", "후보 국적", "모델 점수"]
                parts.append('<h3>1순위 후보</h3><div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody><tr>')
                for col in cols:
                    text = fmt(row[col], 4) if col == "모델점수" else fmt(row[col]) if col == "1순위후보_GT" else value(row[col])
                    parts.append(f'<td>{text}</td>')
                parts.append('</tr></tbody></table></div>')
            parts.append(f'<p>검색 후보 : {fmt(row["검색후보수"])}척 · 1·2위 차이 : {fmt(row["1·2위점수차이"], 4) if pd.notna(pd.to_numeric(row["1·2위점수차이"], errors="coerce")) else "해당 없음"}</p>')
        parts.append('</section>')
    parts.append('<footer>읽기·인쇄용 보고서입니다. 검토 필요의 1순위 후보는 확정 선박이 아닙니다. 모델 점수는 정답 확률이 아닙니다. 표시 숫자는 반올림되어 있습니다.</footer></main></body></html>')
    return ''.join(parts).encode("utf-8")


def source_basename(filename):
    # Accept either slash convention and remove only the final extension.
    leaf = str(filename).replace(chr(92), "/").rsplit("/", 1)[-1]
    return Path(leaf).stem.strip() or "입력파일"


def download_basename(filename):
    name = source_basename(filename)
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:120] or "입력파일"


def named_report(payload, title):
    document = payload.decode("utf-8")
    escaped = html.escape(title, quote=True)
    for tag in ("title", "h1"):
        before, opening, rest = document.partition(f"<{tag}>")
        _, closing, after = rest.partition(f"</{tag}>")
        if opening and closing:
            document = before + opening + escaped + closing + after
    return document.encode("utf-8")


def clear_saved_result():
    for key in ["processed_output", "processed_input_name", "processed_input_size", "processed_input_hash"]:
        st.session_state.pop(key, None)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 항만 선박 식별 CSV 서비스")
st.caption(
    "CSV를 업로드하면 기존 검증 완료 파이프라인 "
    "(후보 검색 → 선박 순위 계산 → 자동/검토 판정)으로 처리합니다."
)

with st.expander("입력 CSV 형식", expanded=False):
    st.write("필수 컬럼")
    st.code(
        "vessel_name,gross_tonnage,vessel_type,flag_name",
        language="text",
    )
    st.write(
        "일부 값이 비어 있어도 처리 가능한 경우는 계속 진행합니다. "
        "처리할 수 없는 입력은 별도로 분리합니다."
    )

uploaded_file = st.file_uploader(
    "CSV 파일 업로드",
    type=["csv"],
)

if uploaded_file is None:
    st.info("처리할 CSV 파일을 선택해주세요.")
    st.stop()

try:
    input_df = read_uploaded_csv(uploaded_file)
except Exception:
    LOGGER.exception(
        "csv_read_failed | file_name=%s",
        getattr(uploaded_file, "name", "UNKNOWN"),
    )
    st.error(
        "CSV 파일을 읽을 수 없습니다. "
        "파일 형식과 문자 인코딩을 확인한 뒤 다시 시도해주세요."
    )
    st.stop()

reserved_columns = sorted((RESERVED_SERVICE_COLUMNS | {
    "service_row_id", "candidate_vessel_id", "m6_rank", "m6_score", "행번호",
    "처리결과", "후보순위", "입력_선박명", "1순위후보_ID"
}).intersection(input_df.columns))
if reserved_columns:
    LOGGER.warning("reserved_columns_detected")
    st.error("이 파일은 이전 선박 식별 결과 파일로 보입니다. 결과 CSV를 다시 입력 파일로 사용할 수 없습니다. 원본 입력 CSV를 업로드해주세요.")
    st.stop()
missing_columns = [col for col in REQUIRED_COLUMNS if col not in input_df.columns]
if missing_columns:
    LOGGER.warning("schema_error | missing_columns=%s", ",".join(missing_columns))
    st.error("필수 컬럼이 없습니다: " + ", ".join(missing_columns))
    st.stop()
if input_df.empty:
    st.error("입력 데이터가 없습니다. 데이터가 있는 CSV를 업로드해주세요.")
    st.stop()

if len(input_df) > MAX_INPUT_ROWS:
    LOGGER.warning(
        "input_too_large | file_name=%s | input_rows=%d | max_rows=%d",
        uploaded_file.name,
        len(input_df),
        MAX_INPUT_ROWS,
    )
    st.error(
        f"한 번에 처리할 수 있는 최대 행 수는 {MAX_INPUT_ROWS:,}건입니다. "
        f"현재 파일은 {len(input_df):,}건입니다. "
        "파일을 나누어 업로드해주세요."
    )
    st.stop()

current_file_name = uploaded_file.name
current_file_size = uploaded_file.size
current_file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
if st.session_state.get("processed_input_hash") != current_file_hash:
    clear_saved_result()

st.subheader("1. 업로드 확인")
c1, c2 = st.columns(2)
c1.metric("입력 행 수", f"{len(input_df):,}")
c2.metric("입력 컬럼 수", f"{len(input_df.columns):,}")

st.dataframe(
    input_df.head(20),
    use_container_width=True,
    hide_index=True,
)

run_clicked = st.button(
    "선박 식별 실행",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    clear_saved_result()
    try:
        with st.spinner(
            "검증된 모델과 Catalog를 준비하고 CSV를 처리하는 중입니다..."
        ):
            matcher = get_matcher()
            output = matcher.process_dataframe(input_df)

        st.session_state.processed_output = output
        st.session_state.processed_input_name = current_file_name
        st.session_state.processed_input_size = current_file_size
        st.session_state.processed_input_hash = current_file_hash

        LOGGER.info(
            "processing_success | input_rows=%d | processable=%d | "
            "auto_match=%d | review_required=%d | not_processed=%d | "
            "no_candidate=%d | candidate_pairs=%d | elapsed_seconds=%.3f | "
            "pipeline=%s | ranker=%s | artifact_sha256=%s",
            output.summary.total_rows,
            output.summary.processable_rows,
            output.summary.auto_match_rows,
            output.summary.review_required_rows,
            output.summary.not_processed_rows,
            output.summary.no_candidate_rows,
            output.summary.candidate_pairs,
            output.summary.elapsed_seconds,
            PIPELINE_VERSION,
            "M6_FINAL_FROZEN",
            ARTIFACT_HASH,
        )

    except Exception:
        LOGGER.exception(
            "processing_failed | file_name=%s | input_rows=%d",
            current_file_name,
            len(input_df),
        )
        st.error(
            "처리 중 오류가 발생했습니다. "
            "입력 파일을 확인한 뒤 다시 시도해주세요. "
            "문제가 계속되면 관리자에게 문의해주세요."
        )
        st.stop()

if "processed_output" not in st.session_state:
    st.info("파일을 확인한 뒤 '선박 식별 실행'을 눌러주세요.")
    st.stop()

output = st.session_state.processed_output

try:
    results_download, review_download, error_download, analysis_download = build_user_tables(output.results, output.candidates)
except Exception:
    LOGGER.exception("output_format_failed")
    st.error("결과 화면을 준비할 수 없습니다. 관리자에게 문의해주세요.")
    st.stop()

st.success("처리가 완료되었습니다.")
st.subheader("2. 처리 요약")
summary = output.summary
info_shortage_rows = 0
if "review_category" in output.results.columns:
    info_shortage_rows = int(
        (
            output.results["match_status"].eq("REVIEW_REQUIRED")
            & output.results["review_category"].eq("INFORMATION_INSUFFICIENT")
        ).sum()
    )
general_review_rows = int(summary.review_required_rows - info_shortage_rows)

metrics = [("전체 입력", summary.total_rows), ("처리 가능", summary.processable_rows),
           ("자동 식별", summary.auto_match_rows), ("정보 부족", info_shortage_rows),
           ("검토 필요", general_review_rows), ("처리 불가", summary.not_processed_rows)]
for start in (0, 3):
    for column, (label, value) in zip(st.columns(3), metrics[start:start+3]):
        column.metric(label, f"{value:,}")
st.caption(f"처리 시간: {summary.elapsed_seconds:.2f}초 · 후보별 비교 건수: {summary.candidate_pairs:,}개")
st.caption(
    "정보 부족은 기존 검토 필요 중, 누락된 GT·국적 정보가 현재 1·2위 후보를 "
    "구분하는 데 필요한 경우입니다. 자동 식별 기준과 모델 순위는 변경하지 않습니다."
)
st.subheader("3. 결과 미리보기")
st.dataframe(results_download.head(100), use_container_width=True, hide_index=True)
st.caption("최대 100행 미리보기입니다. 전체 입력 결과는 CSV에서 확인할 수 있습니다.")
st.subheader("4. 검토 필요 후보 비교")
if review_download.empty:
    st.info("검토가 필요한 기록이 없습니다.")
else:
    review_index = (
        review_download[["행번호", "위험도", "검토유형", "_위험도순서"]]
        .drop_duplicates("행번호")
        .sort_values(["_위험도순서", "행번호"], kind="mergesort")
    )
    ids = review_index["행번호"].tolist()
    review_label = review_index.set_index("행번호")[["위험도", "검토유형"]].to_dict("index")
    selected = st.selectbox(
        "검토할 입력 기록",
        ids,
        format_func=lambda n: f"{review_label[n]['위험도']} | #{n} | {review_label[n]['검토유형']}"
    )
    record = results_download.loc[results_download["행번호"].eq(selected)].iloc[0]
    selected_status = str(record.get("처리결과", "검토 필요"))
    priority = str(record.get("위험도", "")).strip()
    review_type = str(record.get("검토유형", "")).strip()
    st.write(f"입력 기록 #{selected} · {selected_status} · {priority} · {review_type}")
    if selected_status == "정보 부족":
        needed = str(record.get("확인필요정보", "")).strip()
        guide = str(record.get("안내", "")).strip()
        if needed:
            st.warning(f"확인 필요 정보: {needed}")
        if guide:
            st.info(guide)
    def display_gt(value):
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            return "정보 없음" if str(value).strip() == "" else str(value)
        return f"{number:,.0f}" if float(number).is_integer() else f"{number:,}"

    def display_text(value):
        return str(value).strip() or "정보 없음"

    st.text(
        f"선박명    {display_text(record['입력_선박명'])}\n"
        f"GT        {display_gt(record['입력_GT'])}\n"
        f"선종      {display_text(record['입력_선종'])}\n"
        f"국적      {display_text(record['입력_국적'])}"
    )
    margin = record["1·2위점수차이"]
    margin_text = f"{margin:.4f}" if pd.notna(margin) else "해당 없음"
    st.text(
        "검토 사유\n"
        f"{record.get('검토사유상세', record['처리메시지'])}\n"
        f"검색 후보 : {record['검색후보수']:,}척\n"
        f"1·2위 차이 : {margin_text}"
    )
    view_cols = ["후보순위", "후보_선박명", "후보_GT", "후보_선종", "후보_국적", "이름비교", "GT차이", "선종비교", "국적비교", "모델점수"]
    view = review_download.loc[review_download["행번호"].eq(selected), view_cols].rename(columns={
        "후보순위": "순위", "후보_선박명": "후보 선박명", "후보_GT": "후보 GT", "후보_선종": "후보 선종",
        "후보_국적": "후보 국적", "이름비교": "이름 비교", "GT차이": "GT 차이", "선종비교": "선종 비교",
        "국적비교": "국적 비교", "모델점수": "모델 점수"})
    view = view.copy()
    view["후보 GT"] = view["후보 GT"].map(display_gt)
    view["GT 차이"] = view["GT 차이"].map(lambda x: "정보 없음" if pd.isna(x) else display_gt(x))
    view["모델 점수"] = view["모델 점수"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "정보 없음")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("최대 5개 후보를 함께 표시합니다. 이름 동일은 정규화 기준이며, 선종·국적 비교도 표준화 기준입니다. 모델 점수는 정답 확률이 아닙니다.")
report_source = st.session_state.get("processed_input_name", current_file_name)
report_name = source_basename(report_source)
download_name = download_basename(report_source)
st.subheader("5. 결과 다운로드")
st.caption("일반 사용자용은 읽기·인쇄용 HTML 보고서입니다. 다운로드 후 더블클릭하면 브라우저에서 열립니다.")
for column, (label, payload, filename, empty) in zip(st.columns(3), [
    ("전체 처리결과 보고서", named_report(summary_report_bytes(results_download), f"{report_name} 결과"), f"{download_name}_결과.html", results_download.empty),
    ("검토필요 후보비교 보고서", named_report(review_report_bytes(results_download, review_download), f"{report_name} 검토 필요"), f"{download_name}_검토필요.html", review_download.empty),
    ("입력오류·미처리 보고서", named_report(summary_report_bytes(error_download, errors=True), f"{report_name} 입력 오류·미처리"), f"{download_name}_입력오류.html", error_download.empty),
]):
    column.download_button(label, data=payload, file_name=filename, mime="text/html",
                           use_container_width=True, disabled=empty)
with st.expander("분석가용 Top10 후보", expanded=False):
    st.download_button("분석용 Top10 CSV", data=dataframe_to_csv_bytes(analysis_download),
                       file_name=f"{download_name}_Top10.csv", mime="text/csv", disabled=analysis_download.empty)
# Local developer mode only; not an authentication mechanism.
if os.environ.get("VESSEL_DEVELOPER_MODE") == "1":
    with st.expander("개발자용 전체 후보 Raw", expanded=False):
        if st.checkbox("Raw CSV 준비 (최대 100후보/입력)"):
            raw_results = output.results.rename(columns={"match_status": "confidence_decision"})
            raw = output.candidates.merge(raw_results, on="service_row_id", how="left",
                                          validate="many_to_one", suffixes=("", "_input"))
            st.download_button("개발자용 Raw CSV", data=dataframe_to_csv_bytes(raw),
                               file_name=f"{download_name}_전체후보_Raw.csv", mime="text/csv", disabled=raw.empty)
st.caption("자동 식별은 자동 처리 기준을 통과한 결과입니다. 정보 부족과 검토 필요의 1순위 후보는 확정 선박이 아닙니다. 모델 점수는 정답 확률이 아닙니다.")
