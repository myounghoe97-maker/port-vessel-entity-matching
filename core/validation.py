"""
Input validation for the port vessel entity-matching service.

Service V1 schema:
- vessel_name
- gross_tonnage
- vessel_type
- flag_name

Important policy:
- Missing != mismatch.
- Missing values may still be processable.
- Invalid values are not silently converted into missing values.
- Raw values are preserved.
- A row is processable only when:
    1) GT is not invalid, and
    2) at least one retrieval anchor exists:
       vessel_name OR valid gross_tonnage.

This module does NOT perform matching or model scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "vessel_name",
    "gross_tonnage",
    "vessel_type",
    "flag_name",
]

STATUS_VALID = "VALID"
STATUS_MISSING_PROCESSABLE = "MISSING_BUT_PROCESSABLE"
STATUS_INVALID = "INVALID"

MATCH_NOT_PROCESSED = "NOT_PROCESSED"
MATCH_PENDING = "PENDING"


@dataclass(frozen=True)
class ValidationSummary:
    total_rows: int
    valid_rows: int
    missing_but_processable_rows: int
    invalid_rows: int


def _raw_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _clean_text(value) -> str:
    return _raw_text(value).strip()


def validate_schema(df: pd.DataFrame) -> None:
    """
    Raise ValueError if the V1 service schema is not present.
    """
    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )


def parse_gt(value):
    """
    Returns:
        (status, parsed_value, reason)

    status:
        VALID / MISSING / INVALID
    """
    text = _clean_text(value)

    if text == "":
        return "MISSING", np.nan, "GT_MISSING"

    # Common thousands separators are accepted for parsing,
    # while the original raw string is preserved separately.
    normalized = text.replace(",", "")

    try:
        number = float(normalized)
    except ValueError:
        return "INVALID", np.nan, "GT_NOT_NUMERIC"

    if not np.isfinite(number):
        return "INVALID", np.nan, "GT_NOT_FINITE"

    if number <= 0:
        return "INVALID", np.nan, "GT_NON_POSITIVE"

    return "VALID", float(number), ""


def validate_row(row: pd.Series) -> dict:
    """
    Validate one input row without mutating original values.
    """
    vessel_name_raw = _raw_text(
        row.get("vessel_name", "")
    )
    gross_tonnage_raw = _raw_text(
        row.get("gross_tonnage", "")
    )
    vessel_type_raw = _raw_text(
        row.get("vessel_type", "")
    )
    flag_name_raw = _raw_text(
        row.get("flag_name", "")
    )

    vessel_name_clean = vessel_name_raw.strip()
    vessel_type_clean = vessel_type_raw.strip()
    flag_name_clean = flag_name_raw.strip()

    gt_status, gt_parsed, gt_reason = parse_gt(
        gross_tonnage_raw
    )

    name_missing = vessel_name_clean == ""
    gt_missing = gt_status == "MISSING"
    type_missing = vessel_type_clean == ""
    flag_missing = flag_name_clean == ""

    reasons = []

    if gt_status == "INVALID":
        reasons.append(gt_reason)

    # R_B100 can use exact/fuzzy name or GT-based G3 fallback.
    # If both name and GT are unavailable, retrieval cannot start.
    no_retrieval_anchor = (
        name_missing
        and gt_missing
    )

    if no_retrieval_anchor:
        reasons.append(
            "NAME_AND_GT_BOTH_MISSING"
        )

    if reasons:
        input_status = STATUS_INVALID
        processable = False
        match_status = MATCH_NOT_PROCESSED
    else:
        any_missing = (
            name_missing
            or gt_missing
            or type_missing
            or flag_missing
        )

        if any_missing:
            input_status = (
                STATUS_MISSING_PROCESSABLE
            )
        else:
            input_status = STATUS_VALID

        processable = True
        match_status = MATCH_PENDING

    return {
        # Preserve originals
        "vessel_name_raw": vessel_name_raw,
        "gross_tonnage_raw": gross_tonnage_raw,
        "vessel_type_raw": vessel_type_raw,
        "flag_name_raw": flag_name_raw,

        # Parsed / inspection fields
        "vessel_name_clean": vessel_name_clean,
        "gross_tonnage_parsed": gt_parsed,
        "gross_tonnage_status": gt_status,
        "vessel_type_clean": vessel_type_clean,
        "flag_name_clean": flag_name_clean,

        # Missing flags
        "name_missing": int(name_missing),
        "gt_missing": int(gt_missing),
        "type_missing": int(type_missing),
        "flag_missing": int(flag_missing),

        # Service decision
        "input_status": input_status,
        "input_reason": "|".join(reasons),
        "processable": bool(processable),
        "match_status": match_status,
    }


def validate_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, ValidationSummary]:
    """
    Validate an uploaded CSV dataframe row by row.

    Original dataframe columns are retained.
    Validation columns are appended.
    """
    validate_schema(df)

    original = df.copy()

    validation_rows = [
        validate_row(row)
        for _, row in original.iterrows()
    ]

    validation_df = pd.DataFrame(
        validation_rows,
        index=original.index,
    )

    # Avoid duplicate names if input CSV already happens
    # to contain service-generated columns.
    overlap = [
        c for c in validation_df.columns
        if c in original.columns
    ]

    if overlap:
        raise ValueError(
            "Input CSV already contains reserved "
            "service columns: "
            + ", ".join(overlap)
        )

    result = pd.concat(
        [original, validation_df],
        axis=1,
    )

    counts = result[
        "input_status"
    ].value_counts()

    summary = ValidationSummary(
        total_rows=len(result),
        valid_rows=int(
            counts.get(
                STATUS_VALID,
                0,
            )
        ),
        missing_but_processable_rows=int(
            counts.get(
                STATUS_MISSING_PROCESSABLE,
                0,
            )
        ),
        invalid_rows=int(
            counts.get(
                STATUS_INVALID,
                0,
            )
        ),
    )

    return result, summary
