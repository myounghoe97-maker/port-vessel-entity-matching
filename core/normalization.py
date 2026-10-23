"""
Frozen normalization/mapping adapter for the port vessel-matching service.

Goals
-----
1. Reuse the verified STEP 6 name normalization exactly.
2. Reuse verified Type / Flag mapping tables.
3. Preserve raw values and mapping status.
4. Do not crash an entire uploaded CSV because a new raw Type/Flag value
   was not seen in the historical mapping tables.
5. For known historical values, mapped outputs must equal the frozen
   research pipeline.

This module expects core.validation to run first.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

STEP6_FILE = ROOT / "06_build_pair_features.py"
TYPE_MAP_FILE = (
    ROOT
    / "results_step4b2"
    / "pilot_vessel_type_mapping.csv"
)
FLAG_MAP_FILE = (
    ROOT
    / "results_step4a_verified"
    / "flag_canonical_mapping.csv"
)


def read_mapping_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def load_step6():
    """
    Load the verified STEP 6 module and reuse normalize_name exactly.
    """
    if not STEP6_FILE.exists():
        raise FileNotFoundError(STEP6_FILE)

    spec = importlib.util.spec_from_file_location(
        "step6_service",
        STEP6_FILE,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mapping_lookup(
    mapping: pd.DataFrame,
    source: str,
    key: str,
):
    subset = (
        mapping[
            mapping["source"].eq(source)
        ]
        .copy()
    )

    if not subset[key].is_unique:
        raise AssertionError(
            f"Mapping key not unique: source={source}, key={key}"
        )

    return subset.set_index(key)


def map_with_status(
    raw_values: pd.Series,
    *,
    mapping: pd.DataFrame,
    source: str,
    key: str,
    value: str,
):
    """
    Map one raw categorical field.

    For rows already represented in the frozen mapping table:
      - mapping_status is preserved.
      - mapped value is returned only when mapping_status == "MAPPED".

    For a genuinely unseen non-empty raw value:
      - status = "UNSEEN_RAW"
      - mapped value = ""

    Missing raw values:
      - status = "MISSING"
      - mapped value = ""
    """
    lookup = _build_mapping_lookup(
        mapping,
        source,
        key,
    )

    raw = (
        raw_values
        .fillna("")
        .astype(str)
    )

    status = raw.map(
        lookup["mapping_status"]
    )

    mapped = raw.map(
        lookup[value]
    )

    missing_mask = raw.eq("")
    unseen_mask = (
        raw.ne("")
        & status.isna()
    )

    status = status.astype("object")

    status.loc[missing_mask] = "MISSING"
    status.loc[unseen_mask] = "UNSEEN_RAW"

    mapped_out = (
        mapped
        .where(
            status.eq("MAPPED"),
            "",
        )
        .fillna("")
        .astype(str)
    )

    return mapped_out, status.fillna("")


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "Train",
    type_map: pd.DataFrame | None = None,
    flag_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Append frozen normalized/mapped service columns.

    Required input columns:
      vessel_name
      vessel_type
      flag_name

    gross_tonnage parsing belongs to core.validation.
    """
    required = {
        "vessel_name",
        "vessel_type",
        "flag_name",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing normalization columns: "
            + ", ".join(sorted(missing))
        )

    if type_map is None:
        if not TYPE_MAP_FILE.exists():
            raise FileNotFoundError(TYPE_MAP_FILE)
        type_map = read_mapping_csv(
            TYPE_MAP_FILE
        )

    if flag_map is None:
        if not FLAG_MAP_FILE.exists():
            raise FileNotFoundError(FLAG_MAP_FILE)
        flag_map = read_mapping_csv(
            FLAG_MAP_FILE
        )

    step6 = load_step6()

    result = df.copy()

    result[
        "name_normalized"
    ] = (
        result["vessel_name"]
        .fillna("")
        .astype(str)
        .map(step6.normalize_name)
    )

    type_value, type_status = (
        map_with_status(
            result["vessel_type"],
            mapping=type_map,
            source=source,
            key="raw_vessel_type",
            value="broad_family",
        )
    )

    flag_value, flag_status = (
        map_with_status(
            result["flag_name"],
            mapping=flag_map,
            source=source,
            key="raw_flag",
            value="canonical_iso3",
        )
    )

    result[
        "type_broad_family"
    ] = type_value
    result[
        "type_mapping_status"
    ] = type_status

    result[
        "flag_iso3"
    ] = flag_value
    result[
        "flag_mapping_status"
    ] = flag_status

    return result
