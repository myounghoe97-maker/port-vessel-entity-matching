"""
STEP 27-C
Batch pipeline optimized for service startup.

Change:
- Load the frozen fitted M6 artifact from results_step27 instead of refitting M6
  on every service startup.

Unchanged:
- R_B100_V1 retrieval
- M6 feature construction
- M6 fitted parameters
- Frozen Confidence Gate V1 (service review overlay added below)
- Frozen TEST result files

Safety:
- Verify model artifact SHA256 against metadata before loading.
- No TEST tuning.
- No 19GB Hard Negative Pool scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from core.confidence import (
    CONFIDENCE_MARGIN_THRESHOLD,
    apply_confidence_gate,
)
from core.review_safety import apply_cross_script_review, apply_name_flag_review
from core.features import add_m6_features
from core.normalization import normalize_dataframe
from core.ranker import CATEGORICAL, M6_NUMERIC
from core.retrieval import build_retrieval_index, retrieve_positions
from core.validation import validate_dataframe


ROOT = Path(__file__).resolve().parent.parent

RULE_FILE = ROOT / "results_step15" / "step15_13f_frozen_rule_registry.csv"
CATALOG_FILE = ROOT / "results_step4b2" / "catalog_with_broad_family.csv"

MODEL_ARTIFACT_FILE = ROOT / "results_step27" / "m6_final_frozen_service.joblib"
MODEL_METADATA_FILE = ROOT / "results_step27" / "m6_final_frozen_service_metadata.json"

MODEL_FEATURES = M6_NUMERIC + CATEGORICAL

MATCH_NOT_PROCESSED = "NOT_PROCESSED"
MATCH_NO_CANDIDATE = "NO_CANDIDATE"
MATCH_AUTO = "AUTO_MATCH"
MATCH_REVIEW = "REVIEW_REQUIRED"

PIPELINE_VERSION = "BATCH_PIPELINE_V7_V42_KL_SAFETY_INFO_GAP"
RETRIEVAL_VERSION = "R_B100_V1"
RANKER_VERSION = "M6_FINAL_FROZEN"
CONFIDENCE_VERSION = "CONFIDENCE_GATE_V4_NAME_FLAG_TOLERANCE_WITH_V42_RANKING"


@dataclass
class BatchSummary:
    total_rows: int
    processable_rows: int
    not_processed_rows: int
    no_candidate_rows: int
    auto_match_rows: int
    review_required_rows: int
    candidate_pairs: int
    elapsed_seconds: float


@dataclass
class BatchOutput:
    results: pd.DataFrame
    candidates: pd.DataFrame
    summary: BatchSummary


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_str(path: Path, *, usecols=None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        low_memory=False,
    )


def _load_frozen_rules() -> list[dict]:
    if not RULE_FILE.exists():
        raise FileNotFoundError(RULE_FILE)

    rules_df = pd.read_csv(
        RULE_FILE,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if len(rules_df) != 23:
        raise AssertionError(
            f"Unexpected Korean-Latin rule count: {len(rules_df)}"
        )

    if not rules_df["registry_status"].eq("FROZEN").all():
        raise AssertionError("Korean-Latin registry is not frozen")

    return (
        rules_df[
            ["rule_id", "position", "hangul_part", "latin_part"]
        ]
        .to_dict("records")
    )


def _load_frozen_m6_artifact():
    if not MODEL_ARTIFACT_FILE.exists():
        raise FileNotFoundError(MODEL_ARTIFACT_FILE)

    if not MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(MODEL_METADATA_FILE)

    with MODEL_METADATA_FILE.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_sha = meta.get("artifact_sha256", "")
    actual_sha = _sha256_file(MODEL_ARTIFACT_FILE)

    if not expected_sha:
        raise AssertionError("Model metadata has no artifact_sha256")

    if actual_sha != expected_sha:
        raise AssertionError(
            "Frozen M6 artifact hash mismatch. "
            "Refusing to load a changed artifact."
        )

    expected_features = list(MODEL_FEATURES)
    if meta.get("model_features") != expected_features:
        raise AssertionError(
            "Frozen M6 artifact feature metadata mismatch"
        )

    if meta.get("frozen_ranker") != "M6_FINAL_FROZEN":
        raise AssertionError(
            "Unexpected frozen_ranker in model metadata"
        )

    return joblib.load(MODEL_ARTIFACT_FILE)


def _normalize_catalog() -> pd.DataFrame:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(CATALOG_FILE)

    catalog = _read_csv_str(CATALOG_FILE)

    required = {
        "vessel_id",
        "vessel_name",
        "gross_tonnage",
        "vessel_type",
        "flag_name",
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(
            "Catalog missing columns: " + ", ".join(sorted(missing))
        )

    catalog = normalize_dataframe(
        catalog,
        source="Catalog",
    )

    catalog["gross_tonnage_numeric"] = pd.to_numeric(
        catalog["gross_tonnage"].replace("", np.nan),
        errors="raise",
    )

    return catalog


def _build_catalog_index(catalog: pd.DataFrame):
    return build_retrieval_index(
        catalog_ids=catalog["vessel_id"],
        catalog_names=catalog["name_normalized"],
        catalog_gt=catalog["gross_tonnage_numeric"],
        catalog_type=catalog["type_broad_family"],
        catalog_flag=catalog["flag_iso3"],
    )


def _status_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _compare_status_one(
    left_value,
    right_value,
    left_status,
    right_status,
    *,
    prefix: str = "",
) -> str:
    lv = _status_text(left_value)
    rv = _status_text(right_value)
    ls = _status_text(left_status)
    rs = _status_text(right_status)

    statuses = {ls, rs}

    if "MISSING" in statuses:
        result = "MISSING"
    elif "AMBIGUOUS" in statuses:
        result = "AMBIGUOUS"
    elif ls != "MAPPED" or rs != "MAPPED":
        result = "UNMAPPED"
    else:
        result = "MATCH" if lv == rv else "MISMATCH"

    return prefix + result


def _add_base_pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()

    train_name = out["train_name_normalized"].fillna("").astype(str)
    catalog_name = out["catalog_name_normalized"].fillna("").astype(str)

    out["name_missing"] = (
        train_name.eq("") | catalog_name.eq("")
    ).astype(int)

    out["name_normalized_exact"] = (
        train_name.ne("")
        & catalog_name.ne("")
        & train_name.eq(catalog_name)
    ).astype(int)

    train_gt = pd.to_numeric(
        out["train_gt"].replace("", np.nan),
        errors="coerce",
    )
    catalog_gt = pd.to_numeric(
        out["catalog_gt"].replace("", np.nan),
        errors="coerce",
    )

    out["gt_missing"] = (
        train_gt.isna() | catalog_gt.isna()
    ).astype(int)

    out["gt_abs_diff"] = (train_gt - catalog_gt).abs()

    out["type_comparison_status"] = [
        _compare_status_one(
            lv, rv, ls, rs, prefix="TYPE_"
        )
        for lv, rv, ls, rs in zip(
            out["train_type_value"],
            out["catalog_type_value"],
            out["train_type_status"],
            out["catalog_type_status"],
        )
    ]

    out["flag_comparison_status"] = [
        _compare_status_one(lv, rv, ls, rs)
        for lv, rv, ls, rs in zip(
            out["train_flag_value"],
            out["catalog_flag_value"],
            out["train_flag_status"],
            out["catalog_flag_status"],
        )
    ]

    return out


def _apply_gate(
    candidate_count: int,
    score_margin: float,
) -> tuple[str, str]:
    raw = apply_confidence_gate(
        candidate_count,
        score_margin,
    )

    if isinstance(raw, tuple):
        decision = str(raw[0])
        reason = str(raw[1]) if len(raw) > 1 else ""
        return decision, reason

    decision = str(raw)

    if decision == MATCH_AUTO:
        if candidate_count == 1:
            return decision, "SINGLE_CANDIDATE"

        if (
            np.isfinite(score_margin)
            and score_margin >= CONFIDENCE_MARGIN_THRESHOLD
        ):
            return decision, "MARGIN_PASS"

    return decision, "REVIEW_REQUIRED"


def _apply_v42_kl_safety(ranked: pd.DataFrame) -> pd.DataFrame:
    """Apply validated V4.2 KL conflict safety rule by demoting, never deleting.

    Rule:
      korean_latin_rule_match == 1
      AND gt_missing == 0
      AND gt_abs_diff > 0
      AND flag_comparison_status == "MISMATCH"

    Non-gated candidates keep their original M6 score ordering, followed by
    gated candidates in their original M6 score ordering.
    """
    out = ranked.copy()

    kl_match = pd.to_numeric(
        out["korean_latin_rule_match"], errors="coerce"
    ).fillna(0).eq(1)
    gt_missing = pd.to_numeric(
        out["gt_missing"], errors="coerce"
    ).fillna(1).eq(0)
    gt_diff = pd.to_numeric(
        out["gt_abs_diff"], errors="coerce"
    ).fillna(0).gt(0)
    flag_mismatch = (
        out["flag_comparison_status"]
        .fillna("")
        .astype(str)
        .eq("MISMATCH")
    )

    out["v42_kl_safety_demoted"] = (
        kl_match & gt_missing & gt_diff & flag_mismatch
    ).astype(int)

    # 0 (safe/non-gated) first, then 1 (demoted); preserve frozen M6 ordering
    # inside each group and candidate ID as the deterministic final tie-break.
    out = out.sort_values(
        [
            "service_row_id",
            "v42_kl_safety_demoted",
            "m6_score",
            "candidate_vessel_id",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()

    out["m6_rank"] = (
        out.groupby("service_row_id", sort=False).cumcount() + 1
    )

    return out


def _classify_review_information_gap(
    decision: str,
    query_row: pd.Series,
    group: pd.DataFrame,
) -> tuple[str, str, str]:
    """Classify REVIEW_REQUIRED rows where missing input can distinguish top candidates.

    This is a presentation/review classification only.
    It never changes AUTO_MATCH to REVIEW_REQUIRED and never changes ranking.

    Rule V1:
      - final decision is REVIEW_REQUIRED
      - at least two ranked candidates exist
      - input GT and/or flag is missing
      - rank1 and rank2 have different usable values for that missing field

    Returns:
      review_category, missing_info_needed, review_message
    """
    if decision != MATCH_REVIEW or len(group) < 2:
        return "", "", ""

    rank1 = group.iloc[0]
    rank2 = group.iloc[1]
    needed = []

    query_gt_missing = bool(
        pd.to_numeric(
            pd.Series([query_row.get("gt_missing", 0)]),
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    if query_gt_missing:
        gt1 = pd.to_numeric(
            pd.Series([rank1.get("catalog_gt", np.nan)]),
            errors="coerce",
        ).iloc[0]
        gt2 = pd.to_numeric(
            pd.Series([rank2.get("catalog_gt", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(gt1) and pd.notna(gt2) and float(gt1) != float(gt2):
            needed.append("GT")

    query_flag_missing = bool(
        pd.to_numeric(
            pd.Series([query_row.get("flag_missing", 0)]),
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    if query_flag_missing:
        flag1 = _status_text(rank1.get("catalog_flag_value", ""))
        flag2 = _status_text(rank2.get("catalog_flag_value", ""))
        status1 = _status_text(rank1.get("catalog_flag_status", ""))
        status2 = _status_text(rank2.get("catalog_flag_status", ""))
        usable1 = bool(flag1) and status1 == "MAPPED"
        usable2 = bool(flag2) and status2 == "MAPPED"
        if usable1 and usable2 and flag1 != flag2:
            needed.append("국적")

    if not needed:
        return "GENERAL_REVIEW", "", ""

    needed_text = " + ".join(needed)
    message = (
        f"후보를 구분하는 데 필요한 {needed_text} 정보가 없습니다. "
        f"{needed_text} 정보를 확인하면 후보를 더 명확하게 비교할 수 있습니다."
    )
    return "INFORMATION_INSUFFICIENT", needed_text, message


class BatchMatcher:
    def __init__(self):
        start = perf_counter()

        self.rules = _load_frozen_rules()
        self.model = _load_frozen_m6_artifact()
        self.catalog = _normalize_catalog()
        self.retrieval_index = _build_catalog_index(self.catalog)

        self.initialization_seconds = perf_counter() - start

    def process_dataframe(self, df: pd.DataFrame) -> BatchOutput:
        start = perf_counter()

        validated, _ = validate_dataframe(df)
        normalized = normalize_dataframe(
            validated,
            source="Train",
        ).reset_index(drop=True)

        normalized.insert(
            0,
            "service_row_id",
            [
                f"ROW_{i:06d}"
                for i in range(1, len(normalized) + 1)
            ],
        )

        if "match_status" not in normalized.columns:
            normalized["match_status"] = MATCH_NOT_PROCESSED

        normalized["retrieval_branch"] = ""
        normalized["candidate_count"] = 0
        normalized["rank1_candidate_id"] = ""
        normalized["rank1_score"] = np.nan
        normalized["rank2_candidate_id"] = ""
        normalized["rank2_score"] = np.nan
        normalized["score_margin"] = np.nan
        normalized["confidence_reason"] = ""
        normalized["review_category"] = ""
        normalized["missing_info_needed"] = ""
        normalized["review_message"] = ""
        normalized["confidence_v1_decision"] = ""
        normalized["confidence_v1_reason"] = ""
        normalized["v42_kl_safety_applied"] = 0
        normalized["v42_kl_safety_demoted_count"] = 0
        normalized["pipeline_version"] = PIPELINE_VERSION
        normalized["retrieval_version"] = RETRIEVAL_VERSION
        normalized["ranker_version"] = RANKER_VERSION
        normalized["confidence_version"] = CONFIDENCE_VERSION

        pair_frames = []

        for row in normalized.itertuples(index=False):
            if not bool(row.processable):
                normalized.loc[
                    normalized["service_row_id"].eq(row.service_row_id),
                    "match_status",
                ] = MATCH_NOT_PROCESSED
                continue

            positions, branch = retrieve_positions(
                query_name=row.name_normalized,
                query_gt=row.gross_tonnage_parsed,
                query_type=row.type_broad_family,
                query_flag=row.flag_iso3,
                index=self.retrieval_index,
            )

            row_mask = normalized["service_row_id"].eq(
                row.service_row_id
            )

            normalized.loc[
                row_mask,
                "retrieval_branch",
            ] = branch

            normalized.loc[
                row_mask,
                "candidate_count",
            ] = len(positions)

            if len(positions) == 0:
                normalized.loc[
                    row_mask,
                    "match_status",
                ] = MATCH_NO_CANDIDATE

                normalized.loc[
                    row_mask,
                    "confidence_reason",
                ] = "NO_CANDIDATE"
                continue

            cat = self.catalog.iloc[positions].copy()

            pair_frames.append(
                pd.DataFrame(
                    {
                        "service_row_id": row.service_row_id,
                        "candidate_position": positions.astype(np.int32),
                        "candidate_vessel_id": cat[
                            "vessel_id"
                        ].astype(str).to_numpy(),
                        "retrieval_branch": branch,
                        "train_name_normalized": row.name_normalized,
                        "catalog_name_normalized": cat[
                            "name_normalized"
                        ].astype(str).to_numpy(),
                        "train_gt": row.gross_tonnage_parsed,
                        "catalog_gt": cat[
                            "gross_tonnage_numeric"
                        ].to_numpy(),
                        "train_type_value": row.type_broad_family,
                        "catalog_type_value": cat[
                            "type_broad_family"
                        ].astype(str).to_numpy(),
                        "train_type_status": row.type_mapping_status,
                        "catalog_type_status": cat[
                            "type_mapping_status"
                        ].astype(str).to_numpy(),
                        "train_flag_value": row.flag_iso3,
                        "catalog_flag_value": cat[
                            "flag_iso3"
                        ].astype(str).to_numpy(),
                        "train_flag_status": row.flag_mapping_status,
                        "catalog_flag_status": cat[
                            "flag_mapping_status"
                        ].astype(str).to_numpy(),
                        "catalog_vessel_name": cat[
                            "vessel_name"
                        ].astype(str).to_numpy(),
                        "catalog_gross_tonnage": cat[
                            "gross_tonnage"
                        ].astype(str).to_numpy(),
                        "catalog_vessel_type": cat[
                            "vessel_type"
                        ].astype(str).to_numpy(),
                        "catalog_flag_name": cat[
                            "flag_name"
                        ].astype(str).to_numpy(),
                    }
                )
            )

        if pair_frames:
            pairs = pd.concat(pair_frames, ignore_index=True)
            pairs = _add_base_pair_features(pairs)
            pairs = add_m6_features(pairs, self.rules)

            pairs["m6_score"] = self.model.predict_proba(
                pairs[MODEL_FEATURES]
            )[:, 1]

            ranked = pairs.sort_values(
                [
                    "service_row_id",
                    "m6_score",
                    "candidate_vessel_id",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            ).copy()

            # V4.2 validated KL conflict safety:
            # demote risky KL candidates without deleting any candidate.
            ranked = _apply_v42_kl_safety(ranked)

            for service_row_id, group in ranked.groupby(
                "service_row_id",
                sort=False,
            ):
                group = group.sort_values(
                    "m6_rank",
                    kind="mergesort",
                )

                count = len(group)

                v42_demoted_count = int(
                    pd.to_numeric(
                        group.get(
                            "v42_kl_safety_demoted",
                            pd.Series(0, index=group.index),
                        ),
                        errors="coerce",
                    ).fillna(0).sum()
                )
                v42_mask = normalized["service_row_id"].eq(service_row_id)
                normalized.loc[
                    v42_mask, "v42_kl_safety_applied"
                ] = int(v42_demoted_count > 0)
                normalized.loc[
                    v42_mask, "v42_kl_safety_demoted_count"
                ] = v42_demoted_count

                rank1 = group.iloc[0]
                rank1_score = float(rank1["m6_score"])

                if count >= 2:
                    rank2 = group.iloc[1]
                    rank2_id = str(rank2["candidate_vessel_id"])
                    rank2_score = float(rank2["m6_score"])
                    score_margin = rank1_score - rank2_score
                else:
                    rank2_id = ""
                    rank2_score = np.nan
                    score_margin = np.nan

                decision, reason = _apply_gate(
                    count,
                    score_margin,
                )

                # V2 service overlay: retain the original V1 decision for audit.
                query_row = normalized.loc[
                    normalized["service_row_id"].eq(service_row_id)
                ].iloc[0]
                v1_mask = normalized["service_row_id"].eq(service_row_id)
                normalized.loc[v1_mask, "confidence_v1_decision"] = decision
                normalized.loc[v1_mask, "confidence_v1_reason"] = reason
                decision, reason = apply_cross_script_review(
                    decision,
                    reason,
                    query_row.get("vessel_name_raw", query_row["vessel_name"]),
                    rank1["catalog_vessel_name"],
                )

                # Preserve V2 output, then apply the validated review-only rule.
                normalized.loc[v1_mask, "confidence_v2_decision"] = decision
                normalized.loc[v1_mask, "confidence_v2_reason"] = reason
                decision, reason, supporting_id = apply_name_flag_review(
                    decision, reason, group,
                )
                normalized.loc[v1_mask, "review_supporting_candidate_id"] = supporting_id

                # Information-gap V1: classify only rows that are already
                # REVIEW_REQUIRED after all existing safety overlays.
                # AUTO_MATCH and ranking are never changed here.
                review_category, missing_info_needed, review_message = (
                    _classify_review_information_gap(
                        decision,
                        query_row,
                        group,
                    )
                )
                normalized.loc[v1_mask, "review_category"] = review_category
                normalized.loc[v1_mask, "missing_info_needed"] = missing_info_needed
                normalized.loc[v1_mask, "review_message"] = review_message

                mask = normalized["service_row_id"].eq(
                    service_row_id
                )

                normalized.loc[
                    mask,
                    "match_status",
                ] = decision

                normalized.loc[
                    mask,
                    "rank1_candidate_id",
                ] = str(rank1["candidate_vessel_id"])

                normalized.loc[
                    mask,
                    "rank1_score",
                ] = rank1_score

                normalized.loc[
                    mask,
                    "rank2_candidate_id",
                ] = rank2_id

                normalized.loc[
                    mask,
                    "rank2_score",
                ] = rank2_score

                normalized.loc[
                    mask,
                    "score_margin",
                ] = score_margin

                normalized.loc[
                    mask,
                    "confidence_reason",
                ] = reason

            candidates = ranked

        else:
            candidates = pd.DataFrame(
                columns=[
                    "service_row_id",
                    "candidate_vessel_id",
                    "m6_score",
                    "m6_rank",
                ]
            )

        elapsed = perf_counter() - start
        match_counts = normalized["match_status"].value_counts()

        summary = BatchSummary(
            total_rows=len(normalized),
            processable_rows=int(
                normalized["processable"].sum()
            ),
            not_processed_rows=int(
                match_counts.get(MATCH_NOT_PROCESSED, 0)
            ),
            no_candidate_rows=int(
                match_counts.get(MATCH_NO_CANDIDATE, 0)
            ),
            auto_match_rows=int(
                match_counts.get(MATCH_AUTO, 0)
            ),
            review_required_rows=int(
                match_counts.get(MATCH_REVIEW, 0)
            ),
            candidate_pairs=len(candidates),
            elapsed_seconds=float(elapsed),
        )

        return BatchOutput(
            results=normalized,
            candidates=candidates,
            summary=summary,
        )

    def process_csv(self, path) -> BatchOutput:
        path = Path(path)

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        return self.process_dataframe(df)
