"""Service review overlay; the frozen V1 gate and ranking stay intact."""
import re

POLICY_VERSION = "CONFIDENCE_GATE_V4_NAME_FLAG_TOLERANCE"
REVIEW_REASON = "CROSS_SCRIPT_SAFETY_REVIEW"
REVIEW_MESSAGE = "한글·영문 이름 간 연결로, 동일 선박인지 추가 확인이 필요합니다."

def cross_script_names(left, right):
    left = "" if left is None else str(left)
    right = "" if right is None else str(right)
    lh = bool(re.search(r"[\uac00-\ud7a3]", left))
    ll = bool(re.search(r"[A-Za-z]", left))
    rh = bool(re.search(r"[\uac00-\ud7a3]", right))
    rl = bool(re.search(r"[A-Za-z]", right))
    return (lh and not ll and rl and not rh) or (ll and not lh and rh and not rl)

def apply_cross_script_review(decision, reason, input_name, candidate_name):
    if decision == "AUTO_MATCH" and cross_script_names(input_name, candidate_name):
        return "REVIEW_REQUIRED", REVIEW_REASON
    return decision, reason


NAME_FLAG_REASON = "NAME_FLAG_CONFLICT_REVIEW"
NAME_FLAG_MESSAGE = "1위 후보의 국적이 입력과 다르고, 이름이 더 비슷하면서 국적도 일치하는 상위 후보가 있어 검토가 필요합니다."


NAME_SIMILARITY_TOLERANCE = 0.015
NEAR_TIE_REASON = "NAME_FLAG_NEAR_TIE_REVIEW"
NEAR_TIE_MESSAGE = "1위 후보의 국적이 입력과 다르고, 이름 유사도가 비슷하면서 국적이 일치하는 상위 후보가 있어 검토가 필요합니다."


def name_flag_conflict_candidate(ranked_group, tolerance=None):
    """Return supporting rank2..5 ID; labels and target IDs are never used.

    Requires mapped, nonempty ISO3 on both sides. Missing/unmapped is not
    disagreement. Only review is allowed: no score, rank, or ID is modified.
    """
    import math
    import pandas as pd
    if ranked_group is None or ranked_group.empty:
        return ""
    required = {"m6_rank", "candidate_vessel_id", "name_similarity",
                "train_flag_status", "catalog_flag_status",
                "train_flag_value", "catalog_flag_value"}
    missing = required - set(ranked_group.columns)
    if missing:
        raise ValueError("Name/flag review fields missing: " + str(sorted(missing)))
    ranks = pd.to_numeric(ranked_group["m6_rank"], errors="raise")
    if ranks.duplicated().any() or not ranks.eq(1).any():
        raise ValueError("Invalid candidate ranks for name/flag review")
    group = ranked_group.loc[ranks.le(5)].copy()
    group["m6_rank"] = pd.to_numeric(group["m6_rank"], errors="raise")
    group = group.sort_values("m6_rank", kind="mergesort")

    def comparable_flag(row):
        if row["train_flag_status"] != "MAPPED" or row["catalog_flag_status"] != "MAPPED":
            return None
        a, b = row["train_flag_value"], row["catalog_flag_value"]
        if pd.isna(a) or pd.isna(b):
            return None
        a, b = str(a).strip(), str(b).strip()
        return a == b if a and b else None

    top = group.iloc[0]
    if comparable_flag(top) is not False:
        return ""
    top_sim = pd.to_numeric(top["name_similarity"], errors="coerce")
    if not math.isfinite(float(top_sim)):
        return ""
    for _, candidate in group.iloc[1:].iterrows():
        sim = pd.to_numeric(candidate["name_similarity"], errors="coerce")
        if (comparable_flag(candidate) is True and math.isfinite(float(sim))
                and (sim > top_sim + 1e-12 if tolerance is None
                     else sim >= top_sim - tolerance - 1e-12)):
            return str(candidate["candidate_vessel_id"])
    return ""


def apply_name_flag_review(decision, reason, ranked_group):
    if decision != "AUTO_MATCH":
        return decision, reason, ""
    candidate_id = name_flag_conflict_candidate(ranked_group)
    if candidate_id:
        return "REVIEW_REQUIRED", NAME_FLAG_REASON, candidate_id
    candidate_id = name_flag_conflict_candidate(ranked_group, NAME_SIMILARITY_TOLERANCE)
    if candidate_id:
        return "REVIEW_REQUIRED", NEAR_TIE_REASON, candidate_id
    return decision, reason, ""
