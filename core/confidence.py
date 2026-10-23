"""
CONFIDENCE_GATE_V1

Frozen policy:
1. candidate_count == 1 -> AUTO_MATCH
2. score_margin >= 0.20 -> AUTO_MATCH
3. otherwise -> REVIEW_REQUIRED

Do not tune this threshold using Frozen TEST.
"""

CONFIDENCE_MARGIN_THRESHOLD = 0.20


def apply_confidence_gate(
    candidate_count,
    score_margin,
):
    if int(candidate_count) == 1:
        return "AUTO_MATCH"

    if float(score_margin) >= CONFIDENCE_MARGIN_THRESHOLD:
        return "AUTO_MATCH"

    return "REVIEW_REQUIRED"
