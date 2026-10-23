"""
Shared R_B100_V1 retrieval logic for the port entity-matching service.

Frozen policy:
- If normalized exact-name candidates exist:
    return exact-name candidates ordered by candidate_vessel_id, max 100.
- Otherwise:
    Fuzzy Name Top25
    UNION
    G3 Top75
    with fuzzy priority preserved, unseen G3 appended, max 100.

This module does NOT perform raw-field normalization or mapping.
It expects already-normalized arrays/columns so the service can reuse
the exact verified normalization pipeline separately.
"""

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


PREFILTER_K = 1000
FUZZY_K = 25
G3_K = 75
MAX_K = 100
GT_THRESHOLD = 2455.0


def make_name_index(values):
    groups = {}

    for i, value in enumerate(values):
        if value:
            groups.setdefault(
                value,
                [],
            ).append(i)

    return {
        key: np.asarray(
            positions,
            dtype=np.int32,
        )
        for key, positions
        in groups.items()
    }


def make_grams(text):
    text = str(text)

    if not text:
        return set()

    padded = "^" + text + "$"

    if len(padded) <= 3:
        return {padded}

    return {
        padded[i:i + 3]
        for i in range(
            len(padded) - 2
        )
    }


def seq_similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
        autojunk=False,
    ).ratio()


@dataclass
class RB100Index:
    catalog_ids: np.ndarray
    catalog_names: np.ndarray
    catalog_gt: np.ndarray
    catalog_type: np.ndarray
    catalog_flag: np.ndarray
    name_index: dict
    gram_index: dict
    catalog_gram_count: np.ndarray
    gt_order: np.ndarray
    gt_sorted: np.ndarray


def _as_text_array(values):
    s = pd.Series(values, dtype="object")
    return (
        s.fillna("")
        .astype(str)
        .to_numpy(dtype=object)
    )


def _as_gt_array(values):
    return pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)


def build_retrieval_index(
    catalog_ids,
    catalog_names,
    catalog_gt,
    catalog_type,
    catalog_flag,
):
    """
    Build catalog-side indexes once and reuse them for every uploaded CSV row.
    """

    catalog_ids = _as_text_array(catalog_ids)
    catalog_names = _as_text_array(catalog_names)
    catalog_gt = _as_gt_array(catalog_gt)
    catalog_type = _as_text_array(catalog_type)
    catalog_flag = _as_text_array(catalog_flag)

    n = len(catalog_ids)

    lengths = {
        len(catalog_names),
        len(catalog_gt),
        len(catalog_type),
        len(catalog_flag),
        n,
    }
    if len(lengths) != 1:
        raise ValueError(
            "Catalog arrays must have equal lengths."
        )

    name_index = make_name_index(
        catalog_names
    )

    gram_index = defaultdict(list)
    catalog_gram_count = np.zeros(
        n,
        dtype=np.int32,
    )

    for idx, name in enumerate(
        catalog_names
    ):
        grams = make_grams(name)

        catalog_gram_count[idx] = len(
            grams
        )

        for gram in grams:
            gram_index[gram].append(idx)

    gram_index = {
        gram: np.asarray(
            positions,
            dtype=np.int32,
        )
        for gram, positions
        in gram_index.items()
    }

    finite_gt = np.isfinite(
        catalog_gt
    )

    finite_positions = np.flatnonzero(
        finite_gt
    )

    if len(finite_positions) > 0:
        finite_ids = catalog_ids[
            finite_positions
        ]

        # Stable deterministic order:
        # GT first, vessel ID tie-break.
        local_order = np.lexsort(
            (
                finite_ids,
                catalog_gt[
                    finite_positions
                ],
            )
        )

        gt_order = finite_positions[
            local_order
        ].astype(
            np.int32,
            copy=False,
        )

        gt_sorted = catalog_gt[
            gt_order
        ]
    else:
        gt_order = np.empty(
            0,
            dtype=np.int32,
        )
        gt_sorted = np.empty(
            0,
            dtype=float,
        )

    return RB100Index(
        catalog_ids=catalog_ids,
        catalog_names=catalog_names,
        catalog_gt=catalog_gt,
        catalog_type=catalog_type,
        catalog_flag=catalog_flag,
        name_index=name_index,
        gram_index=gram_index,
        catalog_gram_count=catalog_gram_count,
        gt_order=gt_order,
        gt_sorted=gt_sorted,
    )


def retrieve_positions(
    query_name,
    query_gt,
    query_type,
    query_flag,
    index,
):
    """
    Apply frozen R_B100_V1 to one already-normalized record.

    Returns
    -------
    final_positions : np.ndarray[int32]
    branch : str
        EXACT or FUZZY_G3
    """

    query = (
        ""
        if query_name is None
        else str(query_name)
    )

    query_type = (
        ""
        if query_type is None
        else str(query_type)
    )

    query_flag = (
        ""
        if query_flag is None
        else str(query_flag)
    )

    try:
        gt_value = float(query_gt)
    except (TypeError, ValueError):
        gt_value = np.nan

    # --------------------------------------------------------
    # Branch A — Exact Name
    # --------------------------------------------------------

    exact_positions = (
        index.name_index.get(
            query,
            np.empty(
                0,
                dtype=np.int32,
            ),
        )
        if query
        else np.empty(
            0,
            dtype=np.int32,
        )
    )

    if len(exact_positions) > 0:
        ordered_exact = sorted(
            exact_positions.tolist(),
            key=lambda p:
                str(
                    index.catalog_ids[p]
                ),
        )

        final_positions = np.asarray(
            ordered_exact[:MAX_K],
            dtype=np.int32,
        )

        return final_positions, "EXACT"

    # --------------------------------------------------------
    # Branch B — Fuzzy25 UNION G3-75
    # --------------------------------------------------------

    # -------------------------
    # Fuzzy shortlist
    # -------------------------

    query_grams = make_grams(
        query
    )

    overlap_count = defaultdict(int)

    for gram in query_grams:
        for idx in index.gram_index.get(
            gram,
            [],
        ):
            overlap_count[
                int(idx)
            ] += 1

    fuzzy_positions = []

    if overlap_count:
        positions = np.fromiter(
            overlap_count.keys(),
            dtype=np.int32,
        )

        overlaps = np.array(
            [
                overlap_count[
                    int(idx)
                ]
                for idx
                in positions
            ],
            dtype=np.float32,
        )

        query_count = max(
            len(query_grams),
            1,
        )

        denominator = (
            query_count
            + index.catalog_gram_count[
                positions
            ]
            - overlaps
        )

        jaccard = (
            overlaps
            / denominator
        )

        ids = index.catalog_ids[
            positions
        ]

        length_diff = np.array(
            [
                abs(
                    len(query)
                    - len(
                        index.catalog_names[
                            idx
                        ]
                    )
                )
                for idx
                in positions
            ],
            dtype=np.int16,
        )

        cheap_order = np.lexsort(
            (
                ids,
                length_diff,
                -jaccard,
            )
        )

        shortlist = positions[
            cheap_order[
                :PREFILTER_K
            ]
        ]

        shortlist_ids = (
            index.catalog_ids[
                shortlist
            ]
        )

        seq_scores = np.array(
            [
                seq_similarity(
                    query,
                    index.catalog_names[
                        idx
                    ],
                )
                for idx
                in shortlist
            ],
            dtype=np.float32,
        )

        seq_order = np.lexsort(
            (
                shortlist_ids,
                -seq_scores,
            )
        )

        fuzzy_positions = (
            shortlist[
                seq_order[
                    :FUZZY_K
                ]
            ]
            .tolist()
        )

    # -------------------------
    # G3 fallback
    # -------------------------

    g3_positions = []

    if np.isfinite(
        gt_value
    ):
        lo = np.searchsorted(
            index.gt_sorted,
            gt_value - GT_THRESHOLD,
            side="left",
        )

        hi = np.searchsorted(
            index.gt_sorted,
            gt_value + GT_THRESHOLD,
            side="right",
        )

        positions = index.gt_order[
            lo:hi
        ]

        if len(positions) > 0:
            ids = index.catalog_ids[
                positions
            ]

            gt_diff = np.abs(
                index.catalog_gt[
                    positions
                ]
                - gt_value
            )

            type_match = (
                (query_type != "")
                &
                (
                    index.catalog_type[
                        positions
                    ]
                    == query_type
                )
            ).astype(
                np.int8
            )

            flag_match = (
                (query_flag != "")
                &
                (
                    index.catalog_flag[
                        positions
                    ]
                    == query_flag
                )
            ).astype(
                np.int8
            )

            signal_count = (
                type_match
                + flag_match
            )

            g3_order = np.lexsort(
                (
                    ids,
                    gt_diff,
                    -signal_count,
                )
            )

            g3_positions = (
                positions[
                    g3_order[
                        :G3_K
                    ]
                ]
                .tolist()
            )

    # -------------------------
    # Deterministic Union
    #
    # Preserve fuzzy priority,
    # then append unseen G3.
    # -------------------------

    combined = []
    seen = set()

    for pos in (
        fuzzy_positions
        + g3_positions
    ):
        pos = int(pos)

        if pos not in seen:
            seen.add(pos)
            combined.append(pos)

        if len(combined) >= MAX_K:
            break

    final_positions = np.asarray(
        combined,
        dtype=np.int32,
    )

    return final_positions, "FUZZY_G3"


def retrieve_candidates(
    query_name,
    query_gt,
    query_type,
    query_flag,
    index,
):
    """
    Return candidate metadata for one normalized input row.
    This is an internal building block for batch processing.
    """

    positions, branch = (
        retrieve_positions(
            query_name=query_name,
            query_gt=query_gt,
            query_type=query_type,
            query_flag=query_flag,
            index=index,
        )
    )

    out = pd.DataFrame(
        {
            "candidate_rank":
                np.arange(
                    1,
                    len(positions) + 1,
                    dtype=np.int32,
                ),
            "candidate_position":
                positions,
            "candidate_vessel_id":
                index.catalog_ids[
                    positions
                ],
        }
    )

    out["retrieval_branch"] = branch

    return out


def retrieve_batch(
    records,
    index,
    name_col,
    gt_col,
    type_col,
    flag_col,
    record_id_col=None,
):
    """
    Apply R_B100_V1 to every row in a dataframe.

    This is the service-facing batch entry point.
    """

    required = {
        name_col,
        gt_col,
        type_col,
        flag_col,
    }

    missing = required.difference(
        records.columns
    )

    if missing:
        raise KeyError(
            "Missing retrieval columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    rows = []

    for row_pos, row in enumerate(
        records.itertuples(
            index=False,
            name=None,
        )
    ):
        col_idx = {
            col: records.columns.get_loc(
                col
            )
            for col in required
        }

        record_id = (
            records.iloc[
                row_pos
            ][record_id_col]
            if record_id_col
            is not None
            else row_pos
        )

        candidates = (
            retrieve_candidates(
                query_name=row[
                    col_idx[name_col]
                ],
                query_gt=row[
                    col_idx[gt_col]
                ],
                query_type=row[
                    col_idx[type_col]
                ],
                query_flag=row[
                    col_idx[flag_col]
                ],
                index=index,
            )
        )

        if candidates.empty:
            continue

        candidates.insert(
            0,
            "record_id",
            record_id,
        )

        rows.append(
            candidates
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id",
                "candidate_rank",
                "candidate_position",
                "candidate_vessel_id",
                "retrieval_branch",
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )
