"""Synthetic fixtures only. No production model fitting or private data reads."""
import ast
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import batch_pipeline as bp
from core.confidence import apply_confidence_gate
from core.features import add_m6_features, build_kl_features
from core.normalization import load_step6, map_with_status
from core.ranker import build_model
from core.retrieval import build_retrieval_index, retrieve_positions
from core.review_safety import apply_cross_script_review, apply_name_flag_review
from core.validation import validate_dataframe

ROOT = Path(__file__).resolve().parents[1]


class FixedScores:
    """Test-only deterministic stub; never used by the shipped service."""
    def predict_proba(self, frame):
        scores = np.where(frame.name_normalized_exact.eq(1), 0.9, 0.1)
        return np.column_stack([1 - scores, scores])


class PublicSmoke(unittest.TestCase):
    def test_syntax_and_imports(self):
        for path in ROOT.rglob('*.py'):
            if '.git' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for name in ['validation', 'normalization', 'retrieval', 'features',
                     'ranker', 'confidence', 'review_safety', 'batch_pipeline']:
            importlib.import_module('core.' + name)
        self.assertEqual(build_model().named_steps['model'].__class__.__name__,
                         'LogisticRegression')
        self.assertEqual(load_step6().normalize_name(' Ａ-b_ c./ '), 'ABC')

    def test_validation_raw_and_invalid(self):
        frame = pd.DataFrame([
            dict(vessel_name=' SYNTHETIC ', gross_tonnage='1,000', vessel_type='', flag_name=''),
            dict(vessel_name='', gross_tonnage='', vessel_type='', flag_name=''),
            dict(vessel_name='SYNTHETIC', gross_tonnage='bad', vessel_type='', flag_name=''),
        ])
        output, _ = validate_dataframe(frame)
        self.assertEqual(output.iloc[0].gross_tonnage_raw, '1,000')
        self.assertEqual(output.iloc[0].gross_tonnage_parsed, 1000)
        self.assertEqual(output.processable.tolist(), [True, False, False])
        mapping = pd.DataFrame([dict(source='Train', raw_flag='SYNTHETIC',
                                    canonical_iso3='XXX', mapping_status='MAPPED')])
        value, status = map_with_status(pd.Series(['SYNTHETIC', 'NEW', '']),
            mapping=mapping, source='Train', key='raw_flag', value='canonical_iso3')
        self.assertEqual(status.tolist(), ['MAPPED', 'UNSEEN_RAW', 'MISSING'])
        self.assertEqual(value.tolist(), ['XXX', '', ''])

    def test_retrieval(self):
        index = build_retrieval_index(['S002', 'S001', 'S003'],
            ['ALPHA', 'ALPHA', 'BETA'], [1000, 1000, 1500], ['X'] * 3, ['XXX'] * 3)
        positions, branch = retrieve_positions('ALPHA', 1000, 'X', 'XXX', index)
        self.assertEqual(branch, 'EXACT')
        self.assertEqual(index.catalog_ids[positions].tolist(), ['S001', 'S002'])
        positions, branch = retrieve_positions('ALPH', 1000, 'X', 'XXX', index)
        self.assertEqual(branch, 'FUZZY_G3')
        self.assertGreater(len(positions), 0)
        self.assertLessEqual(len(positions), 100)
        self.assertEqual(len(set(positions)), len(positions))

    def test_features_and_gate(self):
        pairs = pd.DataFrame([dict(train_name_normalized='ALPHA',
            catalog_name_normalized='ALPHA', name_missing=0, name_normalized_exact=1)])
        result = add_m6_features(pairs, [])
        self.assertEqual(result.iloc[0].name_similarity, 1)
        self.assertEqual(build_kl_features('가상', 'SYNTH', [dict(
            rule_id='SYNTHETIC', position='PREFIX', hangul_part='가', latin_part='SYN')]), (1, 1))
        self.assertEqual(apply_confidence_gate(1, np.nan), 'AUTO_MATCH')
        self.assertEqual(apply_confidence_gate(2, 0.2), 'AUTO_MATCH')
        self.assertEqual(apply_confidence_gate(2, 0.19999), 'REVIEW_REQUIRED')

    def test_v42_and_review_preserve_scores(self):
        group = pd.DataFrame([
            dict(service_row_id='S', candidate_vessel_id='A', m6_score=0.9,
                 korean_latin_rule_match=1, gt_missing=0, gt_abs_diff=1,
                 flag_comparison_status='MISMATCH'),
            dict(service_row_id='S', candidate_vessel_id='B', m6_score=0.8,
                 korean_latin_rule_match=0, gt_missing=0, gt_abs_diff=0,
                 flag_comparison_status='MATCH')])
        original = group.copy(deep=True)
        output = bp._apply_v42_kl_safety(group)
        self.assertEqual(output.candidate_vessel_id.tolist(), ['B', 'A'])
        self.assertEqual(output.m6_rank.tolist(), [1, 2])
        self.assertEqual(dict(zip(output.candidate_vessel_id, output.m6_score)), {'A': 0.9, 'B': 0.8})
        pd.testing.assert_frame_equal(group, original)
        self.assertEqual(apply_cross_script_review('AUTO_MATCH', '', '가상', 'SYNTHETIC')[0],
                         'REVIEW_REQUIRED')
        candidates = pd.DataFrame([
            dict(m6_rank=1, candidate_vessel_id='A', name_similarity=0.8,
                 train_flag_status='MAPPED', catalog_flag_status='MAPPED',
                 train_flag_value='XXX', catalog_flag_value='YYY'),
            dict(m6_rank=2, candidate_vessel_id='B', name_similarity=0.79,
                 train_flag_status='MAPPED', catalog_flag_status='MAPPED',
                 train_flag_value='XXX', catalog_flag_value='XXX')])
        before = candidates.copy(deep=True)
        self.assertEqual(apply_name_flag_review('AUTO_MATCH', '', candidates)[0], 'REVIEW_REQUIRED')
        pd.testing.assert_frame_equal(candidates, before)
        candidates['catalog_gt'] = [1000, 2000]
        candidates['catalog_flag_status'] = 'MAPPED'
        candidates['catalog_flag_value'] = ['XXX', 'YYY']
        before = candidates.copy(deep=True)
        category, _, _ = bp._classify_review_information_gap('REVIEW_REQUIRED',
            pd.Series(dict(gt_missing=1, flag_missing=1)), candidates)
        self.assertEqual(category, 'INFORMATION_INSUFFICIENT')
        self.assertEqual(bp._classify_review_information_gap('AUTO_MATCH',
            pd.Series(dict(gt_missing=1)), candidates), ('', '', ''))
        pd.testing.assert_frame_equal(candidates, before)

    def test_synthetic_batch_connection(self):
        catalog = pd.DataFrame([dict(vessel_id='SYNTH001', vessel_name='SYNTHETIC ALPHA',
            gross_tonnage='1000', vessel_type='SYNTHETIC', flag_name='SYNTHETIC',
            name_normalized='SYNTHETICALPHA', gross_tonnage_numeric=1000,
            type_broad_family='', type_mapping_status='UNSEEN_RAW', flag_iso3='',
            flag_mapping_status='UNSEEN_RAW')])
        frame = catalog[['vessel_name', 'gross_tonnage', 'vessel_type', 'flag_name']].copy()
        from core.normalization import normalize_dataframe
        def normalize_synthetic(frame, *, source):
            return normalize_dataframe(frame, source=source,
                type_map=pd.DataFrame(columns=['source', 'raw_vessel_type', 'broad_family', 'mapping_status']),
                flag_map=pd.DataFrame(columns=['source', 'raw_flag', 'canonical_iso3', 'mapping_status']))
        with patch.object(bp, '_load_frozen_rules', return_value=[]), \
             patch.object(bp, '_load_frozen_m6_artifact', return_value=FixedScores()), \
             patch.object(bp, '_normalize_catalog', return_value=catalog), \
             patch.object(bp, 'normalize_dataframe', side_effect=normalize_synthetic):
            output = bp.BatchMatcher().process_dataframe(frame)
        self.assertEqual(output.results.iloc[0].match_status, 'AUTO_MATCH')
        self.assertEqual(output.results.iloc[0].rank1_candidate_id, 'SYNTH001')
        self.assertEqual(output.results.iloc[0].rank1_score, 0.9)
        self.assertEqual(output.candidates.iloc[0].m6_rank, 1)

    def test_source_manifest(self):
        import hashlib
        for rel, digest in json.loads((ROOT / 'docs/source-integrity.json').read_text())['copied_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)


if __name__ == '__main__':
    unittest.main()
