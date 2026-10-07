"""Small negative fixtures for recorded-row provenance (no new simulations)."""
import importlib.util
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_calibration_audit.py'
spec = importlib.util.spec_from_file_location('calibration_analysis', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    cfg = {'B': 100, 'decision': 'pvalue'}
    meta = {'arguments': {'seed': 6060, 'cohort': 'local-arm64'},
            'configurations': {'conservative': cfg}}
    seed = int(np.random.SeedSequence(6060, spawn_key=(0,)).generate_state(1)[0])
    row = {'case': 'gaussian_balanced', 'N': '500', 'M': '500', 'rep': '0',
           'data_seed': str(seed), 'method_seed': str(seed + 17), 'cohort': 'local-arm64',
           'method': 'conservative', 'orientation': 'original', 'alternative': '0',
           'config_hash': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
           'effective_L': '20'}
    row.update({k: 'a'*64 for k in ('data_hash', 'score_input_hash', 'views_hash', 'reference_positions_hash')})
    return row, meta


def test_valid_provenance_fixture():
    row, meta = fixture()
    module.validate_row(row, meta)


@pytest.mark.parametrize('field,value', [('N', '800'), ('data_seed', '1'),
    ('method_seed', '1'), ('cohort', 'tu-cpu'), ('config_hash', '0'*64),
    ('score_input_hash', 'b'*64), ('effective_L', '19')])
def test_corrupted_provenance_rejected(field, value):
    row, meta = fixture()
    row[field] = value
    with pytest.raises(AssertionError):
        module.validate_row(row, meta)
