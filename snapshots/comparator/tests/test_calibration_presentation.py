"""Synthetic renderer fixtures: collapse duplicates, never hide disagreements."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/export_calibration_manuscript.py'
spec = importlib.util.spec_from_file_location('calibration_export', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    result = []
    for case in ('gaussian_balanced', 'rounded_balanced', 'gaussian_200_800', 'gaussian_800_200'):
        for method in ('conservative', 'midp_only', 'wilson_only', 'antithetic_only', 'legacy_combined'):
            result.append(dict(case=case, alternative='0', method=method, orientation='original',
                R='8000', rejections='400', rate='0.05', lo95='0.04', hi95='0.06', se='0.002',
                mean_B_used='100', p_over_bonf50='1'))
    for alternative in ('0', '1'):
        for orientation in ('original', 'rotated'):
            for method in ('conservative', 'canonical_pca', 'canonical_whiten'):
                if alternative == '0' and orientation == 'original' and method == 'conservative':
                    continue
                result.append(dict(case='gaussian_balanced', alternative=alternative, method=method,
                    orientation=orientation, R='500' if alternative == '1' else '8000',
                    rejections='250' if alternative == '1' else '400',
                    rate='0.5' if alternative == '1' else '0.05', lo95='0.04', hi95='0.6',
                    se='0.002', mean_B_used='100', p_over_bonf50='1'))
    return result


def test_renderer_collapses_only_duplicate_display_not_source_rows():
    source = rows()
    assert len(source) == 31
    text = module.render(source)
    assert len(source) == 31
    assert 'Pooled PCA$^{\\ast}$ &' in text and 'Pooled whitening &' not in text
    assert 'Legacy-style combined &' in text and 'Historical combined &' not in text
    assert 'not a byte-for-byte rerun' in text
    assert 'all 17000 paired scores' in text


@pytest.mark.parametrize('key', ['rate', 'rejections', 'lo95', 'mean_B_used'])
def test_unequal_whitening_results_cannot_be_collapsed(key):
    source = rows()
    row = next(r for r in source if r['method'] == 'canonical_whiten' and r['alternative'] == '1')
    row[key] = '123'
    with pytest.raises(AssertionError):
        module.render(source)
