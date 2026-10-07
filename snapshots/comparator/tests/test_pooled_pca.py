from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scripts.rotation_bench import pooled_pca_transform

@pytest.mark.parametrize('whiten',[False,True])
def test_canonical_pca_is_row_equivariant_even_with_repeated_spectrum(whiten):
    z=np.vstack([np.eye(3),-np.eye(3),np.zeros((2,3))])
    a,b=pooled_pca_transform(z[:4],z[4:],whiten=whiten,canonical_order=True)
    original=np.vstack([a,b])
    for seed in range(20):
        p=np.random.default_rng(seed).permutation(len(z))
        c,d=pooled_pca_transform(z[p[:4]],z[p[4:]],whiten=whiten,canonical_order=True)
        assert np.array_equal(np.vstack([c,d]),original[p])
