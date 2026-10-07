"""Gaussian-window SCF, published Chwialkowski et al. (2015) Eqs. (13),(14).

This is NOT the archived unsmoothed Fourier alias, nor a reproduction of the
source's discrepant Table-1 frequency-shifted window. Equal sample sizes only.
Training-only gamma selection is a statistic proxy, not their manual bootstrap
median-p-value selection. Fixed-budget ordered permutations are the default.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.stats import chi2


def gaussian_scf_features(Z, frequencies, gamma):
    Z, frequencies = np.asarray(Z, dtype=float), np.asarray(frequencies, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    if Z.ndim != 2 or frequencies.ndim != 2 or Z.shape[1] != frequencies.shape[1]:
        raise ValueError('Compatible data/frequency matrices required')
    scaled = Z / gamma
    phase = scaled @ frequencies.T
    window = np.exp(-np.sum(scaled**2, axis=1) / 2)
    features = np.stack([window[:, None]*np.sin(phase),
                         window[:, None]*np.cos(phase)], axis=-1).reshape(len(Z), -1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features: inspect data and gamma')
    return features


def paired_feature_statistic(FX, FY, ridge=0.):
    if FX.shape != FY.shape or len(FX) < 2:
        raise ValueError('Equal-sized samples with at least two pairs required')
    if ridge < 0 or not np.isfinite(ridge): raise ValueError('Nonnegative finite ridge required')
    difference = FX - FY
    mean = difference.mean(axis=0)
    centered = difference - mean
    covariance = centered.T @ centered / len(difference)
    regularized = covariance + ridge*np.eye(len(mean))
    try:
        value = len(difference)*float(mean @ np.linalg.solve(regularized, mean))
    except np.linalg.LinAlgError:
        value = len(difference)*float(mean @ np.linalg.pinv(regularized, hermitian=True) @ mean)
    if not np.isfinite(value) or value < -1e-10:
        raise FloatingPointError('Unstable paired statistic; inspect covariance/ridge')
    return value


@dataclass
class AnalyticSCFResult:
    stat_obs: float
    p_perm: float
    reject: bool
    perms_used: int
    train_seconds: float
    J: int
    gamma: float
    calibration: str
    ridge: float


class GaussianWindowSCF:
    def __init__(self, J=10, gamma=1., B=1000, alpha=.05, seed=2033,
                 ridge=1e-6, calibration='permutation'):
        if J < 1 or B < 1 or not 0 < alpha < 1:
            raise ValueError('Positive J/B and alpha in (0,1) required')
        if calibration not in ('permutation', 'chi2'):
            raise ValueError('calibration must be permutation|chi2')
        self.J,self.gamma,self.B,self.alpha,self.seed,self.ridge,self.calibration = J,gamma,B,alpha,seed,ridge,calibration
        self.frequencies_=None
        self.training_scores_=None

    def _frequencies(self, d):
        if self.frequencies_ is None:
            self.frequencies_=np.random.default_rng(self.seed).normal(size=(self.J,d))
        if self.frequencies_.shape != (self.J,d): raise ValueError('Feature dimension changed')
        return self.frequencies_

    def fit_gamma(self, X_training, Y_training, candidates):
        """Explicit independent training data only; caller protects the split."""
        x,y=np.asarray(X_training,dtype=float),np.asarray(Y_training,dtype=float)
        frequencies=self._frequencies(x.shape[1])
        candidates=list(candidates)
        if not candidates: raise ValueError('At least one predeclared candidate required')
        scores=[]
        for gamma in candidates:
            score=paired_feature_statistic(gaussian_scf_features(x,frequencies,gamma),
                                          gaussian_scf_features(y,frequencies,gamma),self.ridge)
            scores.append({'gamma':float(gamma),'statistic':score})
        self.gamma=float(candidates[int(np.argmax([s['statistic'] for s in scores]))])
        self.training_scores_=scores
        return self

    def run(self,X,Y):
        started=time.perf_counter()
        x,y=np.asarray(X,dtype=float),np.asarray(Y,dtype=float)
        if x.ndim!=2 or x.shape!=y.shape or len(x)<2:
            raise ValueError('Equal-sized compatible feature matrices required; do not truncate imbalance')
        frequencies=self._frequencies(x.shape[1])
        fx,fy=gaussian_scf_features(x,frequencies,self.gamma),gaussian_scf_features(y,frequencies,self.gamma)
        observed=paired_feature_statistic(fx,fy,self.ridge)
        if self.calibration=='chi2':
            p=float(chi2.sf(observed,2*self.J));used=0
        else:
            # Permute full ordered role assignments: within-role pairing affects
            # the covariance and must not be omitted from the allocation space.
            features=np.vstack([fx,fy]);n=len(x)
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,0x5045524D]))
            greater_equal=0
            for _ in range(self.B):
                order=rng.permutation(2*n)
                candidate=paired_feature_statistic(features[order[:n]],features[order[n:]],self.ridge)
                greater_equal+=candidate>=observed
            used=self.B;p=(1+greater_equal)/(self.B+1)
        return AnalyticSCFResult(observed,p,p<=self.alpha,used,time.perf_counter()-started,
                                 self.J,self.gamma,self.calibration,self.ridge)
