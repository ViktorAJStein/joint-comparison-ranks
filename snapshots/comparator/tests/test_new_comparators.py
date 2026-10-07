from itertools import combinations, permutations
import numpy as np
from scipy.stats import fisher_exact, binomtest
from sklearn.dummy import DummyClassifier
from baselines.conditional_c2st import prediction_hypergeometric_pvalue, C2STHypergeometric
from baselines.c2st import c2st_holdout_binomial_test
from baselines.analytic_scf import gaussian_scf_features, paired_feature_statistic, GaussianWindowSCF


def test_confusion_tail_equals_fisher_and_is_superuniform():
    for n,n1,k in [(8,3,5),(8,4,4),(7,2,0),(7,2,7)]:
        prediction=np.array([1]*k+[0]*(n-k));values=[]
        for positives in combinations(range(n),n1):
            truth=np.zeros(n,dtype=int);truth[list(positives)]=1
            p=prediction_hypergeometric_pvalue(truth,prediction)
            table=np.array([[np.sum((truth==1)&(prediction==1)),np.sum((truth==1)&(prediction==0))],
                            [np.sum((truth==0)&(prediction==1)),np.sum((truth==0)&(prediction==0))]])
            assert abs(p-fisher_exact(table,alternative='greater').pvalue)<1e-12
            values.append(p)
        for alpha in (.01,.05,.1,.25,.5,1.):
            assert np.mean(np.array(values)<=alpha)<=alpha+1e-12


def test_imbalanced_majority_classifier_is_not_evidence_of_shift():
    x,y=np.zeros((40,2)),np.zeros((80,2))
    factory=lambda:DummyClassifier(strategy='most_frequent')
    old=c2st_holdout_binomial_test(x,y,estimator_factory=factory,split_prop=.5,seed=601)
    new=C2STHypergeometric(estimator_factory=factory,split_prop=.5,seed=601).run(x,y)
    assert old.reject and not new.reject and new.p_perm==1.


def test_scf_is_actual_gaussian_window_and_published_paired_covariance():
    rng=np.random.default_rng(195);x,y=rng.normal(size=(40,3)),rng.normal(size=(40,3))
    frequencies=rng.normal(size=(3,3));gamma=2.3
    fx,fy=gaussian_scf_features(x,frequencies,gamma),gaussian_scf_features(y,frequencies,gamma)
    complex_features=np.exp(-np.sum((x/gamma)**2,axis=1)/2)[:,None]*np.exp(1j*(x/gamma)@frequencies.T)
    assert np.allclose(fx[:,0::2],complex_features.imag)
    assert np.allclose(fx[:,1::2],complex_features.real)
    difference=fx-fy;mean=difference.mean(axis=0)
    covariance=np.cov(difference,rowvar=False,bias=True)
    expected=len(x)*float(mean@np.linalg.solve(covariance,mean))
    assert abs(paired_feature_statistic(fx,fy)-expected)<1e-10


def test_scf_training_frozen_and_independent_full_budget_repeatable():
    rng=np.random.default_rng(72)
    x,y=rng.normal(size=(32,3)),rng.normal(size=(32,3))
    tx,ty=rng.normal(size=(40,3)),rng.normal(loc=.4,size=(40,3))
    test=GaussianWindowSCF(J=3,B=19,seed=731).fit_gamma(tx,ty,[.5,1.,2.])
    gamma=test.gamma;frequencies=test.frequencies_.copy();scores=list(test.training_scores_)
    a,b=test.run(x,y),test.run(x,y)
    assert a.stat_obs==b.stat_obs and a.p_perm==b.p_perm and a.perms_used==19
    assert test.gamma==gamma and np.array_equal(test.frequencies_,frequencies) and scores==test.training_scores_


def test_gaussian_window_expectation_by_deterministic_quadrature():
    # Integrate N(mu,sigma^2) against physicists' Gauss-Hermite nodes.
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    for mu, sigma, gamma in [(0., 1., 1.), (.7, 1.5, 2.), (-2., .3, .8)]:
        frequencies = np.array([[.2], [1.1], [-.8]])
        x = (mu + np.sqrt(2)*sigma*nodes)[:, None]
        features = gaussian_scf_features(x, frequencies, gamma)
        numerical = weights @ (features[:, 1::2] + 1j*features[:, 0::2]) / np.sqrt(np.pi)
        a, b = sigma**2/gamma**2, mu/gamma
        t = frequencies[:, 0]
        exact = (1+a)**(-.5)*np.exp(-b*b/(2*(1+a))-a*t*t/(2*(1+a))+1j*t*b/(1+a))
        assert np.max(np.abs(numerical-exact)) < 1e-10


def test_paired_scf_full_ordered_space_exact_superuniformity():
    features = gaussian_scf_features(np.array([[-1.], [-.2], [.3], [1.4]]),
                                     np.array([[.7]]), 1.2)
    scores = np.array([paired_feature_statistic(features[list(order[:2])],
                      features[list(order[2:])], ridge=1e-3)
                      for order in permutations(range(4))])
    p = np.mean(scores[None, :] >= scores[:, None], axis=1)
    for alpha in (.05, .1, .25, .5, 1.):
        assert np.mean(p <= alpha) <= alpha + 1e-12


def test_runner_does_not_publish_survivor_rejection_rate():
    from scripts.run_power_bench import complete_rejection_summary
    assert complete_rejection_summary(3, 10, 10)[0] == .3
    assert np.isnan(complete_rejection_summary(3, 9, 10)).all()
    assert np.isnan(complete_rejection_summary(0, 0, 10)).all()


def test_runner_cli_forwards_new_comparator_settings(tmp_path, monkeypatch):
    import csv
    import json
    from scripts import run_power_bench as runner
    output = tmp_path / 'smoke.csv'
    monkeypatch.setattr('sys.argv', ['run_power_bench', '--experiment', 'calibration',
        '--families', 'locshift', '--methods', 'C2ST_Hypergeometric', 'SCF_GaussianWindow', 'MMD_RFF',
        '--Ns', '16', '--Ms', '16', '--d', '2', '--R', '1', '--B', '3', '--jobs', '1',
        '--scf_J', '2', '--scf_gamma', '2.5', '--c2st_split_prop', '.5',
        '--rff_features', '8', '--rff_bandwidth', '1.5',
        '--rank_calibration', 'conservative', '--out_csv', str(output)])
    runner.main()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 3 and all(r['status'] == 'OK' and r['R_completed'] == '1' for r in rows)
    cfg = {r['method']: json.loads(r['method_config']) for r in rows}
    assert cfg['SCF_GaussianWindow']['J'] == 2 and cfg['SCF_GaussianWindow']['gamma'] == 2.5
    assert cfg['C2ST_Hypergeometric']['split_prop'] == .5
    assert cfg['MMD_RFF']['features'] == 8 and cfg['MMD_RFF']['bandwidth'] == 1.5
