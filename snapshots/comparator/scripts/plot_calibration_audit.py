"""Plot the audited cohort; no simulations or selective row filtering."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis / 'summary.csv'
    audit = json.loads((args.analysis / 'audit.json').read_text())
    assert audit['status'] == 'verified'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit['summary_sha256']
    rows = list(csv.DictReader(source.open()))
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ['gaussian_balanced', 'rounded_balanced', 'gaussian_200_800', 'gaussian_800_200']
    labels = ['Gaussian (500,500)', 'Rounded (500,500)', 'Gaussian (200,800)', 'Gaussian (800,200)']
    methods = ['conservative', 'midp_only', 'wilson_only', 'antithetic_only', 'legacy_combined']
    names = ['Independent plus-one', 'Mid-p only', 'Wilson only', 'Antithetic only', 'Legacy-style combined']
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True, constrained_layout=True)
    for ax, case, label in zip(axes, cases, labels):
        for j, method in enumerate(methods):
            row, = [r for r in rows if r['case'] == case and r['alternative'] == '0'
                    and r['method'] == method and r['orientation'] == 'original']
            rate, lo, hi = map(float, (row['rate'], row['lo95'], row['hi95']))
            color = 'tab:red' if float(row['p_over_bonf50']) < .05 else 'tab:blue'
            ax.errorbar(j, rate, yerr=[[rate-lo], [hi-rate]], fmt='o', capsize=3, color=color)
        ax.axhline(.05, color='black', ls='--', lw=1)
        ax.set_xticks(range(5), names, rotation=65, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(.035, .075)
    axes[0].set_ylabel('Null rejection frequency (95% CI)')
    fig.suptitle('Local ARM64 cohort, R=8000 per row; red: Bonferroni-50 over-rejection')
    for ext in ('pdf', 'png'):
        fig.savefig(args.output / f'calibration_audit.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    assert audit['whitening_pairs'] == audit['whitening_equal_pairs'] == 17000
    for orientation in ('original', 'rotated'):
        pca, = [r for r in rows if r['alternative'] == '1' and r['method'] == 'canonical_pca'
                and r['orientation'] == orientation]
        whiten, = [r for r in rows if r['alternative'] == '1' and r['method'] == 'canonical_whiten'
                   and r['orientation'] == orientation]
        assert all(pca[k] == whiten[k] for k in ('R', 'rejections', 'rate', 'lo95', 'hi95'))
    methods = ['conservative', 'canonical_pca']
    names = ['Raw coordinates', 'Pooled PCA\n(whitening identical)']
    for offset, orientation, color in [(-.18, 'original', 'tab:blue'), (.18, 'rotated', 'tab:orange')]:
        for j, method in enumerate(methods):
            row, = [r for r in rows if r['alternative'] == '1' and r['method'] == method
                    and r['orientation'] == orientation]
            rate, lo, hi = map(float, (row['rate'], row['lo95'], row['hi95']))
            ax.errorbar(j+offset, rate, yerr=[[rate-lo], [hi-rate]], fmt='o', capsize=4,
                        color=color, label=orientation if j == 0 else None)
    ax.set_xticks(range(2), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Power (pointwise exact 95% CI)')
    ax.set_title('Paired sparse-scale diagnostic, d=20, N=M=500, R=500')
    ax.legend()
    for ext in ('pdf', 'png'):
        fig.savefig(args.output / f'coupled_rotation_audit.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    manifest = {'summary_sha256': audit['summary_sha256'], 'source_rows': 31,
                'plotted_null_rows': 20, 'plotted_power_rows': 4,
                'collapsed_power_rows': 'two whitening duplicates retained unchanged in summary.csv',
                'whitening_equal_pairs': audit['whitening_equal_pairs'],
                'omitted_null_rows': 'five balanced rotated/PCA/whitening rows remain in summary.csv',
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.glob('*audit.*'))}}
    (args.output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
