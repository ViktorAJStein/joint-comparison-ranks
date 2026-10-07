# Maintainer publication and final-paper checklist

The user authorizes publishing this experimental reproducibility snapshot under the active GitHub account `ViktorAJStein`, in a separate `joint-comparison-ranks` repository. The snapshot tag is `reproducibility-20261007`, not `arxiv-v1`; final-paper/citation/licensing and policy evidence below remains pending unless actually supplied. Public experimental code does not certify an anonymous submission or provide a final paper upload.

Do not push the project root, the September-only `main` branch, the ICML repository, or the research git history.

## Upload these files

Commit exactly the source/docs/tests/configs/small tables/plots listed in `UPLOAD_MANIFEST.tsv`, plus that manifest and `manifest.sha256`. No globbed `git add .`. The payload contains two frozen scientific snapshots, all three current overlays, five complete numerical bundles, raw/plot verification and a portable runner. Retain notices.

Upload all five `*-raw.tar.gz` assets described by `release-assets.json` to the **same versioned GitHub release**; do not put them in git. These complete per-repetition assets are necessary for no-compute result auditing. Supply the final author-approved paper PDF/link separately after submission/arXiv; do not include private portal evidence or an unfrozen draft.

## Never upload

`.git` from research checkouts, `.venv`, caches, local paths/SSH config/credentials, cluster launch or authorization records, reviewer reports, rebuttals, private notes, session exports, unrelated projects, downloaded third-party papers, datasets, old withdrawn CSVs/figures, or logs. `private-build-provenance.json` is for local auditing only. Do not change `josemanuel22/ranktwosample`.

## Remaining human decisions / final binding

- [ ] Confirm final submitted-paper contents match this empirical snapshot (source bound to8bb5c6c); update the release map if authors changed experiments.
- Destination for this user-requested snapshot: `ViktorAJStein/joint-comparison-ranks` (active authenticated personal account).
- [ ] Approve public-release timing after submission and check anonymous-review/preprint/publicity policy; this package is not anonymized.
- [ ] Approve licensing of new overlays/scripts/data, preserving upstream notice.
- [ ] Bind exact title, full final author list, arXiv ID/version and verified `CITATION.cff`.
- [ ] Inspect files/links/PDF metadata for secrets and unintended material; automated scans are not human legal/security clearance.
- [ ] Decide whether supplying the original Gaussian-copula illustration PDF without its unavailable plotting source is sufficient; do not claim all analytic figures can be regenerated from source.
- [ ] Verify clean install, tests, manifests, all raw assets and plotting commands; archive verification receipts.
- [ ] Check release URLs signed out and download/verify all assets before any availability claim.
- [ ] Create immutable `arxiv-v1` tag only after the final binding.

## Commands for a future final-paper release

The experimental snapshot uses the prepared clean repository and the explicit upload allowlist. For a future final-paper release, resolve the remaining decisions above first. Stage paths explicitly using the generated TSV, excluding output/cache paths. Review `git diff --cached --stat` before committing:

```bash
git init -b main
uv run python -c 'import csv,subprocess; paths=[r["path"] for r in csv.DictReader(open("UPLOAD_MANIFEST.tsv"),delimiter="\t")]+["UPLOAD_MANIFEST.tsv","manifest.sha256"]; subprocess.run(["git","add","--",*paths],check=True)'
git diff --cached --stat
git commit -m "release: reproducible joint comparison rank experiments"
# Replace OWNER only after confirmation. This next command creates a PUBLIC repo.
gh repo create OWNER/joint-comparison-ranks --public --source=. --remote=origin --push
# After binding the paper and checking the payload:
git tag arxiv-v1
git push origin arxiv-v1
gh release create arxiv-v1 /path/to/release-assets/*-raw.tar.gz --title "arXiv v1 reproduction artifacts" --notes "See README and REPRODUCING.md."
```

Replace null `published_url` values in `release-assets.json` with verified release URLs and refresh payload manifests before the final commit/tag. The release-asset checksums must stay unchanged. Enable CI/branch protection and verify signed-out access. Do not add an unsupported ready-to-submit statement or retrospectively change submission checklist answers.
