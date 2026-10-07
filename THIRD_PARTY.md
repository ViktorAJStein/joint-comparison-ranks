# Source attribution and redistribution scope

The inherited implementation is MIT-licensed, copyright José Manuel de Frutos. Preserve `LICENSE` and the frozen source notices. Provenance base: https://github.com/josemanuel22/ranktwosample/tree/42bbe058fb8b25188b30b61f52b61765506cbb0c . The supplied anonymous archive was byte-identical to that source, SHA256 `9639c7e20b1a9f6f2a94d40fede340e824924a16b0cba850d05916051865af85`.

Dependencies are downloaded by uv, not vendored: NumPy, SciPy, pandas, scikit-learn, tqdm, threadpoolctl; optional pytest and Matplotlib. Their own distributions retain their license notices. Dependency names are not a legal clearance for new code/data; maintainers must approve the release license for overlays, scripts and generated numerical assets before public upload.

The current paper experiments use synthetic data generated from the included routines. No CIFAR images, CIFAR-10-C files, third-party PDFs, pretrained models, or external source archives are redistributed. The old image entry point remains in the attributed frozen implementation but is outside the reproducibility scope and is not invoked by the paper driver.

New paper asset provenance is described by `REPRODUCING.md`, the per-study `environment.json`, source hashes and release manifests. The public raw asset excludes operational author/job/account evidence while retaining full per-repetition scientific values. A public (named-author) release is **not** an anonymous review artifact and should not be linked into an anonymous submission without separately checking conference policy.
