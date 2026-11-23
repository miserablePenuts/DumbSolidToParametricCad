"""Feature extraction: STEP solid → structured JSON of the features it contains.

Rule-based recognition on the exact B-Rep (holes, hole patterns, fillets, chamfers, extrusions /
cuts with classified profiles, symmetry), cross-checked face by face against the learned
segmenter (``runs/seg``). See ``features/README.md``.

    .venv\\Scripts\\python.exe -m features.extract part.stp --out runs\\features\\part
"""
