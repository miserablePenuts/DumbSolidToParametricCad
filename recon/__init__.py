"""Geometry-first reconstruction of a design history from a STEP solid.

Feature candidates (pad / pocket / hole / fillet / chamfer / mirror) are derived from the exact
B-Rep of the input; a search picks the candidates that best explain the input and every accepted
feature is verified by rebuilding with the CAD kernel. No learning yet — this is the baseline that
a learned candidate scorer will later plug into.

    .venv\\Scripts\\python.exe -m recon.reconstruct part.stp --out out/part
"""
