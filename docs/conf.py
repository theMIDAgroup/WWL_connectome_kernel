"""Sphinx configuration for wwl-connectome-kernel."""

project = "wwl-connectome-kernel"
author = "Chiara Razzetta"
copyright = "CC BY 4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
# atlas.get_schaefer_coords downloads data lazily, so nothing heavy runs at import.
# nilearn is only needed at call time; mock it so the docs build without it.
autodoc_mock_imports = ["nilearn"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = []
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build"]
