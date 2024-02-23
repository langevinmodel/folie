# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

project = "folie"
copyright = "2023, Hadrien Vroylandt"
author = "Hadrien Vroylandt"

try:
    from folie import __version__
except ImportError as exc:
    print(exc)
    raise

release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

bibtex_bibfiles = ["references.bib"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "nbsphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_gallery.load_style",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
]

numpydoc_show_class_members = False
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]

templates_path = ["_templates"]

plot_gallery = "True"
master_doc = "index"
# The suffix of source filenames.
source_suffix = ".rst"

exclude_patterns = ["_build", "_templates", "auto_examples/*.ipynb", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "*.ipynb"]


# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True

autodoc_default_options = {"inherited-members": True, "members": True, "member-order": "groupwise", "special-members": "__call__", "exclude-members": "__init__"}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": (" https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "doc_module": "folie",
    "backreferences_dir": os.path.join("api/generated"),
    "reference_url": {"folie": None},
    # path to your examples scripts
    "examples_dirs": "../examples",
    "ignore_pattern": "profile_",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

nbsphinx_execute = "never"


def setup(app):
    # a copy button to copy snippet of code from the documentation
    app.add_js_file("js/copybutton.js")


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
