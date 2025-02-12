# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import inspect
from operator import attrgetter

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

version = __version__.split("+")[0]
release = __version__  # The full version, including alpha/beta/rc tags

html_title = f"folie {version}"  # Use short version number


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "nbsphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    # "sphinx_gallery.load_style",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]


numpydoc_show_class_members = False
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]

templates_path = ["_templates"]

plot_gallery = "True"
master_doc = "index"
# The suffix of source filenames.
source_suffix = ".rst"

exclude_patterns = ["_build", "_templates", "auto_examples/*.ipynb", "auto_examples/*/*.ipynb", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "*.ipynb"]


# html_context = {
#     "current_version": version,
#     "latest_version": version,
#     "branches": [{"name": "fem", "url": "fem"}, {"name": version, "url": "fem"}],
# }

# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True

autodoc_default_options = {"inherited-members": True, "members": True, "member-order": "groupwise", "special-members": "__call__", "exclude-members": "__init__"}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        # "github_button.html",
        # "versioning.html",
    ]
}

html_theme_options = {
    "source_repository": "https://github.com/langevinmodel/folie",
    "source_branch": "main",
    "source_directory": "docs/",
}


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
    "backreferences_dir": "api/generated",
    "reference_url": {"folie": None},
    "examples_dirs": ["../examples"],  # path to your examples scripts
    "ignore_pattern": "profile_",
    "gallery_dirs": ["auto_examples"],  # path where to save gallery generated examples
}

nbsphinx_execute = "never"


def setup(app):
    # a copy button to copy snippet of code from the documentation
    app.add_js_file("js/copybutton.js")


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# Function for linkcode


def linkcode_resolve(domain, info):
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='https://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'https://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    package = "folie"
    url_fmt = "https://github.com/langevinmodel/" "folie/blob/{revision}/" "{package}/{path}#L{lineno}"
    revision = "main"

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)
