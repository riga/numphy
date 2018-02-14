# -*- coding: utf-8 -*-


import sys
import os


sys.path.insert(0, os.path.abspath(".."))
import numphy


project = "numphy"
author = numphy.__author__
copyright = numphy.__copyright__
version = numphy.__version__
release = numphy.__version__
language = "en"

templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"
add_module_names = True

exclude_patterns = []
pygments_style = "sphinx"
html_logo = "../logo.png"
html_theme = "alabaster"
html_sidebars = {"**": [
    "about.html",
    "localtoc.html",
    "searchbox.html"]
}
html_theme_options = {
    "github_user": "riga",
    "github_repo": "numphy",
    "travis_button": True,
    "fixed_sidebar": True
}

extensions = [
    "sphinx.ext.autodoc"
]
