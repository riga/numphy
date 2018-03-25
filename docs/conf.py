# -*- coding: utf-8 -*-


import sys
import os


thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, "_extensions"))
sys.path.insert(0, os.path.dirname(thisdir))

import numphy as nph


project = nph.__name__
author = nph.__author__
copyright = nph.__copyright__
version = nph.__version__[:nph.__version__.index(".", 2)]
release = nph.__version__

templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = []
pygments_style = "sphinx"
add_module_names = False

html_title = project + " Documentation"
html_logo = "../logo.png"
html_sidebars = {"**": [
    "about.html",
    "localtoc.html",
    "searchbox.html",
]}
html_theme = "sphinx_rtd_theme"
html_theme_options = {}
if html_theme == "sphinx_rtd_theme":
    html_theme_options.update({
        "logo_only": True,
        "prev_next_buttons_location": None,
        "collapse_navigation": False,
    })
elif html_theme == "alabaster":
    html_theme_options.update({
        "github_user": "riga",
        "github_repo": "numphy",
        "travis_button": True,
    })

extensions = ["sphinx.ext.autodoc", "pydomain_patch"]

autodoc_member_order = "bysource"


def setup(app):
    app.add_stylesheet("styles_common.css")
    if html_theme in ("sphinx_rtd_theme", "alabaster"):
        app.add_stylesheet("styles_{}.css".format(html_theme))
