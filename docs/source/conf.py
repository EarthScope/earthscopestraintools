# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Earthscope Strain Tools'
copyright = '2023, Earthscope Consortium'
author = 'Mike Gottlieb, Catherine Hanagan'
release = '1'

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# note that the absolute paths are referenced relative to the configuration directory (with conf.py)
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/earthscopestraintools'))

# To keep same env as the main project, will allow notebooks in the docs
# by downloading appropriate packages
from sphinx_pyproject import SphinxConfig
config = SphinxConfig("../../pyproject.toml", globalns=globals())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinxcontrib.apidoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'myst_parser'] # myst will allow document building in markdown syntax
myst_enable_extensions = [ "amsmath",
    "dollarmath",
    "colon_fence"]

nbsphinx_kernel_name = 'python'
nbsphinx_allow_errors = True

# APIDOC configuration 
apidoc_module_dir = "../../src/earthscopestraintools"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_toc_file = False
apidoc_module_first = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "./_static/EarthScope_Logo-color.png"


