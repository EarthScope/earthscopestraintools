# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Earthscope Strain Tools'
copyright = '2023, Mike Gottlieb, Catherine Hanagan'
author = 'Mike Gottlieb, Catherine Hanagan'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
#html_logo = "images/cig_short_pylith.png"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
#import pathlib
#import sys
#sys.path.insert(0, pathlib.Path('src/earthscopestraintools/timeseries.py').parents[2].resolve().as_posix())
