# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'xtremes'
copyright = '2024, Erik Haufs'
author = 'Erik Haufs'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_size',
    'sphinxcontrib.bibtex'
]
    
bibtex_bibfiles = ['references.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
sphinx_rtd_size_width = "90%"

# -- Options for EPUB output
epub_show_urls = 'footnote'
