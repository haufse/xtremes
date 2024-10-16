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

# -- Options for LaTeX/PDF output

# Ensure that sections in the toctree appear as separate chapters
latex_documents = [
    ('index', 'xtremes.tex', 'Xtremes Documentation',
     'Erik Haufs', 'manual'),
]

# Set the top-level sectioning to 'chapter' to ensure chapters appear correctly
latex_toplevel_sectioning = 'chapter'

# Additional LaTeX settings to improve PDF output

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
    \usepackage{enumitem}
    \setlistdepth{99}
    ''',
    # Remove the default table of contents behavior
    'tableofcontents': '',
    'sphinxsetup': r'''
    verbatimwithframe=false,
    verbatimwrapslines=true
    ''',
    # Use the "chapter" directive to start each document in the toctree as a new chapter
    'maketitle': r'''
    \sphinxmaketitle
    \frontmatter
    \tableofcontents
    \mainmatter
    ''',
}

# Configure BibTeX to ensure citations are handled correctly
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'
