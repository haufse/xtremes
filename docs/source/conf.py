# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'xtremes'
copyright = '2025, Erik Haufs'
author = 'Erik Haufs'

release = '0.3'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_size',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'sphinx.ext.mathjax',
]

nbsphinx_execute = 'always'

epub_exclude_files = ['notebooks/*.ipynb']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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


# -- Options for LaTeX/PDF output ---------------------------------------------
#latex_engine = 'lualatex' ## somehow destroys title page
latex_engine = 'xelatex'

# Ensure that sections in the toctree appear as separate chapters
latex_documents = [
    ('index', 'xtremes.tex', 'xtremes Documentation',
     'Erik Haufs', 'manual'),
]


# Set the top-level sectioning to 'chapter' to ensure chapters appear correctly
latex_toplevel_sectioning = 'chapter'

# Additional LaTeX settings to improve PDF output

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'classoptions': ',openany',
    'preamble': r'''
    \usepackage{enumitem}
    \setlistdepth{99}
	% Force the use of Computer Modern
    %\usepackage{mathptmx} % This package changes fonts, so we need to reset them back
    \usepackage{fontspec}
    %\setmainfont{Times New Roman} %%if not working, try to use the following line
    \setmainfont{TeX Gyre Termes}
    %\renewcommand{\rmdefault}{cmr} % Roman font to Computer Modern
    %\renewcommand{\sfdefault}{cmss} % Sans-serif font to Computer Modern Sans
    %\renewcommand{\ttdefault}{cmtt} % Typewriter font to Computer Modern Typewriter
    \usepackage{amsmath,amsfonts,amssymb} % For math symbols in Computer Modern
    \usepackage{hyperref} % For hyperlinks in the PDF
    \hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
    ''',

    # Remove the default table of contents behavior
    'tableofcontents': '',
    'sphinxsetup': r'''
    verbatimwithframe=false,
    verbatimwrapslines=true
    ''',

    # Use the "chapter" directive to start each document in the toctree as a new chapter
    
    'maketitle': r'''
    \begin{titlepage}
    \centering
    {\Huge \bfseries \texttt{xtremes} Documentation \par}
    \vspace{1cm}
    {\large Version 0.3.0 \par}
    \vspace{1.5cm}
    {\large Author: Erik Haufs \\ \normalsize{\href{mailto:erik.haufs@rub.de}{erik.haufs@rub.de}} \par}
    \vspace{0.5cm}
    {\large Faculty of Mathematics, Ruhr University Bochum \par}
    \vfill
    \begin{figure}[!htp]
    \centering
    \includegraphics[width=0.4\textwidth]{../../docs/source/images/logo4.png}\par
    \caption{Dall-E's interpretation of the \texttt{xtremes} package}
    \end{figure}
    \vfill
    {\large \today \par}
    \end{titlepage}
    \frontmatter
    \setcounter{tocdepth}{1}
    \tableofcontents
    \mainmatter
    ''',

}

# Configure BibTeX to ensure citations are handled correctly
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'
