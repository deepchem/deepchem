# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_rtd_theme  # noqa
import deepchem  # noqa

# -- Project information -----------------------------------------------------

project = 'deepchem'
copyright = '2020, deepchem-contributors'
author = 'deepchem-contributors'

# The full version, including alpha/beta/rc tags
version = deepchem.__version__
release = deepchem.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]

# Options for autodoc directives
autodoc_default_options = {
    'member-order':
    'bysource',
    'special-members':
    True,
    'exclude-members':
    '__repr__, __str__, __weakref__, __hash__, __eq__, __call__, __dict__',
}

# How to represents typehints
autodoc_typehints = "signature"

mathjax_path = 'http://mathjax.connectmv.com/MathJax.js?config=default'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# autosectionlabel setting
autosectionlabel_prefix_document = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
    ],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo.png'

# Customize the sphinx theme
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}

# -- Source code links ---------------------------------------------------


# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):

  def find_source():
    # try to find the file and line number, based on code from numpy:
    # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
    obj = sys.modules[info['module']]
    for part in info['fullname'].split('.'):
      obj = getattr(obj, part)
    fn = inspect.getsourcefile(obj)
    fn = os.path.relpath(fn, start=os.path.dirname(deepchem.__file__))
    source, lineno = inspect.getsourcelines(obj)
    return fn, lineno, lineno + len(source) - 1

  if domain != 'py' or not info['module']:
    return None
  try:
    filename = 'deepchem/%s#L%d-L%d' % find_source()
  except Exception:
    filename = info['module'].replace('.', '/') + '.py'

  tag = 'master' if 'dev' in release else release
  return "https://github.com/deepchem/deepchem/blob/%s/%s" % (tag, filename)
