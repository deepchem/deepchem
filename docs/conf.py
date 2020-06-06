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
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'deepchem'
copyright = '2020, deepchem-contributors'
author = 'deepchem-contributors'

# The full version, including alpha/beta/rc tags
release = '2.4.0rc'

# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx', 'sphinx.ext.mathjax', 'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon'
]

autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo.png'
# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect
from os.path import relpath, dirname

for name in ['sphinx.ext.linkcode', 'numpydoc.linkcode']:
  try:
    __import__(name)
    extensions.append(name)
    break
  except ImportError:
    pass
  else:
    print("NOTE: linkcode extension not found -- no links to source generated")


# This code was borrowed from Numpy's doc-to-source linker.
def linkcode_resolve(domain, info):
  """
  Determine the URL corresponding to Python object
  """
  if domain != 'py':
    return None

  modname = info['module']
  fullname = info['fullname']

  submod = sys.modules.get(modname)
  if submod is None:
    return None

  obj = submod
  for part in fullname.split('.'):
    try:
      obj = getattr(obj, part)
    except Exception:
      return None

  # strip decorators, which would resolve to the source of the decorator
  # possibly an upstream bug in getsourcefile, bpo-1764286
  try:
    unwrap = inspect.unwrap
  except AttributeError:
    pass
  else:
    obj = unwrap(obj)

  try:
    fn = inspect.getsourcefile(obj)
  except Exception:
    fn = None
  if not fn:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except Exception:
    lineno = None

  if lineno:
    linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
  else:
    linespec = ""

  fn = relpath(
      fn, start=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

  return "https://github.com/deepchem/deepchem/blob/master/%s%s" % (fn,
                                                                    linespec)
  # TODO: Should we do similar dev handling?
  #if 'dev' in numpy.__version__:
  #  return "https://github.com/numpy/numpy/blob/master/numpy/%s%s" % (
  #       fn, linespec)
  #else:
  #    return "https://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
  #       numpy.__version__, fn, linespec)


# Document __init__ methods
def skip(app, what, name, obj, would_skip, options):
  if name == "__init__":
    return False
  return would_skip


def setup(app):
  app.connect("autodoc-skip-member", skip)
