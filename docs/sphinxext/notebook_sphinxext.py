# Copied from the yt_project, commit e8fb57e
# yt/doc/extensions/notebook_sphinxext.py
#  https://bitbucket.org/yt_analysis/yt/src/e8fb57e66ca42e26052dadf054a5c782740abec9/doc/extensions/notebook_sphinxext.py?at=yt

# Almost completely re-written by Matthew Harrigan to use nbconvert v4

from __future__ import print_function

import os
import shutil

from sphinx.util.compat import Directive
from docutils import nodes
from docutils.parsers.rst import directives
import nbformat
from nbconvert import HTMLExporter, PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor


def export_html(nb, f):
  config = {
      'Exporter': {
          'template_file': 'basic',
          'template_path': ['./sphinxext/']
      },
      'ExtractOutputPreprocessor': {
          'enabled': True
      },
      'CSSHTMLHeaderPreprocessor': {
          'enabled': True
      }
  }

  exporter = HTMLExporter(config)
  body, resources = exporter.from_notebook_node(
      nb, resources={'output_files_dir': f['nbname']})

  for fn, data in resources['outputs'].items():
    bfn = os.path.basename(fn)
    with open("{destdir}/{fn}".format(fn=bfn, **f), 'wb') as res_f:
      res_f.write(data)

  return body


def export_python(nb, destfn):
  exporter = PythonExporter()
  body, resources = exporter.from_notebook_node(nb)
  with open(destfn, 'w') as f:
    f.write(body)


class NotebookDirective(Directive):
  """Insert an evaluated notebook into a document
    """
  required_arguments = 1
  optional_arguments = 1
  option_spec = {'skip_exceptions': directives.flag}
  final_argument_whitespace = True

  def run(self):
    f = {
        'docdir': setup.confdir,
        'builddir': setup.app.builder.outdir,
        'nbname': self.arguments[0],
    }
    f['nbpath'] = "{docdir}/../../examples/notebooks/{nbname}.ipynb".format(**f)
    f['destdir'] = "{builddir}/notebooks/{nbname}".format(**f)

    if not os.path.exists(f['destdir']):
      os.makedirs(f['destdir'])

    f['uneval'] = "{destdir}/{nbname}.ipynb".format(**f)
    f['eval'] = "{destdir}/{nbname}.eval.ipynb".format(**f)
    f['py'] = "{destdir}/{nbname}.py".format(**f)

    # 1. Uneval notebook
    shutil.copyfile(f['nbpath'], f['uneval'])
    with open(f['nbpath']) as nb_f:
      nb = nbformat.read(nb_f, as_version=4)
    # 2. Python
    export_python(nb, f['py'])
    # 3. HTML (execute first)
    # Set per-cell timeout to 60 seconds
    executer = ExecutePreprocessor(timeout=60)
    executer.preprocess(nb, {})
    html = export_html(nb, f)
    # 4. Eval'd notebook
    with open(f['eval'], 'w') as eval_f:
      nbformat.write(nb, eval_f)

    # Create link to notebook and script files
    link_rst = "({uneval}; {eval}; {py})".format(
        uneval=formatted_link(f['uneval']),
        eval=formatted_link(f['eval']),
        py=formatted_link(f['py']),)

    rst_file = self.state_machine.document.attributes['source']
    self.state_machine.insert_input([link_rst], rst_file)

    # create notebook node
    nb_node = notebook_node('', html, format='html', source='nb_path')
    nb_node.source, nb_node.line = (
        self.state_machine.get_source_and_line(self.lineno))

    # add dependency
    self.state.document.settings.record_dependencies.add(f['nbpath'])
    return [nb_node]


class notebook_node(nodes.raw):
  pass


def formatted_link(path):
  return "`%s <%s>`__" % (os.path.basename(path), path)


def visit_notebook_node(self, node):
  self.visit_raw(node)


def depart_notebook_node(self, node):
  self.depart_raw(node)


def setup(app):
  setup.app = app
  setup.config = app.config
  setup.confdir = app.confdir

  app.add_node(notebook_node, html=(visit_notebook_node, depart_notebook_node))

  app.add_directive('notebook', NotebookDirective)
