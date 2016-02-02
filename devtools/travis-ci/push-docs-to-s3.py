import os
import pip
import tempfile
import subprocess


BUCKET_NAME = 'deepchem.io'

if not any(d.project_name == 's3cmd' for d in pip.get_installed_distributions()):
  raise ImportError('The s3cmd pacakge is required. try $ pip install s3cmd')

# The secret key is available as a secure environment variable
# on travis-ci to push the build documentation to Amazon S3.
with tempfile.NamedTemporaryFile('w') as f:
  f.write('''[default]
access_key = {AWS_ACCESS_KEY_ID}
secret_key = {AWS_SECRET_ACCESS_KEY}
'''.format(**os.environ))
  f.flush()

  template = ('s3cmd -M --config {config} '
              'sync docs/_build/ s3://{bucket}/')
  cmd = template.format(
          config=f.name,
          bucket=BUCKET_NAME)
  subprocess.call(cmd.split())
