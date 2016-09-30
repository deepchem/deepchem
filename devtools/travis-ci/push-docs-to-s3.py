import os
import pip
import tempfile
import subprocess


BUCKET_NAME = 'deepchem.io'

if not any(d.project_name == 's3cmd' for d in pip.get_installed_distributions()):
  raise ImportError('The s3cmd package is required. try $ pip install s3cmd')

# The secret key is available as a secure environment variable
# on travis-ci to push the build documentation to Amazon S3.
with tempfile.NamedTemporaryFile('w') as f:
  f.write('''[default]
access_key = {AWS_ACCESS_KEY_ID}
secret_key = {AWS_SECRET_ACCESS_KEY}
default_mime_type = binary/octet-stream
guess_mime_type = True
'''.format(**os.environ))
  f.flush()

  #s3cmd -M -H sync docs/_build/ s3://deepchem.io/
  template = ('s3cmd -M -H --config {config} '
              'sync docs/_build/ s3://{bucket}/')
  cmd = template.format(
          config=f.name,
          bucket=BUCKET_NAME)
  subprocess.call(cmd.split())
