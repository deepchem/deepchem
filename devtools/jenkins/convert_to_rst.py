import os
import delegator
import shutil

target_dir = "../../docs/source/notebooks/"


def convert_to_rst(fname):
  cmd = "jupyter-nbconvert --to rst %s" % fname
  c = delegator.run(cmd)

  base_name = os.path.splitext(fname)[0]
  image_files = "%s_files" % base_name
  new_path = os.path.join(target_dir, image_files)
  if os.path.isdir(new_path):
    shutil.rmtree(new_path)
  if os.path.isdir(image_files):
    shutil.move(image_files, target_dir)

  rst_file = '%s.rst' % base_name
  new_path = os.path.join(target_dir, rst_file)
  if os.path.isfile(new_path):
    os.remove(new_path)
  shutil.move(rst_file, target_dir)


def main():
  fnames = os.listdir('./')
  fnames = [x for x in filter(lambda x: x.endswith('ipynb') > 0, fnames)]
  for fname in fnames:
    convert_to_rst(fname)


if __name__ == "__main__":
  main()
