import ftplib
import os
import time
import deepchem


def main():
  ftp = ftplib.FTP("ftp.ncbi.nih.gov")
  ftp.login("anonymous", "anonymous")

  # First download all SDF files. We need these to get smiles
  ftp.cwd("/pubchem/Compound/CURRENT-Full/SDF")
  data_dir = deepchem.utils.get_data_dir()
  sdf_dir = os.path.join(data_dir,"SDF")
  if not os.path.exists(sdf_dir):
    os.mkdir(sdf_dir)

  filelist = ftp.nlst()
  existingfiles = os.listdir(sdf_dir)
  print("Downloading: {0} SDF files".format(len(filelist)))

  i = 0
  for filename in filelist:

    local_filename = os.path.join(sdf_dir, filename)
    if filename in existingfiles or "README" in filename:
        i = i + 1
        continue

    with open(local_filename, 'wb') as file :
        ftp.retrbinary('RETR ' + filename, file.write)
        i = i + 1

    # Next download all Bioassays
    ftp.cwd("/pubchem/Bioassay/CSV/Data")
    data_dir = deepchem.utils.get_data_dir()
    bioassay_dir = os.path.join(data_dir, "Data")
    if not os.path.exists(bioassay_dir):
      os.mkdir(bioassay_dir)

    filelist = ftp.nlst()
    existingfiles = os.listdir(bioassay_dir)
    print("Downloading: {0} Bioassay files".format(len(filelist)))

    i = 0
    for filename in filelist:

      local_filename = os.path.join(bioassay_dir, filename)
      if filename in existingfiles or "README" in filename:
        i = i + 1
        continue

      with open(local_filename, 'wb') as file:
        ftp.retrbinary('RETR ' + filename, file.write)
        i = i + 1

      print("Processed file {0} of {1}".format(i, len(filelist)))

  ftp.quit()

if __name__ == "__main__" :
    main()



