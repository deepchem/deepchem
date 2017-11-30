import ftplib
import os
import time

def main() :
    ftp = ftplib.FTP("ftp.ncbi.nih.gov")
    ftp.login("anonymous", "anonymous")
    ftp.cwd("/pubchem/Compound/CURRENT-Full/SDF")
    try :
        os.mkdir("/media/data/pubchem/SDF")
    except :
        print("Directory SDF already exists")

    filelist = ftp.nlst()
    existingfiles = os.listdir("/media/data/pubchem/SDF/")
    print("Downloading: {0} files".format(len(filelist)))
    i = 0
    for filename in filelist:

        local_filename = os.path.join('/media/data/pubchem/SDF', filename)
        if filename in existingfiles or "README" in filename:
            i = i + 1
            continue

        with open(local_filename, 'wb') as file :
            ftp.retrbinary('RETR ' + filename, file.write)
            i = i + 1

        print("Processed file {0} of {1}".format(i,len(filelist)))

    ftp.quit()

if __name__ == "__main__" :
    main()



