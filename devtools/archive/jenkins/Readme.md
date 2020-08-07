# Usage Instructions
On Stanford Internal Network http://171.65.103.54:8080/

username: deepAdmin

password: check sticky-note on tensorbro 

# Moving Instructions
All the configuration nesseccary to run jenkins is in /var/jenkins

Simply rsync /var/jenkins to the new server.

# Install Instructions
#### Update the System
``` bash
sudo apt-get update
sudo apt-get upgrade
```
#### Install Build Dependency Packages
``` bash
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev libcurl3-dev
```

#### Install latest Cuda and cudnn
``` bash 
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda
sudo dpkg -i libcudnn5_5.1.5-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn5-dev_5.1.5-1+cuda8.0_amd64.deb
```

#### Install Jenkins
``` bash
wget -q -O - https://pkg.jenkins.io/debian/jenkins-ci.org.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt-get update
sudo apt-get install jenkins
``` 

#### Install Conda For the Jenkins User
``` bash
sudo su - jenkins
bash Anaconda3-4.3.0-Linux-x86_64.sh
```

#### Configure Through Web-UI
``` bash
GOTO http://server:8080
Install Suggested Plugins Through Web-UI
Create First Admin User
```
