Docker Tutorial
================

Docker is a software used for easy building, testing and deploying of software. Docker creates an isolated workspace called containers which can avoid dependency version clashes making development of software faster. Also, software can be modularized in different containers, which allows it to be tested without impacting other components or the host computer. Containers contain all the dependencies and the user need not worry about required packages  

**This makes it easy for users to access older version of deepchem via docker and to develop with them.**

Docker works with the following layers:

- Images:

*Images are the instructions for creating docker containers. It specifies all the packages and their version to be installed fo the application to run. Images for deep chem can found at docker Hub.*

- Containers:

*Containers are live instances of Images and are lightweight isolated work-spaces(it does not put much workload on your PC), where you can run and devlop on previous deepchem versions.*

- Docker engine:

*It is the main application that manages, runs and build containers and images. It also provides a means to interact with the docker container after its built and when it is run.*

- Registries:

*It is a hub or place where docker images can be found. For deepchem, the default registry is the Docker Hub.*

**For docker installation, visit:** https://docs.docker.com/engine/install/

Using deepchem with docker:
---------------------------
To work with deepchem in docker, we first have to pull deepchem images from docker hub. It can be done in the following way.

if latest deepchem version is needed, then:-

.. code-block:: bash
    
    #if latest:
    docker pull deepchemio/deepchem:latest

Else if one wants to work with older version, then the following method should be used:-

.. code-block:: bash

    docker pull deepchemio/deepchem:x.x.x
    #x.x.x refers to the version number

Now, wait for some time until the image gets downloaded. Then we have to create a container using the image.
Then, you have to create a container and use it.

.. code-block:: bash

    docker run --rm -it deepchemio/deepchem:x.x.x
    #x.x.x refers to the version number
    #replace "x.x.x" with "latest" if latest version is used

If you want GPU support:

.. code-block:: bash

    # If nvidia-docker is installed
    nvidia-docker run --rm -it deepchemio/deepchem:latest
    docker run --runtime nvidia --rm -it deepchemio/deepchem:latest

    # If nvidia-container-toolkit is installed
    docker run --gpus all --rm -it deepchemio/deepchem:latest

Now, you have successfully entered the container's bash where you can execute your programs.
**To exit the container press "Ctrl+D".** This stops the container and opens host computer's bash.

To view all the containers present, open up a new terminal/bash of the host computer, then:-

.. code-block:: bash
    
   docker ps -a

This gives a containers list like this:

.. code-block:: bash
    
   CONTAINER ID   IMAGE                       COMMAND           CREATED       STATUS       PORTS     NAMES

Thus you can see all the created container's Names and its details.

*Now you can develop code in you host computer(development environment) and test it in a container having specific version of the deepchem(testing environment).*

To test the program you have written, you should copy the program to the container. Open a new host computer's terminal:

.. code-block:: bash
    
   docker cp host-file-path <container-id>:path-in-container
   #container ID should be check in a separate terminal

Similarly if you want to copy files out from a container, then open a new host computer's terminal:

.. code-block:: bash
    
   docker cp <container-id>:path-in-container host-file-path
   #container ID should be check in a separate terminal

Hands-on tutorial
-----------------
Lets create a simple deepchem script and work it out in the docker container of deepchem 2.4.0.

Let the script be named deepchem.py in the host computer's location: /home/

*deepchem.py contents:*

.. code-block:: bash
   
   import deepchem as dc

   print(dc.__version__)

*Step 1:* pull deepchem 2.4.0 image and wait for it to be dowloaded

.. code-block:: bash

    $docker pull deepchemio/deepchem:2.4.0
    
*Step 2:* Create a container

.. code-block:: bash

    $docker run --rm -it deepchemio/deepchem:2.4.0
    (deepchem) root@51b1d2665016:~/mydir# 

*Step 3:* Open a new terminal/bash and copy deep.py

.. code-block:: bash

    $docker ps -a
    CONTAINER ID   IMAGE                       COMMAND       CREATED         STATUS         PORTS     NAMES
    51b1d2665016   deepchemio/deepchem:2.4.0   "/bin/bash"   5 seconds ago   Up 4 seconds             friendly_lehmann
    $docker cp /home/deepchem.py 51b1d2665016:/root/mydir/deepchemp.py

*step 4:* return back to the previous terminal in which container is runing

.. code-block:: bash

   (deepchem) root@51b1d2665016:~/mydir#python3 deepchem.py>>output.txt
   2022-01-12 15:33:27.967170: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1

This should have created a output file in the container having the deepchem version number. The you should copy it back to host container.

*step 5:* In a new terminal execute the following commands.

.. code-block:: bash
   
   $docker cp 51b1d2665016:/root/mydir/output.txt ~/output.txt
   $cat ~/output.txt
   2.4.0

Thus you have successfully executed the program in deepchem 2.4.0!!!
