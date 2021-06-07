Infrastructures
===============

The DeepChem project maintains supporting infrastructure on a number of
different services. This infrastructure is maintained by the DeepChem
development team.

GitHub
------
The core DeepChem repositories are maintained in the `deepchem`_ GitHub organization.
And, we use GitHub Actions to build a continuous integration pipeline.

.. _`deepchem`: https://github.com/deepchem

DeepChem developers have write access to the repositories on this repo and 
technical steering committee members have admin access.

Conda Forge
-----------
The DeepChem `feedstock`_ repo maintains the build recipe for conda-forge.

.. _`feedstock`: https://github.com/conda-forge/deepchem-feedstock

Docker Hub
----------
DeepChem hosts major releases and nightly docker build instances on `Docker Hub`_.

.. _`Docker Hub`: https://hub.docker.com/r/deepchemio/deepchem

PyPI
----
DeepChem hosts major releases and nightly builds on `PyPI`_.

.. _`PyPI`: https://pypi.org/project/deepchem/

Amazon Web Services
-------------------

DeepChem's website infrastructure is all managed on AWS through different AWS
services. All DeepChem developers have access to these services through the
deepchem-developers IAM role. (An IAM role controls access permissions.) At
present, @rbharath is the only developer with admin access to the IAM role, but
longer term we should migrate this so other folks have access to the roles.

S3
^^

Amazon's S3 allows for storage of data on "buckets" (Think of buckets like folders.)
There are two core deepchem S3 buckets:

  - deepchemdata: This bucket hosts the MoleculeNet datasets, pre-featurized datasets, 
    and pretrained models.

  - deepchemforum: This bucket hosts backups for the forums. The bucket is private for security reasons.
    The forums themselves are hosted on a digital ocean instance that only @rbharath currently has access to.
    Longer term, we should migrate the forums onto AWS so all DeepChem developers can access the forums.
    The forums themselves are a discord instance. The forums upload their backups to this S3 bucket once a day.
    If the forums crash, they can be restored from the backups in this bucket.

Route 53
^^^^^^^^
DNS for the deepchem.io website is handled by Route 53. The "hosted zone"
deepchem.io holds all DNS information for the website.

Certificate Manager
^^^^^^^^^^^^^^^^^^^
The AWS certificate manager issues the SSL/TLS certificate for the
\*.deepchem.io and deepchem.io domains.

GitHub Pages
^^^^^^^^^^
We make use of GitHub Pages to serve our static website. GitHub Pages
connects to the certificate in Certificate Manager. We set CNAME for
www.deepchem.io, and an A-record for deepchem.io.

The GitHub Pages repository is [deepchem/deepchem.github.io](https://github.com/deepchem/deepchem.github.io).

GoDaddy
-------
The deepchem.io domain is registered with GoDaddy. If you change the name
servers in AWS Route 53, you will need to update the GoDaddy record. At
present, only @rbharath has access to the GoDaddy account that owns the
deepchem.io domain name. We should explore how to provide access to the domain
name for other DeepChem developers.

Digital Ocean
-------------
The forums are hosted on a digital ocean instance. At present, only @rbharath
has access to this instance. We should migrate this instance onto AWS so other
DeepChem developers can help maintain the forums.
