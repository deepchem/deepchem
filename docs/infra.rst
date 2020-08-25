DeepChem Infrastructure
=======================

The DeepChem project maintains supporting infrastructure on a number of
different services. This infrastructure is maintained by the DeepChem
development team.

Github
------
The core DeepChem repositories are maintained in the `deepchem`_ GitHub organization.

.. _`deepchem`: https://github.com/deepchem

DeepChem developers have write access to the repositories on this repo and technical steering committee members have admin access.

Travis CI
---------
DeepChem runs continuous integration tests on `Travis CI`_.

.. _`Travis CI`: https://travis-ci.org/github/deepchem

Conda Forge
-----------
The DeepChem `feedstock`_ repo maintains the build recipe for Conda-Forge.

.. _`feedstock`: https://github.com/conda-forge/deepchem-feedstock


Dockerhub
---------
DeepChem hosts nightly docker build instances on `dockerhub`_.

.. _`dockerhub`: https://hub.docker.com/r/deepchemio/deepchem

PyPi
----
DeepChem hosts major releases and nightly builds on `pypi`_.

.. _`pypi`: https://pypi.org/project/deepchem/

Amazon Web Services
-------------------

DeepChem's website infrastructure is all managed on AWS through different AWS
services. All DeepChem developers have access to these services through the
deepchem-developers IAM role. (An IAM role controls access permissions.) At
present, @rbharath is the only developer with access to the IAM role, but
longer term we should migrate this so other folks have access to the roles.

S3
^^

Amazon's S3 allows for storage of data on "buckets" (Think of buckets like folders.) There are two core deepchem S3 buckets:

  - deepchemdata: This bucket hosts the deepchem.io website, MoleculeNet datasets, pre-featurized datasets, and pretrained models. This bucket is set up to host a static website (at `static`_).
  - deepchemforum: This bucket hosts backups for the forums. The bucket is private for security reasons. The forums themselves are hosted on a digital ocean instance that only @rbharath currently has access to. Longer term, we should migrate the forums onto AWS so all DeepChem developers can access the forums. The forums themselves are a discord instance. The forums upload their backups to this S3 bucket once a day. If the forums crash, they can be restored from the backups in this bucket

.. _`static`: https://deepchemdata.s3-us-west-1.amazonaws.com/index.html

Route 53
^^^^^^^^
DNS for the deepchem.io website is handled by Route 53. The "hosted zone"
deepchem.io holds all DNS information for the website.

Certificate Manager
^^^^^^^^^^^^^^^^^^^
The AWS certificate manager issues the SSL/TLS certificate for the
\*.deepchem.io and deepchem.io domains.


Cloudfront
^^^^^^^^^^
We make use of a cloudfront distribution to serve our static website. The
cloudfront distribution connects to the certificate in Certificate Manager and
uses the deepchemdata bucket as the origin domain. We set CNAME for
www.deepchem.io and deepchem.io

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
