Documentation Tutorial
======================

This page provides instructions on how to build and test DeepChem documentation.

Building the Documentation
--------------------------

To build the docs, you can use the `Makefile` that's been added to
this directory. To generate docs in HTML, run the following commands:

.. code-block:: shell

   $ pip install -r requirements.txt
   $ make html
   # Clean build
   $ make clean html
   $ open build/html/index.html

Testing
-------

To check if the changes to the docs rendered properly, open build/html on a web browser.

If you want to confirm logs in more detail, use the following command:

.. code-block:: shell

   $ make clean html SPHINXOPTS=-vvv

If you want to confirm the example tests, run:

.. code-block:: shell

   $ make doctest_examples