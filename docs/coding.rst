Coding Conventions
==================

Code Formatting
---------------

.. _`yapf`: https://github.com/google/yapf

We use `yapf`_ to format all of the code in DeepChem.  Although it sometimes
produces slightly awkward formatting, it does have two major benefits.  First,
it ensures complete consistency throughout the entire codebase.  And second, it
avoids disagreements about how a piece of code should be formatted.

Whenever you modify a file, run :code:`yapf` on it to reformat it before
checking it in.

.. code-block:: bash

  yapf -i <modified file>

Yapf is run on every pull request to make sure the formatting is correct, so if
you forget to do this the continuous integration system will remind you.
Because different versions of yapf can produce different results, it is
essential to use the same version that is being run on CI.  At present, that
is 0.22.  We periodically update it to newer versions.


Docstrings
----------

All classes and functions should include docstrings describing their purpose and
intended usage.  When in doubt about how much information to include, always err
on the side of including more rather than less.  Explain what problem a class is
intended to solve, what algorithms it uses, and how to use it correctly.  When
appropriate, cite the relevant publications.

.. _`numpy`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

All docstrings should follow the `numpy`_ docstring formatting conventions.


Unit Tests
----------

Having an extensive collection of test cases is essential to ensure the code
works correctly.  If you haven't written tests for a feature, that means the
feature isn't finished yet.  Untested code is code that probably doesn't work.

Complex numerical code is sometimes challenging to fully test.  When an
algorithm produces a result, it sometimes is not obvious how to tell whether the
result is correct or not.  As far as possible, try to find simple examples for
which the correct answer is exactly known.  Sometimes we rely on stochastic
tests which will *probably* pass if the code is correct and *probably* fail if
the code is broken.  This means these tests are expected to fail a small
fraction of the time.  Such tests can be marked with the :code:`@flaky`
annotation.  If they fail during continuous integration, they will be run a
second time and an error only reported if they fail again.

If possible, each test should run in no more than a few seconds.  Occasionally
this is not possible.  In that case, mark the test with the :code:`@pytest.mark.slow`
annotation.  Slow tests are skipped during continuous integration, so changes
that break them may sometimes slip through and get merged into the repository.
We still try to run them regularly, so hopefully the problem will be discovered
fairly soon.

Testing Machine Learning Models
-------------------------------

Testing the correctness of a machine learning model can be quite tricky to do
in practice. When adding a new machine learning model to DeepChem, you should
add at least a few basic types of unit tests:

- Overfitting test: Create a small synthetic dataset and test that your model
  can learn this datasest with high accuracy. For regression and classification
  task, this should correspond to low training error on the dataset. For
  generative tasks, this should correspond to low training loss on the dataset.
- Reloading test: Check that a trained model can be saved to disk and reloaded
  correctly. This should involve checking that predictions from the saved and
  reloaded models
  matching exactly.

Note that unit tests are not sufficient to gauge the real performance of a
model. You should benchmark your model on larger datasets as well and report
your benchmarking tests in the PR comments.

Type Annotations
----------------

Type annotations are an important tool for avoiding bugs.  All new code should
provide type annotations for function arguments and return types.  When you make
significant changes to existing code that does not have type annotations, please
consider adding them at the same time.

.. _`mypy`: http://mypy-lang.org/

We use the `mypy`_ static type checker to verify code correctness.  It is
automatically run on every pull request.  If you want to run it locally to make
sure you are using types correctly before checking in your code, :code:`cd` to
the top level directory of the repository and execute the command

.. code-block:: bash

  mypy -p deepchem --ignore-missing-imports

Because Python is such a dynamic language, it sometimes is not obvious what type
to specify.  A good rule of thumb is to be permissive about input types and
strict about output types.  For example, many functions are documented as taking
a list as an argument, but actually work just as well with a tuple.  In those
cases, it is best to specify the input type as :code:`Sequence` to accept either
one.  But if a function returns a list, specify the type as :code:`List` because
we can guarantee the return value will always have that exact type.

Another important case is NumPy arrays.  Many functions are documented as taking
an array, but actually can accept any array-like object: a list of numbers, a
list of lists of numbers, a list of arrays, etc.  In that case, specify the type
as :code:`Sequence` to accept any of these.  On the other hand, if the function
truly requires an array and will fail with any other input, specify it as
:code:`np.ndarray`.

The :code:`deepchem.utils.typing` module contains definitions of some types that
appear frequently in the DeepChem API.  You may find them useful when annotating
code.
