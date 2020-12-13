# Contributing to DeepChem

We actively encourage community contributions to DeepChem. The first place to start getting involved is [the tutorials](https://deepchem.readthedocs.io/en/latest/get_started/tutorials.html). Afterwards, we encourage contributors to give a shot to improving our documentation. While we take effort to provide good docs, there's plenty of room for improvement. All docs are hosted on Github, either in `README.md` file, or in the `docs/` directory.

Once you've got a sense of how the package works, we encourage the use of Github issues to discuss more complex changes, raise requests for new features or propose changes to the global architecture of DeepChem. Once consensus is reached on the issue, please submit a PR with proposed modifications. All contributed code to DeepChem will be reviewed by a member of the DeepChem team, so please make sure your code style and documentation style match our guidelines!

## Pull Request Process

Every contribution, must be a pull request and must have adequate time for review by other committers.

A member of the Technical Steering Committee will review the pull request. The default path of every contribution should be to merge. The discussion, review, and merge process should be designed as corrections that move the contribution into the path to merge. Once there are no more corrections, (dissent) changes should merge without further process.

On successful merge the author will be added as a member of the DeepChem organization.

## Coding Conventions

DeepChem uses these tools or styles for keeping our codes healthy.

- [YAPF](https://github.com/google/yapf) (code format)
- [Flake8](https://flake8.pycqa.org/en/latest/) (code style check)
- [mypy](http://mypy-lang.org/) (type check)
- [Numpy Doctsring](https://numpydoc.readthedocs.io/en/latest/index.html) (API documents)
- [doctest](https://docs.python.org/3/library/doctest.html) (interactive examples)

Before making a PR, please check your codes using them.
You can confirm how to check your codes from [Coding Conventions](https://deepchem.readthedocs.io/en/latest/development_guide/coding.html).


## Document Conventions

DeepChem use [Sphinx](https://www.sphinx-doc.org/en/master/) to build [the document](https://deepchem.readthedocs.io/en/latest/index.html).
The document is automatically built by [Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide) in source codes and [Napoleon extension](http://www.sphinx-doc.org/en/stable/ext/napoleon.html).
It means you must add or modify Numpy style docstrings if you modify our codes.

Before making a PR, please don't forget to modify not only codes but also docstrings.

## The Agreement

Contributor offers to license certain software (a “Contribution” or multiple “Contributions”) to DeepChem, and DeepChem agrees to accept said Contributions, under the terms of the open source license [The MIT License](https://opensource.org/licenses/MIT)

The Contributor understands and agrees that DeepChem shall have the irrevocable and perpetual right to make and distribute copies of any Contribution, as well as to create and distribute collective works and derivative works of any Contribution, under [The MIT License](https://opensource.org/licenses/MIT).

DeepChem understands and agrees that Contributor retains copyright in its Contributions. Nothing in this Contributor Agreement shall be interpreted to prohibit Contributor from licensing its Contributions under different terms from the [The MIT License](https://opensource.org/licenses/MIT) or this Contributor Agreement.

## DeepChem Technical Steering Committee

The Technical Steering Committee admits and oversees all top-level of DeepChem.

The TSC exercises autonomy in setting up and maintaining procedures, policies, and management and administrative structures as it deems appropriate for the maintenance and operation of these projects and resources.
Included in the responsibilities of the TSC are:

* Managing code and documentation creation and changes for the listed projects and resources
* Performing code reviews on incoming pull requests and merging suitable code changes.
* Setting and maintaining standards covering contributions of code, documentation and other materials
* Managing code and binary releases: types, schedules, frequency, delivery mechanisms
* Making decisions regarding dependencies of DeepChem, including what those dependencies are and how they are bundled with source code and releases
* Creating new repositories and projects under the deepchem GitHub organization as required
* Setting overall technical direction for the DeepChem project, including high-level goals and low-level specifics regarding features and functionality
* Setting and maintaining appropriate standards for community discourse via the various mediums under TSC control (gitter, facebook, blog)

Members of the TSC will meet regularly (over phone or video conferencing) to coordinate efforts. Minutes from the TSC meetings will be published publicly on an ongoing basis.
The current members of the TSC are (alphabetically)
* Peter Eastman
* Karl Leswing
* Bharath Ramsundar
* Zhenqin Wu

If you want to join the technical steering committee, you will need to submit an application. The application process is relatively lightweight: submit a one page document discussing your past contributions to DeepChem and propose potential projects you could commit to working on as a member of the steering committee. Note that steering committee membership comes with responsibilities. In particular, you will need to commit to spending about 10 hours a week working on DeepChem. The committee will review your application, and if suitable, will accept you as a probationary member of the TSC. Your application will be posted publicly to the DeepChem blog if accepted. Membership on the committee will be confirmed after 6 months if you’ve successfully implemented some of your proposed projects and demonstrated your ability to meet the necessary time commitment.
