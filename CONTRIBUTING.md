# DeepChem

## Contributing to DeepChem

We actively encourage community contributions to DeepChem. The first place to start getting involved is by running our examples locally. Afterwards, we encourage contributors to give a shot to improving our documentation. While we take effort to provide good docs, there's plenty of room for improvement. All docs are hosted on Github, either in `README.md` file, or in the `docs/` directory.

Once you've got a sense of how the package works, we encourage the use of Github issues to discuss more complex changes,  raise requests for new features or propose changes to the global architecture of DeepChem. Once consensus is reached on the issue, please submit a PR with proposed modifications. All contributed code to DeepChem will be reviewed by a member of the DeepChem team, so please make sure your code style and documentation style match our guidelines!

## Contributor License Agreement
In order to get a pull request accepted you must fill out our [License Agreement](https://www.clahub.com/agreements/deepchem/deepchem).  The purpose of this agreement is to enable DeepChem to distribute your code and its derivatives.

### The Agreement
Contributor offers to license certain software (a “Contribution” or multiple “Contributions”) to DeepChem, and DeepChem agrees to accept said Contributions, under the terms of the open source license [The MIT License](https://opensource.org/licenses/MIT)


The Contributor understands and agrees that DeepChem shall have the irrevocable and perpetual right to make and distribute copies of any Contribution, as well as to create and distribute collective works and derivative works of any Contribution, under [The MIT License](https://opensource.org/licenses/MIT).


DeepChem understands and agrees that Contributor retains copyright in its Contributions. Nothing in this Contributor Agreement shall be interpreted to prohibit Contributor from licensing its Contributions under different terms from the [The MIT License](https://opensource.org/licenses/MIT) or this Contributor Agreement.

### Code Style Guidelines
DeepChem uses [yapf](https://github.com/google/yapf) to autoformat code.

``` bash
pip install yapf==0.17.0
cd <git_root>
yapf -i <python_files changed>
```

Our integration tests will fail if code is not formatted correctly

### Documentation Style Guidelines
DeepChem uses [NumPy style documentation](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt). Please follow these conventions when documenting code, since we use [Sphinx+Napoleon](http://www.sphinx-doc.org/en/stable/ext/napoleon.html) to automatically generate docs on [deepchem.io](deepchem.io).

