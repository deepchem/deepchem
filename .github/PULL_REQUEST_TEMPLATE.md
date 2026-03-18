## Description

Fix #(issue)

<!-- Please include a summary of the change and which issue is fixed.
Please also include relevant motivation and context.
List any dependencies that are required for this change. -->


## Type of change

Please check the option that is related to your PR.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
  - In this case, we recommend to discuss your modification on GitHub issues before creating the PR
- [ ] Documentations (modification for documents)

## Checklist

- [ ] My code follows [the style guidelines of this project](https://deepchem.readthedocs.io/en/latest/development_guide/coding.html)
  - [ ] Run `yapf -i <modified file>` and check no errors (**yapf version must be  0.32.0**)
  - [ ] Run `mypy -p deepchem` and check no errors
  - [ ] Run `flake8 <modified file> --count` and check no errors
  - [ ] Run `python -m doctest <modified file>` and check no errors
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New unit tests pass locally with my changes
- [ ] I have checked my code and corrected any misspellings
