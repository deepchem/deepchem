echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


if [[ "$python" == "2.7" ]]; then
    # Create the docs and push them to S3
    # -----------------------------------

    conda install --yes `conda build devtools/conda-recipe --output`
    pip install numpydoc s3cmd
    conda install --yes `cat docs/requirements.txt | xargs`

    conda list -e
    mkdir -p docs/_build
    sphinx build -b html docs docs/_build
    python devtools/travis-ci/push-docs-to-s3.py
fi
