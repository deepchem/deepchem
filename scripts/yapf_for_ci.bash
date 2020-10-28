CHANGED_FILES=`git diff --name-only $TRAVIS_COMMIT_RANGE | grep .py$ | grep -v contrib/`

if [ -z $CHANGED_FILES ]
then
  echo "No Python Files Changed"
  echo "Passed Formatting Test"
  return 1
fi

yapf -d $CHANGED_FILES > diff.txt

if [ -s diff.txt ]
then
  cat diff.txt
  echo ""
  echo "Failing Formatting Test"
  echo "Please run yapf over the files changed"
  echo "pip install yapf==0.22.0"
  echo "yapf -i $CHANGED_FILES"
else
  echo "Passed Formatting Test"
fi
