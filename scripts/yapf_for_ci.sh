# This script may work on only Bash and Zsh
# usage: source scripts/yapf_for_ci.sh

CHANGED_FILES=`git diff --name-only $COMMIT_RANGE | grep .py$ | grep -v contrib/`

if [ -z $CHANGED_FILES ]; then
  echo "No Python Files Changed"
  echo "Passed Formatting Test"
  return 0
fi

yapf -d $CHANGED_FILES > diff.txt

if [ -s diff.txt ]; then
  cat diff.txt
  echo ""
  echo "Failing Formatting Test"
  echo "Please run yapf over the files changed"
  echo "pip install yapf==0.22.0"
  echo "yapf -i $CHANGED_FILES"
  return 1
else
  echo "Passed Formatting Test"
  return 0
fi
