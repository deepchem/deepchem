from nose.tools import assert_true, nottest

CUSHION_PERCENT = 0.01
BENCHMARK_TO_DESIRED_KEY_MAP = {
  "index": "Index splitting",
  "random": "Random splitting",
  "scaffold": "Scaffold splitting",
  "logreg": "logistic regression",
  "tf": "Multitask network",
  "tf_robust": "robust MT-NN",
  "graphconv": "graph convolution",
}
DESIRED_RESULTS_CSV = "devtools/jenkins/desired_results.csv"
TEST_RESULTS_CSV = "examples/results.csv"


def parse_desired_results(desired_results):
  retval = []
  for line in desired_results:
    vars = line.split(',')
    retval.append({
      "split": vars[0],
      "data_set": vars[1],
      "model": vars[2],
      "train_score": float(vars[3]),
      "test_score": float(vars[4])
    })
  return retval


@nottest
def parse_test_results(test_results):
  retval = []
  for line in test_results:
    vars = line.split(',')
    retval.append({
      "split": BENCHMARK_TO_DESIRED_KEY_MAP[vars[2]],
      "data_set": vars[1],
      "model": BENCHMARK_TO_DESIRED_KEY_MAP[vars[5]],
      "train_score": float(vars[6]),
      "test_score": float(vars[9])
    })
  return retval


def find_desired_result(result, desired_results):
  for desired_result in desired_results:
    if result['data_set'] == desired_result['data_set'] and \
        result['split'] == desired_result['split'] and \
        result['model'] == desired_result['model']:
      return desired_result
  raise Exception("Unable to find desired result \n%s" % result)


def is_good_result(my_result, desired_result):
  for key in ['train_score', 'test_score']:
    # Higher is Better
    desired_value = desired_result[key] * (1.0 - CUSHION_PERCENT)
    if my_result[key] < desired_value:
      return False
  return True


def test_compare_results():
  desired_results = open(DESIRED_RESULTS_CSV).readlines()[1:]
  desired_results = parse_desired_results(desired_results)
  test_results = open(TEST_RESULTS_CSV).readlines()
  test_results = parse_test_results(test_results)
  exceptions = []
  for test_result in test_results:
    desired_result = find_desired_result(test_result, desired_results)
    if not is_good_result(test_result, desired_result):
      exceptions.append(({"test_result": test_result}, {"desired_result": desired_result}))
  if len(exceptions) > 0:
    for exception in exceptions:
      print(exception)
    assert_true(len(exceptions) == 0, "Some performance benchmarks not passed")

  if __name__ == "__main__":
    test_compare_results()
