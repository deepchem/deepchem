from nose.tools import assert_true, nottest

CUSHION_PERCENT = 0.01
LOG_ALL_RESULTS = False

BENCHMARK_TO_DESIRED_KEY_MAP = {
    "index": "Index splitting",
    "random": "Random splitting",
    "scaffold": "Scaffold splitting",
    "logreg": "logistic regression",
    "tf": "Multitask network",
    "tf_robust": "robust MT-NN",
    "tf_regression": "NN regression",
    "graphconv": "graph convolution",
    "graphconvreg": "graphconv regression",
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
        "split": BENCHMARK_TO_DESIRED_KEY_MAP[vars[1]],
        "data_set": vars[0],
        "model": BENCHMARK_TO_DESIRED_KEY_MAP[vars[3]],
        "train_score": float(vars[6]),
        "test_score": float(vars[8])
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
  retval = True
  message = []
  for key in ['train_score', 'test_score']:
    # Higher is Better
    desired_value = desired_result[key] - CUSHION_PERCENT
    if my_result[key] < desired_value or LOG_ALL_RESULTS:
      message_part = "%s,%s,%s,%s,%s,%s" % (my_result['data_set'],
                                            my_result['model'],
                                            my_result['split'], key,
                                            my_result[key], desired_result[key])
      message.append(message_part)
      retval = False
  return retval, message


def test_compare_results():
  desired_results = open(DESIRED_RESULTS_CSV).readlines()[1:]
  desired_results = parse_desired_results(desired_results)
  test_results = open(TEST_RESULTS_CSV).readlines()
  test_results = parse_test_results(test_results)
  failures = []
  exceptions = []
  for test_result in test_results:
    try:
      desired_result = find_desired_result(test_result, desired_results)
      passes, message = is_good_result(test_result, desired_result)
      if not passes:
        failures.extend(message)
    except Exception as e:
      exceptions.append("Unable to find desired result for %s" % test_result)
  for exception in exceptions:
    print(exception)
  for failure in failures:
    print(failure)
  assert_true(len(exceptions) == 0, "Error parsing performance results")
  assert_true(len(failures) == 0, "Some performance benchmarks not passed")

  if __name__ == "__main__":
    test_compare_results()
