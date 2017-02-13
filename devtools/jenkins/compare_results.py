from nose.tools import assert_true

CUSHION_PERCENT = 0.01
DESIRED_TO_SAMPLE_MAP = {
  "index": "Index splitting",
  "random": "Random splitting",
  "scaffold": "Scaffold splitting",
  "logreg": "logistic regression",
  "tf": "Multitask network",
  "tf_robust": "robust MT-NN",
  "graphconv": "graph convolution",
}


def find_desired_result(result, desired_results):
  vars = result.split(',')
  data_set, split, model = vars[1], DESIRED_TO_SAMPLE_MAP[vars[2]], DESIRED_TO_SAMPLE_MAP[vars[5]]
  for line in desired_results:
    desired_vars = line.split(',')
    if data_set == desired_vars[1] and split == desired_vars[0] and model == desired_vars[2]:
      return float(desired_vars[-2]), float(desired_vars[-1])
  raise Exception("Unable to find desired result \n%s" % result)


def get_my_results(result):
  vars = result.split(',')
  return float(vars[6]), float(vars[9])


def is_good_result(my_result, desired_result):
  for i in range(2):
    my_value = my_result[i]
    desired_value = desired_result[i]
    if my_value > desired_value * (1.0 + CUSHION_PERCENT):
      return False
  return True


def test_compare_results():
  desired_results = open("devtools/jenkins/desired_results.csv").readlines()
  given_results = open("results.csv").readlines()
  exceptions = []
  for result in given_results:
    desired_result = find_desired_result(result, desired_results)
    my_result = get_my_results(result)
    if not is_good_result(my_result, desired_result):
      exceptions.append((result, my_result, desired_result))
    if len(exceptions) > 0:
      for exception in exceptions:
        print(exception)
    assert_true(len(exceptions) == 0, "Some performance benchmarks not passed")

  if __name__ == "__main__":
    test_compare_results()
