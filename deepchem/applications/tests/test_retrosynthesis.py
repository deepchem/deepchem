import logging
from unittest.mock import patch

import pytest

try:
  from aizynthfinder.chem import RetroReaction
except (ModuleNotFoundError, ImportError):
  NO_AIZYNTHFINDER = True
else:
  NO_AIZYNTHFINDER = False

from deepchem.applications.synthesis import RetrosynthesisEngine


@pytest.fixture
def mock_load_policy():

  def load_from_config(self, **kwargs):
    for name in kwargs:
      self._items[name] = None

  def load(self, **kwargs):
    self._items[kwargs["key"]] = None

  patchers = []

  def wrapper(*class_names):
    for class_name in class_names:
      patchers.append(
          patch(
              f"aizynthfinder.context.policy.{class_name}.load_from_config",
              new=load_from_config,
          ))
      patchers.append(
          patch(f"aizynthfinder.context.policy.{class_name}.load", new=load))
    for patcher in patchers:
      patcher.start()

  yield wrapper

  for patcher in patchers:
    patcher.stop()


@pytest.fixture
def mock_expansion_policy():

  smarts = "([#8:4]-[N;H0;D3;+0:5](-[C;D1;H3:6])-[C;H0;D3;+0:1](-[C:2])=[O;D1;H0:3])>>(Cl-[C;H0;D3;+0:1](-[C:2])=[O;D1;H0:3]).([#8:4]-[NH;D2;+0:5]-[C;D1;H3:6])"

  def get_actions(self, mols):
    rxn = RetroReaction(
        mol=mols[0],
        smarts=smarts,
    )
    return [rxn], [0.99]

  action_patcher = patch(
      "aizynthfinder.context.policy.ExpansionPolicy.get_actions",
      new=get_actions)
  action_patcher.start()
  yield
  action_patcher.stop()


@pytest.fixture
def mock_filter_policy():

  def feasibility(self, reaction):
    return False, 0.01

  action_patcher = patch(
      "aizynthfinder.context.policy.FilterPolicy.feasibility", new=feasibility)
  action_patcher.start()
  yield
  action_patcher.stop()


@pytest.fixture
def mocked_stock(tmpdir):

  def wrapper(*inchi_keys):
    filename = str(tmpdir / "stock.txt")
    with open(filename, "w") as fileobj:
      fileobj.write("\n".join(inchi_keys))
    return filename

  return wrapper


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_simple_retrosynthesis(mock_load_policy, mock_expansion_policy,
                               mocked_stock):
  mock_load_policy("ExpansionPolicy")
  stock_path = mocked_stock("ULGZOBNLBUIMAA-UHFFFAOYSA-N",
                            "CPQCSJYYDADLCZ-UHFFFAOYSA-N")
  engine = RetrosynthesisEngine(("any", "any"), stock_path)
  target_smiles = "CCCCOc1ccc(CC(=O)N(C)O)cc1"
  routes = engine.find_routes(target_smiles)

  assert len(routes) == 2
  assert routes[0]["in_stock"] == ["CCCCOc1ccc(CC(=O)Cl)cc1", "CNO"]
  assert routes[0]["not_in_stock"] == []
  # Check structure of route dictionary
  assert routes[0]["tree"]["smiles"] == target_smiles
  assert len(routes[0]["tree"]["children"][0]["children"]) == 2
  assert (routes[0]["tree"]["children"][0]["children"][0]["smiles"] ==
          "CCCCOc1ccc(CC(=O)Cl)cc1")
  assert routes[0]["tree"]["children"][0]["children"][1]["smiles"] == "CNO"
  # Check a single score
  assert routes[0]["scores"]["state score"] == pytest.approx(0.998, abs=0.001)

  assert routes[1]["in_stock"] == []
  assert routes[1]["not_in_stock"] == ["CCCCOc1ccc(CC(=O)N(C)O)cc1"]
  assert routes[1]["tree"]["smiles"] == target_smiles
  assert "children" not in routes[1]["tree"]
  assert routes[1]["scores"]["state score"] == pytest.approx(0.049, abs=0.001)


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_retrosynthesis_with_filter(mock_load_policy, mock_expansion_policy,
                                    mock_filter_policy, mocked_stock, caplog):
  mock_load_policy("ExpansionPolicy", "FilterPolicy")
  stock_path = mocked_stock("ULGZOBNLBUIMAA-UHFFFAOYSA-N",
                            "CPQCSJYYDADLCZ-UHFFFAOYSA-N")
  engine = RetrosynthesisEngine(
      ("any", "any"), stock_path, filter_policy_path="any")
  target_smiles = "CCCCOc1ccc(CC(=O)N(C)O)cc1"

  routes = engine.find_routes(target_smiles)

  assert len(routes) == 1
  assert routes[0]["in_stock"] == []
  assert routes[0]["not_in_stock"] == ["CCCCOc1ccc(CC(=O)N(C)O)cc1"]
  assert routes[0]["tree"]["smiles"] == target_smiles
  assert "children" not in routes[0]["tree"]
  assert routes[0]["scores"]["state score"] == pytest.approx(0.049, abs=0.001)


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_settings_retrosynthesis(mock_load_policy, mocked_stock):
  mock_load_policy("ExpansionPolicy")
  stock_path = mocked_stock()
  engine = RetrosynthesisEngine(("any", "any"), stock_path)

  assert engine.finder_settings["iteration_limit"] == 100

  engine.finder_settings = {"iteration_limit": 500}

  assert engine.finder_settings["iteration_limit"] == 500


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_update_expansion_policy(caplog, mock_load_policy, mocked_stock):
  mock_load_policy("ExpansionPolicy")
  stock_path = mocked_stock()
  engine = RetrosynthesisEngine(("any", "any"), stock_path)

  with caplog.at_level(logging.INFO, logger="aizynthfinder"):
    engine.update_expansion_policy("any", "any")
    assert any("Selected as expansion policy" in record.message
               for record in caplog.records)


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_update_filter_policy(caplog, mock_load_policy, mocked_stock):
  mock_load_policy("ExpansionPolicy", "FilterPolicy")
  stock_path = mocked_stock()
  engine = RetrosynthesisEngine(("any", "any"), stock_path)

  with caplog.at_level(logging.INFO, logger="aizynthfinder"):
    engine.update_filter_policy("any")
    assert any("Selected as filter policy" in record.message
               for record in caplog.records)


@pytest.mark.skipif(NO_AIZYNTHFINDER, reason="AiZynthFinder is not installed")
def test_update_stock(caplog, mock_load_policy, mocked_stock):
  mock_load_policy("ExpansionPolicy")
  stock_path = mocked_stock()
  engine = RetrosynthesisEngine(("any", "any"), stock_path)

  with caplog.at_level(logging.INFO, logger="aizynthfinder"):
    engine.update_stock(stock_path)
    assert any(
        "Selected as stock" in record.message for record in caplog.records)
