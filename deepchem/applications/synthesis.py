from typing import Tuple, Dict, Any, Union, List

_SettingsType = Dict[str, Union[int, float, str, bool]]


class RetrosynthesisEngine:
  """Retrosynthesis engine

  This holds an interface to the AiZynthFinder retrosynthesis library

  References
  ----------
  Genheden S, Thakkar A, Chadimova V, et al (2020) J. Cheminf. 12:70, 10.1186/s13321-020-00472-1

  Note
  ----
  This class requires AiZynthFinder to be installed.
  """

  _policy_name = "deepchem_policy"
  _stock_name = "deepchem_stock"

  def __init__(self,
               expansion_policy_paths: Tuple[str, str],
               stock_path: str,
               filter_policy_path: str = None) -> None:
    """Initialize the engine

    Parameters
    ----------
    expansion_policy_paths: tuple of str,
      the path to the expansion policy model and the corresponding template file
    stock_path: str
      the path to the file with the stock molecules as InChI keys
    filter_policy_path: str, optional
      the path to the filter policy model

    Raises
    ------
    ImportError
      if AiZynthFinder is not installed
    """
    try:
      from aizynthfinder.aizynthfinder import AiZynthFinder
      from aizynthfinder.chem import none_molecule
    except ModuleNotFoundError:
      raise ImportError("This class requires AiZynthFinder to be installed.")
    self._none_mol = none_molecule()
    conf = {
        "policy": {
            "files": {
                self._policy_name: expansion_policy_paths
            }
        },
        "stock": {
            "files": {
                self._stock_name: stock_path
            }
        }
    }
    if filter_policy_path:
      conf["filter"] = {"files": {self._policy_name: filter_policy_path}}
    self._finder = AiZynthFinder(configdict=conf)
    self._finder.expansion_policy.select(self._policy_name)
    self._finder.stock.select(self._stock_name)
    if filter_policy_path:
      self._finder.filter_policy.select(self._policy_name)
    self.routes = None

  @property
  def finder_settings(self) -> _SettingsType:
    """ Access the settings of the tree search

    Returns
    -------
    dict
        the settings as a dictionary
    """
    # TODO: this should be a function of the Configuration class.
    dict_ = {}
    for item in dir(self._finder.config):
      if item.startswith("_"):
        continue
      attr = getattr(self._finder.config, item)
      if isinstance(attr, (int, float, str, bool)):
        dict_[item] = attr
    return dict_

  @finder_settings.setter
  def finder_settings(self, value: _SettingsType) -> None:
    self._finder.config.update(**value)

  def find_routes(self, target: str) -> List[Dict[str, Any]]:
    """ Find routes for a given target molecule

    The method will return a dictionary with tree items
    1. Statistics on the search
    2. The top-ranked routes as list of dictionaries
    3. The top-ranked routes as a list of PIL images

    Parameters
    ----------
    target: str
      the SMILES string of the target molecule

    Returns
    -------
      the routes
    """
    self._finder.target_smiles = target
    self._finder.tree_search()
    self._finder.build_routes()
    self.routes = self._finder.routes
    return self._collect_results()

  def update_expansion_policy(self, model_path: str,
                              template_path: str) -> None:
    """ Update the expansion policy.

    This will load a new policy from disc.

    Parameters
    ----------
    model_path: str
      the path to the model file
    template_path: str
      the path to the template file
    """
    self._finder.expansion_policy.load(
        source=model_path, templatefile=template_path, key=self._policy_name)
    self._finder.expansion_policy.select(self._policy_name)
    self._finder.target_mol = self._none_mol
    self.routes = None

  def update_filter_policy(self, model_path: str) -> None:
    """ Update the filter policy.

    This will load a new policy from disc.

    Parameters
    ----------
    model_path: str
      the path to the model file
    """
    self._finder.filter_policy.load(source=model_path, key=self._policy_name)
    self._finder.filter_policy.select(self._policy_name)
    self._finder.target_mol = self._none_mol
    self.routes = None

  def update_stock(self, stock_path: str) -> None:
    """ Update the stock

    Parameters
    ----------
    stock_path: str
      the path to a file with stock molecules as InChI keys
    """
    self._finder.stock.load(stock_path, self._stock_name)
    self._finder.stock.select(self._stock_name)
    self._finder.target_mol = self._none_mol
    self.routes = None

  def _collect_results(self) -> List[Dict[str, Any]]:
    results = []
    routes = self._finder.routes
    for rt, dict_, scores, image in zip(routes.reaction_trees, routes.dicts,
                                        routes.all_scores, routes.images):
      in_stock = []
      not_in_stock = []
      for leaf in rt.leafs():
        if rt.in_stock(leaf):
          in_stock.append(leaf.smiles)
        else:
          not_in_stock.append(leaf.smiles)
      results.append({
          "tree": dict_,
          "scores": scores,
          "image": image,
          "in_stock": in_stock,
          "not_in_stock": not_in_stock
      })
    return results
