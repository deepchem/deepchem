import re
import numpy as np
from collections import defaultdict

from deepchem.utils.typing import PymatgenComposition
from deepchem.feat import MaterialCompositionFeaturizer


elements_tl = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
 'Hg', 'Tl', 'Pb', 'Bi', 'Ac','Th', 'Pa', 'U', 'Np', 'Pu']

formulare = re.compile(r'([A-Z][a-z]*)(\d*\.*\d*)')


class CompositionFeaturizer(MaterialCompositionFeaturizer):
  """
  Fixed size vector containing raw elemental compositions in the compound.

  Returns a vector containing fractional compositions of each element
  in the compound.

  This featurizer requires the optional dependency pymatgen.

  References
  ----------
  .. [1] Jha, D., Ward, L., Paul, A. et al. Sci Rep 8, 17593 (2018).
     https://doi.org/10.1038/s41598-018-35934-y

  Examples
  --------
  >>> import pymatgen as mg
  >>> comp = mg.Composition("Fe2O3")
  >>> featurizer = CompositionFeaturizer()
  >>> features = featurizer.featurize([comp])
  """

  def get_fractions(self, comp):
    if all(e in elements_tl for e in comp):
        return np.array([comp[e] if e in comp else 0 for e in elements_tl], np.float32)
    else:   return None

  def parse_fractions(self, form):
    while '/' in form:
        di = form.index('/')
        num1 = [x for x in re.findall(r'\d*\.*\d*', form[:di]) if x != ''][-1]
        num2 = [x for x in re.findall(r'\d*\.*\d*', form[di + 1:]) if x != ''][0]
        fract = '%.3f' % (float(num1) / float(num2))
        form = form[:di - len(num1)] + fract + form[di + len(num2) + 1:]
    return form

  def parse_formula(self, formula):
    stack = []
    curr_str = ''
    i = 0
    res = defaultdict(int)
    formula = formula.replace('-', '').replace('@',
                                               '').replace(' ', '').replace('[', '(').replace(']', ')').replace('{',
                                                                                                                '(').replace(
        '}',
        ')').replace('@', '').replace('x', '').replace(' ', '')

    def parse_simple_formula(x):
        x = self.parse_fractions(x)
        pairs = formulare.findall(x)
        length = sum((len(p[0]) + len(p[1]) for p in pairs))
        assert length == len(x)
        formula_dict = defaultdict(int)
        for el, sub in pairs:
            formula_dict[el] += float(sub) if sub else 1
        return formula_dict

    while i < len(formula):
        if formula[i] not in ['(', ')'] and not stack:
            curr_str = ''
            while i < len(formula) and formula[i] != '(':
                curr_str += formula[i]
                i += 1
            fract = re.findall(r'\d*\.*\d*', curr_str)[0]
            curr_str = curr_str[len(fract):]
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            temp_res = parse_simple_formula(curr_str)
            for k, v in temp_res.items():
                res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
        elif formula[i] not in [')']:
            stack.append(formula[i])
            i += 1
        else:
            i += 1
            fract = re.findall(r'\d*\.*\d*', formula[i:])[0]
            i = i + len(fract)
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            curr_str = ''
            while stack[-1] != '(':
                curr_str += stack.pop()
            stack.pop()
            curr_str = curr_str[::-1]
            fract1 = re.findall(r'\d*\.*\d*', curr_str)[0]
            if not len(fract1):
                fract *= 1.
            else:
                fract *= float(fract1)
            curr_str = curr_str[len(fract1):]
            temp_res = parse_simple_formula(curr_str)
            for k, v in temp_res.items():
                temp_res[k] *= fract
            if not stack:
                for k, v in temp_res.items():
                    res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
            else:
                for i, v in temp_res.items():
                    stack.append(i)
                    stack.append(v)
    if any([e for e in res if e in ['T', 'D', 'G', 'M', 'Q']]):
        print (formula, res)
    sum_nums = 1. * sum(res.values())
    for k in res: res[k] = 1. * res[k] / sum_nums
    return res

  def _featurize(self, composition: PymatgenComposition) -> np.ndarray:
    """
    Calculate composition vector from composition.

    Parameters
    ----------
    composition: pymatgen.Composition object
      Composition object.

    Returns
    -------
    feats: np.ndarray
      Vector of fractional compositions of each element.
    """
    try:
      pretty_comp = composition.reduced_formula
      feats = self.get_fractions(self.parse_formula(pretty_comp))
    except:
      feats = []

    return np.array(feats)
