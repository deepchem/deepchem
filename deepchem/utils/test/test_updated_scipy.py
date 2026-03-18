import scipy.stats as st
import scipy
import unittest


class TestVoxelUtils(unittest.TestCase):

    def test_gibrat(self):
        """
        Test fix of function name 'gilbrat' to 'gibrat' of scipy.stats
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gilbrat.html
        """
        assert isinstance(st.gibrat, scipy.stats._continuous_distns.gibrat_gen)
