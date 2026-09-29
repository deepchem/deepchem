import unittest
import os
import platform
import tempfile
import unittest.mock as mock
from deepchem.utils import sequence_utils as seq_utils

IS_WINDOWS = platform.system() == 'Windows'


class TestFileTypeValidation(unittest.TestCase):
    """File-type validation in hhsearch/hhblits, without needing hhsuite.

    The tests above are skipped on Windows because hhsuite is unavailable, and
    they exercise the alignment end to end. These check the decision that is
    made *before* any external process is started, so they run everywhere and
    need no database. system_call is stubbed to capture the command instead of
    shelling out.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.data_dir = self._dir.name

    def _query(self, name):
        path = os.path.join(self._dir.name, name)
        with open(path, 'w') as f:
            f.write('>seq0\nMKV\n')
        return path

    def _command_for(self, name, func='hhsearch'):
        """Return the command the function would have run, or raise."""
        captured = []
        with mock.patch.object(seq_utils, 'system_call', captured.append):
            getattr(seq_utils, func)(self._query(name),
                                     database='db',
                                     data_dir=self.data_dir)
        return captured[0]

    def test_fasta_queries_get_the_first_model_only_flag(self):
        # The '.fasta'/'.fas' branch exists to restrict the search to the first
        # model. Before the fix its condition was always true, so this flag was
        # never added to any command.
        for name in ('q.fasta', 'q.fas'):
            with self.subTest(name=name):
                self.assertIn('-M first', self._command_for(name))

    def test_alignment_input_extensions_do_not_get_that_flag(self):
        for name in ('q.a3m', 'q.a2m', 'q.hmm'):
            with self.subTest(name=name):
                self.assertNotIn('-M first', self._command_for(name))

    def test_the_two_input_families_are_handled_differently(self):
        # Both branches used to build the identical string, because the second
        # always ran and overwrote the first. The query path differs by
        # construction, so normalise it away and assert the *only* remaining
        # difference is the flag the first branch exists to add.
        from_fasta = self._command_for('q.fasta').replace('q.fasta', 'QUERY')
        from_a3m = self._command_for('q.a3m').replace('q.a3m', 'QUERY')
        self.assertEqual(from_fasta, from_a3m + ' -M first')

    def test_unsupported_extensions_are_refused(self):
        for name in ('q.txt', 'q.csv', 'q.pdb', 'q'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    self._command_for(name)
                self.assertIn('Unsupported file type', str(ctx.exception))

    def test_both_entry_points_validate(self):
        for func in ('hhsearch', 'hhblits'):
            with self.subTest(func=func):
                with self.assertRaises(ValueError):
                    self._command_for('q.txt', func=func)
                self.assertIn('-M first', self._command_for('q.fasta',
                                                            func=func))

    def test_an_empty_data_dir_raises_value_error(self):
        # This used to call logging.raiseExceptions, which is the bool True,
        # so the intended message surfaced as TypeError: 'bool' object is not
        # callable.
        with self.assertRaises(ValueError) as ctx:
            with mock.patch.object(seq_utils, 'system_call'):
                seq_utils.hhsearch(self._query('q.fasta'),
                                   database='db',
                                   data_dir='')
        self.assertIn('requires a database', str(ctx.exception))


@unittest.skipIf(IS_WINDOWS,
                 "Skip test on Windows")  # hhsuite does not run on windows
class TestSeq(unittest.TestCase):
    """
    Tests sequence handling utilities.
    """

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.dataset_file = os.path.join(current_dir, 'assets/example.fasta')
        self.database_name = 'example_db'
        self.data_dir = os.path.join(current_dir, 'assets')

    def test_hhsearch(self):
        seq_utils.hhsearch(self.dataset_file,
                           database=self.database_name,
                           data_dir=self.data_dir)
        results_file = os.path.join(self.data_dir, 'results.a3m')
        hhr_file = os.path.join(self.data_dir, 'example.hhr')
        with open(results_file, 'r') as f:
            resultsline = next(f)
        with open(hhr_file, 'r') as g:
            hhrline = next(g)

        assert hhrline[0:5] == 'Query'
        assert resultsline[0:5] == '>seq0'
        os.remove(results_file)
        os.remove(hhr_file)

    def test_hhblits(self):
        seq_utils.hhsearch(self.dataset_file,
                           database=self.database_name,
                           data_dir=self.data_dir)
        results_file = os.path.join(self.data_dir, 'results.a3m')
        hhr_file = os.path.join(self.data_dir, 'example.hhr')
        with open(results_file, 'r') as f:
            resultsline = next(f)
        with open(hhr_file, 'r') as g:
            hhrline = next(g)

        assert hhrline[0:5] == 'Query'
        assert resultsline[0:5] == '>seq0'
        os.remove(results_file)
        os.remove(hhr_file)

    def test_MSA_to_dataset(self):
        seq_utils.hhsearch(self.dataset_file,
                           database=self.database_name,
                           data_dir=self.data_dir)
        results_file = os.path.join(self.data_dir, 'results.a3m')
        msa_path = results_file
        dataset = seq_utils.MSA_to_dataset(msa_path)
        print(dataset.ids[0])
        print(dataset.X)
        assert dataset.ids[0] == 'seq0'
        assert dataset.ids[1] == 'seq1'
        bool_arr = dataset.X[0] == ['X', 'Y']
        assert bool_arr.all()
        os.remove(results_file)
