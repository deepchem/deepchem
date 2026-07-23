"""
Tests for molnet function
"""
import csv
import tempfile
import unittest
import shutil
import os
import pytest
import numpy as np

import deepchem as dc
from deepchem.molnet.run_benchmark import run_benchmark
try:
    import torch  # noqa
    has_pytorch = True
except:
    has_pytorch = False


class TestMolnet(unittest.TestCase):
    """
    Test basic function of molnet
    """

    def setUp(self):
        super(TestMolnet, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_dir = os.path.join(self.current_dir, 'test_data')
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

    def tearDown(self):
        super(TestMolnet, self).tearDown()
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_pdbbind_collections(self):
        """Test loading different PDBBind collections with mock data."""
        try:
            # Create mock data structure
            mock_data = {
                'PL': {
                    'refined': ('pdbbind_v2019_PL_refined', 'index/INDEX_refined_PL_data.2019'),
                    'general': ('pdbbind_v2019_other_PL', 'index/INDEX_general_PL_data.2019')
                },
                'PP': {
                    'refined': ('pdbbind_v2019_PP_refined', 'index/INDEX_refined_PP_data.2019'),
                    'general': ('pdbbind_v2019_other_PP', 'index/INDEX_general_PP_data.2019')
                },
                'PN': {
                    'refined': ('pdbbind_v2019_PN_refined', 'index/INDEX_refined_PN_data.2019'),
                    'general': ('pdbbind_v2019_other_PN', 'index/INDEX_general_PN_data.2019')
                }
            }

            for collection, sets in mock_data.items():
                for set_name, (folder_name, index_file) in sets.items():
                    print(f"\nSetting up {collection} {set_name} set...")
                    # Create mock folder structure
                    data_dir = os.path.join(self.test_data_dir, folder_name)
                    os.makedirs(data_dir, exist_ok=True)
                    
                    # Create mock PDB files
                    pdb_ids = ['1a2b', '3c4d']
                    for pdb in pdb_ids:
                        pdb_dir = os.path.join(data_dir, pdb)
                        os.makedirs(pdb_dir, exist_ok=True)
                        
                        # Create empty files based on collection type
                        if collection == 'PL':
                            open(os.path.join(pdb_dir, f"{pdb}_protein.pdb"), 'w').close()
                            open(os.path.join(pdb_dir, f"{pdb}_pocket.pdb"), 'w').close()
                            open(os.path.join(pdb_dir, f"{pdb}_ligand.sdf"), 'w').close()
                        elif collection == 'PP':
                            open(os.path.join(pdb_dir, f"{pdb}_protein1.pdb"), 'w').close()
                            open(os.path.join(pdb_dir, f"{pdb}_protein2.pdb"), 'w').close()
                        else:  # PN
                            open(os.path.join(pdb_dir, f"{pdb}_protein1.pdb"), 'w').close()
                            open(os.path.join(pdb_dir, f"{pdb}_nucleic.pdb"), 'w').close()
                    
                    # Create mock index file
                    index_path = os.path.join(data_dir, index_file)
                    os.makedirs(os.path.dirname(os.path.join(data_dir, index_file)), exist_ok=True)
                    with open(index_path, 'w') as f:
                        f.write("# Comment line\n")
                        for pdb in pdb_ids:
                            f.write(f"{pdb}  2.50  2019  -6.2  2.0e-06  10.1016/j.example  TEST-1\n")

                    # Create mock tar.gz file
                    import tarfile
                    tar_path = os.path.join(self.test_data_dir, f"{folder_name}.tar.gz")
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(data_dir, arcname=folder_name)

            # Test loading each collection type
            for collection in mock_data.keys():
                for set_name in mock_data[collection].keys():
                    # Skip core set for non-PL collections
                    if set_name == 'core' and collection != 'PL':
                        continue

                    print(f"\nTesting {collection} {set_name} set...")
                    featurizer = dc.feat.RdkitGridFeaturizer()
                    try:
                        tasks, datasets, transformers = dc.molnet.load_pdbbind(
                            featurizer=featurizer,
                            splitter='random',
                            set_name=set_name,
                            collection=collection,
                            data_dir=self.test_data_dir)
                        
                        # Basic validation
                        self.assertEqual(len(tasks), 1)
                        self.assertEqual(tasks[0], '-logKd/Ki')
                        self.assertEqual(len(datasets), 3)  # train, valid, test split
                        for dataset in datasets:
                            self.assertIsInstance(dataset, dc.data.Dataset)
                            self.assertGreater(len(dataset), 0)
                    except Exception as e:
                        print(f"Error testing {collection} {set_name} set: {str(e)}")
                        raise
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def test_pdbbind_collections_simple(self):
        """Test loading different PDBBind collections with a simple mock dataset."""
        try:
            # Create a simple mock dataset for PL collection
            folder_name = 'pdbbind_v2019_PL_refined'
            data_dir = os.path.join(self.test_data_dir, folder_name)
            os.makedirs(data_dir, exist_ok=True)

            # Create mock PDB files
            pdb_id = '1a2b'
            pdb_dir = os.path.join(data_dir, pdb_id)
            os.makedirs(pdb_dir, exist_ok=True)
            
            # Create empty files with minimal valid content
            with open(os.path.join(pdb_dir, f"{pdb_id}_protein.pdb"), 'w') as f:
                f.write("ATOM      1  N   ASP A  30      31.904  -0.904  -0.904  1.00  0.00           N  \n")
                f.write("END\n")
            
            with open(os.path.join(pdb_dir, f"{pdb_id}_pocket.pdb"), 'w') as f:
                f.write("ATOM      1  N   ASP A  30      31.904  -0.904  -0.904  1.00  0.00           N  \n")
                f.write("END\n")
            
            with open(os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf"), 'w') as f:
                f.write("\n")
                f.write("  1  0  0  0  0  0  0  0  0  0999 V2000\n")
                f.write("    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n")
                f.write("M  END\n")
                f.write("$$$$\n")

            # Create mock index file
            os.makedirs(os.path.join(data_dir, 'index'), exist_ok=True)
            index_path = os.path.join(data_dir, 'index', 'INDEX_refined_PL_data.2019')
            with open(index_path, 'w') as f:
                f.write("# Comment line\n")
                f.write(f"{pdb_id}  2.50  2019  -6.2  2.0e-06  10.1016/j.example  TEST-1\n")

            # Create mock tar.gz file
            import tarfile
            tar_path = os.path.join(self.test_data_dir, f"{folder_name}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(data_dir, arcname=folder_name)

            # Test loading
            print("\nCreated test files:")
            print(f"Data dir: {data_dir}")
            print(f"Index file: {index_path}")
            print(f"Tar file: {tar_path}")

            featurizer = dc.feat.RdkitGridFeaturizer()
            tasks, datasets, transformers = dc.molnet.load_pdbbind(
                featurizer=featurizer,
                splitter='random',
                set_name='refined',
                collection='PL',
                data_dir=self.test_data_dir)

            # Basic validation
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0], '-logKd/Ki')
            self.assertEqual(len(datasets), 3)  # train, valid, test split
        except Exception as e:
            print(f"\nTest failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @pytest.mark.slow
    @pytest.mark.tensorflow
    def test_delaney_graphconvreg(self):
        """Tests molnet benchmarking on delaney with graphconvreg."""
        datasets = ['delaney']
        model = 'graphconvreg'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'valid'
            assert float(lastrow[-3]) > 0.65
        os.remove(os.path.join(out_path, 'results.csv'))

    @pytest.mark.slow
    @pytest.mark.torch
    def test_qm7_multitask(self):
        """Tests molnet benchmarking on qm7 with multitask network."""
        datasets = ['qm7']
        model = 'tf_regression_ft'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'valid'
            # TODO For this dataset and model, the R2-scores are less than 0.3.
            # This has to be improved.
            # See: https://github.com/deepchem/deepchem/issues/2776
            assert float(lastrow[-3]) > 0.15
        os.remove(os.path.join(out_path, 'results.csv'))

    @pytest.mark.torch
    def test_clintox_multitask(self):
        """Tests molnet benchmarking on clintox with multitask network."""
        datasets = ['clintox']
        model = 'tf'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path,
                      test=True)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'test'
            assert float(lastrow[-3]) > 0.7
        os.remove(os.path.join(out_path, 'results.csv'))

    @pytest.mark.slow
    def test_pdbbind_collections_original(self):
        """Tests loading different PDBBind collections."""
        featurizer = dc.feat.RdkitGridFeaturizer()
        collections = ['protein_ligand', 'protein_protein', 'protein_nucleic']
        set_names = {
            'protein_ligand': ['refined', 'general', 'core'],
            'protein_protein': ['refined', 'general'],
            'protein_nucleic': ['refined', 'general']
        }

        for collection in collections:
            for set_name in set_names[collection]:
                # Load dataset
                tasks, datasets, transformers = dc.molnet.load_pdbbind(
                    featurizer=featurizer,
                    splitter='random',
                    set_name=set_name,
                    collection_type=collection)
                
                # Basic validation
                assert len(tasks) == 1
                assert tasks[0] == '-logKd/Ki'
                assert len(datasets) == 3  # train, valid, test split
                for dataset in datasets:
                    assert isinstance(dataset, dc.data.Dataset)
                    assert len(dataset) > 0
