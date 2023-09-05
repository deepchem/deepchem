import os
import deepchem as dc


def test_batch_coulomb_matrix_features():
    # Get Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_dir, 'assets/qm9_mini.sdf')
    TASKS = ["alpha", "homo"]
    loader = dc.data.SDFLoader(tasks=TASKS,
                               featurizer=dc.feat.CoulombMatrix(29),
                               sanitize=True)
    data = loader.create_dataset(dataset_file, shard_size=100)
    inputs = dc.utils.batch_utils.batch_coulomb_matrix_features(data.X)

    # Checks that all atoms exits in array
    assert inputs[0].shape == (149,)

    # Checks shape of gaussian distance
    assert inputs[1].shape == (1215, 100)

    # Checks all molecule membership exist
    for i in range(0, 21):
        if i not in inputs[2]:
            raise AssertionError("All molecules not in the matrix")

    # Check Distance Membership shape
    assert inputs[3].shape == (1215,)
    assert inputs[4].shape == (1215,)


def test_batch_elements():
    # Prepare Data
    inputs = [[i, i**2, i**3] for i in range(10)]
    # Run
    output = list(dc.utils.batch_utils.batch_elements(inputs, 3))
    assert output == [[[0, 0, 0], [1, 1, 1], [2, 4, 8]],
                      [[3, 9, 27], [4, 16, 64], [5, 25, 125]],
                      [[6, 36, 216], [7, 49, 343], [8, 64, 512]], [[9, 81,
                                                                    729]]]
