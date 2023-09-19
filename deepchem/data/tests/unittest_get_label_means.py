import numpy as np
import pandas as pd
import unittest

def get_label_means(metadata_df) -> pd.DataFrame:
        """Return pandas series of label means."""
        label_means = np.mean(metadata_df['y'],axis=0)
        return label_means

class TestMeanCalculator(unittest.TestCase):

    def test_get_label_means(self):
        """Test the get_label_means() function."""

        # Create a metadata_df DataFrame with some sample data.
        metadata_df = pd.DataFrame({'y': [1, 2, 3, 4, 5]})

        # Call the get_label_means() function on the metadata_df DataFrame.
        label_means = get_label_means(metadata_df)

        # Assert that the returned value is equal to the expected value.
        expected_label_means = 3.0
        self.assertEqual(label_means, expected_label_means)

if __name__ == '__main__':
    unittest.main()

