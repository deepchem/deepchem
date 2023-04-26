from deepchem.data import Dataset


class VocabularyBuilder():
    """Abstract class for building a vocabulary from a dataset."""

    def build(self, dataset: Dataset):
        """Builds vocabulary from a dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to build vocabulary from.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, fname: str):
        """Loads vocabulary from the specified file

        Parameters
        ----------
        fname: str
            Path containing pre-build vocabulary.
        """
        raise NotImplementedError()

    def save(self, fname: str):
        """Dump vocabulary to the specified file.

        Parameters
        ----------
        fname: str
            A json file fname to save vocabulary.
        """
        raise NotImplementedError()

    def extend(self, dataset: Dataset):
        """Extends vocabulary from a dataset

        Parameters
        ----------
        dataset: Dataset
            dataset used for extending vocabulary
        """
        raise NotImplementedError()
