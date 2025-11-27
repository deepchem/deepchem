import pytest
import numpy as np
import tempfile
import os
import deepchem as dc


@pytest.mark.torch
def test_deepvariant_mobilenetv2_vcf():
    """
    Test the DeepVariant model's basic functionality
    """
    from deepchem.data import NumpyDataset
    from deepchem.models.torch_models import DeepVariant

    # Set parameters
    input_size = 96  # Must be divisible by 32
    in_channels = 6
    n_classes = 3
    batch_size = 2

    # Create test pileup images (simulating genomic pileup data)
    pileup_imgs = np.zeros((batch_size, in_channels, input_size, input_size),
                           dtype=np.float32)
    # Add some signal in the center region
    pileup_imgs[:, 0, 40:60, 40:60] = 0.8  # Base identity channel
    pileup_imgs[:, 1, 40:60, 40:60] = 0.7  # Base quality channel
    pileup_imgs[:, 2, 40:60, 40:60] = 0.9  # Mapping quality channel

    # Create variant information for each sample
    variant_info = [{
        'chrom': 'chr1',
        'pos': 1000,
        'ref_allele': 'A',
        'alt_allele': 'T',
        'depth': 30
    }, {
        'chrom': 'chr2',
        'pos': 2000,
        'ref_allele': 'G',
        'alt_allele': 'C',
        'depth': 25
    }]

    # Create training labels (homozygous ref [1,0,0], heterozygous [0,1,0])
    train_labels = np.array([0, 1]).astype(np.int64)  # 0 for hom-ref, 1 for het

    # Create datasets
    train_dataset = NumpyDataset(X=pileup_imgs,
                                 y=np.reshape(train_labels, (-1, 1)))
    test_dataset = NumpyDataset(X=pileup_imgs, y=variant_info)

    # Initialize model with temporary directory
    model_dir = tempfile.mkdtemp()
    model = DeepVariant(
        n_tasks=1,  # genotype classification task
        n_classes=n_classes,
        in_channels=in_channels,
        input_size=input_size,
        model_dir=model_dir,
        learning_rate=0.01)

    # Test model fitting
    model.fit(train_dataset, nb_epoch=1)

    # Test variant calling functionality
    vcf_path = os.path.join(model_dir, "variants.vcf")
    output_path = model.call_variants(test_dataset, vcf_path, "sample1")

    # Check if VCF file was created
    assert os.path.exists(output_path), "VCF file was not created"

    # Read VCF file to check content
    with open(output_path, 'r') as f:
        vcf_content = f.read()

    # Check VCF header
    assert "##fileformat=VCFv4.2" in vcf_content
    assert "##source=DeepVariant" in vcf_content

    # Check if sample name is in the VCF
    assert "sample1" in vcf_content

    # Check if variant records are present
    assert "chr1\t1000\t.\tA\tT\t" in vcf_content
    assert "chr2\t2000\t.\tG\tC\t" in vcf_content


@pytest.mark.torch
def test_deepvariant_training():
    """
    Test the DeepVariant model's training functionality
    """
    from deepchem.data import NumpyDataset
    from deepchem.models.torch_models import DeepVariant

    # Set parameters
    input_size = 64  # Smaller size for faster testing (must be divisible by 32)
    in_channels = 6
    n_classes = 3
    batch_size = 4

    # Set random seed for reproducibility
    np.random.seed(123)

    # Create test pileup images (simulating genomic pileup data)
    pileup_imgs = np.random.rand(batch_size, in_channels, input_size,
                                 input_size).astype(np.float32)

    # Create training labels (0=hom-ref, 1=het, 2=hom-alt)
    train_labels = np.array([0, 1, 2, 0]).astype(np.int64)

    # Create dataset
    dataset = NumpyDataset(X=pileup_imgs, y=np.reshape(train_labels, (-1, 1)))

    # Split into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset,
                                                            frac_train=0.75,
                                                            seed=123)

    # Initialize model with temporary directory
    model_dir = tempfile.mkdtemp()
    model = DeepVariant(n_tasks=1,
                        in_channels=in_channels,
                        input_size=input_size,
                        model_dir=model_dir,
                        learning_rate=0.005)

    # Test basic model properties
    assert model.n_classes == 3
    assert model.in_channels == in_channels

    # Record initial predictions before training
    initial_preds = model.predict(test_dataset)

    # Train the model for several epochs
    losses = []
    for i in range(5):  # Train for 5 epochs
        loss = model.fit(train_dataset, nb_epoch=1)
        losses.append(loss)

    # Verify that loss decreased during training
    assert losses[0] > losses[-1], "Training loss did not decrease"

    # Get final predictions
    final_preds = model.predict(test_dataset)

    # Verify prediction shape
    assert final_preds.shape[1] == n_classes, \
        f"Expected {n_classes} outputs, got {final_preds.shape[1]}"

    # Verify that model predictions changed after training
    pred_diff = np.sum(np.abs(final_preds - initial_preds))
    assert pred_diff > 0.1, "Model predictions did not change after training"

    # Check that predictions sum to approximately 1
    # (They should be probabilities for each genotype)
    pred_sums = np.sum(final_preds, axis=1)
    print(pred_sums)
    assert np.allclose(pred_sums, 1.0, atol=1e-5), "Predictions do not sum to 1"


@pytest.mark.torch
def test_deepvariant_save_reload():
    """
    Test saving and restoring the DeepVariant model
    """
    from deepchem.data import NumpyDataset
    from deepchem.models.torch_models import DeepVariant

    # Set parameters
    input_size = 96  # Must be divisible by 32
    in_channels = 6
    n_classes = 3
    batch_size = 2

    # Create test pileup images (simulating genomic pileup data)
    pileup_imgs = np.zeros((batch_size, in_channels, input_size, input_size),
                           dtype=np.float32)
    # Add some signal in the center region
    pileup_imgs[:, 0, 40:60, 40:60] = 0.8  # Base identity channel
    pileup_imgs[:, 1, 40:60, 40:60] = 0.7  # Base quality channel
    pileup_imgs[:, 2, 40:60, 40:60] = 0.9  # Mapping quality channel

    # Create variant information for each sample
    variant_info = [{
        'chrom': 'chr1',
        'pos': 1000,
        'ref_allele': 'A',
        'alt_allele': 'T',
        'depth': 30
    }, {
        'chrom': 'chr2',
        'pos': 2000,
        'ref_allele': 'G',
        'alt_allele': 'C',
        'depth': 25
    }]

    # Create training labels (homozygous ref [1,0,0], heterozygous [0,1,0])
    train_labels = np.array([0, 1]).astype(np.int64)  # 0 for hom-ref, 1 for het

    # Create datasets
    train_dataset = NumpyDataset(X=pileup_imgs,
                                 y=np.reshape(train_labels, (-1, 1)))
    test_dataset = NumpyDataset(X=pileup_imgs, y=variant_info)

    # Initialize model with temporary directory
    model_dir = tempfile.mkdtemp()
    model = DeepVariant(
        n_tasks=1,  # genotype classification task
        n_classes=n_classes,
        in_channels=in_channels,
        input_size=input_size,
        model_dir=model_dir,
        learning_rate=0.01)

    # Test model fitting
    model.fit(train_dataset, nb_epoch=1)

    original_preds = model.predict(test_dataset)
    # Save the model
    model.save()

    # Create a new model instance
    reloaded_model = DeepVariant(n_tasks=1,
                                 n_classes=3,
                                 in_channels=in_channels,
                                 input_size=input_size,
                                 model_dir=model_dir)

    # Restore the model
    reloaded_model.restore()

    # Get predictions from the restored model
    restored_preds = reloaded_model.predict(test_dataset)

    # Ensure predictions before and after restoring are identical
    np.testing.assert_allclose(
        original_preds,
        restored_preds,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Predictions changed after model restore")

    # Test that reloaded model can generate VCF
    vcf_path = os.path.join(model_dir, "test_variants.vcf")
    output_path = reloaded_model.call_variants(test_dataset, vcf_path,
                                               "test_sample")

    # Check that VCF file was created
    assert os.path.exists(
        output_path), "VCF file was not created by restored model"

    # Read a few lines from the VCF to verify it's valid
    with open(output_path, 'r') as f:
        header_line = None
        data_lines = []
        for line in f:
            if line.startswith('#CHROM'):
                header_line = line.strip()
            elif not line.startswith('#'):
                data_lines.append(line.strip())
                if len(data_lines) >= 2:
                    break

    # Verify VCF content
    assert header_line is not None, "VCF header line is missing"
    assert "test_sample" in header_line, "Sample name not found in VCF header"
    assert len(data_lines) == 2, "Expected 2 variant lines in VCF"
