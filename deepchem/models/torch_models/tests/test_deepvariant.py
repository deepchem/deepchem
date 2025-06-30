import pytest
import numpy as np
import tempfile
import os
import shutil

try:
    import torch
    import pysam
    has_torch_and_pysam = True
except ModuleNotFoundError:
    has_torch_and_pysam = False
    pass


@pytest.mark.torch
def test_deepvariant_overfit():
    """
    Test the DeepVariant model's ability to overfit a small dataset
    """
    if not has_torch_and_pysam:
        pytest.skip("PyTorch or pysam not installed")

    from deepchem.models.torch_models.deepvariant import DeepChemVariant
    from deepchem.data import NumpyDataset

    # Create a very small dataset for overfitting
    window_size = 224  # Must be divisible by 32
    channels = 6
    height = 100
    n_samples = 3
    n_classes = 3  # ref, het, hom-alt
    
    np.random.seed(123)
    
    # Generate simple pileup images with distinct patterns
    X = np.zeros((n_samples, channels, height, window_size), dtype=np.float32)
    
    # Sample 1: stronger signal in first half (ref-like)
    X[0, 0, :, :window_size//2] = 0.8  
    
    # Sample 2: stronger signal in middle (het-like)
    X[1, 1, :, window_size//4:3*window_size//4] = 0.7
    
    # Sample 3: stronger signal in second half (hom-alt-like)
    X[2, 2, :, window_size//2:] = 0.9
    
    # Add some random noise
    X += np.random.randn(*X.shape) * 0.1
    
    # Simple labels: 0=ref, 1=het, 2=hom-alt
    y = np.array([0, 1, 2], dtype=np.int64)
    
    dataset = NumpyDataset(X=X, y=y.reshape(-1, 1))
    
    # Create model with high learning rate for faster overfitting
    model_dir = tempfile.mkdtemp()
    try:
        dv = DeepChemVariant(
            window_size=window_size,
            height=height,
            channels=channels,
            multiprocessing=False
        )
        
        # Set a higher learning rate and smaller width multiplier
        # for faster training on this tiny dataset
        from deepchem.models.torch_models import MobileNetV2Model
        dv.model = MobileNetV2Model(
            n_tasks=1,
            in_channels=channels,
            input_size=window_size,
            mode='classification',
            n_classes=n_classes,
            learning_rate=0.01,
            width_mult=0.5,
            model_dir=model_dir
        )
        
        # Train for enough epochs to overfit
        dv.model.fit(dataset, nb_epoch=100)
        
        # Get predictions
        pred = dv.model.predict(dataset)
        pred_labels = np.argmax(pred, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(pred_labels == y)
        
        # With such a small dataset and many epochs, we expect near-perfect accuracy
        assert accuracy > 0.9, f"Failed to overfit tiny dataset. Accuracy: {accuracy}"
        
    finally:
        shutil.rmtree(model_dir)


@pytest.mark.torch
def test_deepvariant_vcf_output():
    """
    Test that the DeepVariant model produces a valid VCF file
    """
    if not has_torch_and_pysam:
        pytest.skip("PyTorch or pysam not installed")
    
    from deepchem.models.torch_models.deepvariant import  _DeepChemVariant
    import tempfile
    
    # Create a temporary directory for output files
    temp_dir = tempfile.mkdtemp()
    output_vcf = os.path.join(temp_dir, "test_variants.vcf")
    
    try:
        # Create synthetic candidate variants
        candidates = np.array([
            ["chr1", 1000, "A", "G", 15, 30],
            ["chr1", 2000, "C", "T", 20, 40],
            ["chr2", 1500, "G", "A", 10, 20]
        ], dtype=object)
        
        # Create synthetic genotype probabilities (ref, het, hom-alt)
        genotype_probs = np.array([
            [0.01, 0.98, 0.01],  # Strong het call
            [0.01, 0.01, 0.98],  # Strong hom-alt call
            [0.98, 0.01, 0.01]   # Strong ref call (should be filtered)
        ])
        
        # Create a dummy FASTA file for testing
        fasta_file = os.path.join(temp_dir, "test.fa")
        with open(fasta_file, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 5000 + "\n")
            f.write(">chr2\n")
            f.write("C" * 5000 + "\n")
        
        # Create a BAI index file (empty, just for existence check)
        with open(fasta_file + ".fai", "w") as f:
            f.write("chr1\t5000\t6\t5000\t5001\n")
            f.write("chr2\t5000\t5012\t5000\t5001\n")
        
        # Create an instance of _DeepVariantHelper to use its methods
        helper = _DeepChemVariant()
        
        # Write VCF using helper instance
        helper.write_vcf(
            output_path=output_vcf,
            candidates=candidates,
            genotype_probs=genotype_probs,
            fasta_file=fasta_file,
            sample_name="TestSample"
        )
        
        # Check if VCF file was created
        assert os.path.exists(output_vcf), "VCF file was not created"
        
        # Read VCF file content to verify it's properly formatted
        with open(output_vcf, "r") as f:
            vcf_content = f.read()
        
        # Check VCF header
        assert "##fileformat=VCFv4.2" in vcf_content, "Missing VCF version header"
        assert "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTestSample" in vcf_content, "Missing VCF column headers"
        
        # Check contig headers
        assert "##contig=<ID=chr1" in vcf_content, "Missing contig header for chr1"
        assert "##contig=<ID=chr2" in vcf_content, "Missing contig header for chr2"
        
        # Check variant entries (should only see het and hom variants, not ref)
        assert "chr1\t1001\t.\tA\tG\t" in vcf_content, "Missing het variant"
        assert "chr1\t2001\t.\tC\tT\t" in vcf_content, "Missing hom variant"
        assert "0/1:" in vcf_content, "Missing heterozygous genotype"
        assert "1/1:" in vcf_content, "Missing homozygous genotype"
        
        # Verify ref variant was filtered out (it should not be in the VCF)
        assert "chr2\t1501\t" not in vcf_content, "Reference variant should be filtered out"
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
