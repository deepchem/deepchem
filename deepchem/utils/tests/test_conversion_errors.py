import pandas as pd
from deepchem.utils.conversion_errors import ConversionErrorLogger

def test_log_and_generate_report():
    logger = ConversionErrorLogger()
    
    # Simulate logging of failures
    logger.log_failure("C1=CC=CC=C1", "", "EmptyOutput")
    logger.log_failure("O=C=O", "", "EmptyOutput")
    logger.log_failure("CCO", "Invalid", "InvalidOutput")

    # Generate the report
    report = logger.generate_report()

    # Assertions to verify correct error logging
    assert isinstance(report, pd.DataFrame)
    assert "Error" in report.columns
    assert "Count" in report.columns
    assert report.loc[report["Error"] == "EmptyOutput", "Count"].values[0] == 2
    assert report.loc[report["Error"] == "InvalidOutput", "Count"].values[0] == 1
