from collections import Counter
import pandas as pd

class ConversionErrorLogger:
    """Logs failed SMILES↔IUPAC conversions with error analysis."""

    def __init__(self):
        self.errors = []

    def log_failure(self, input_str, output_str, error_type):
        """Records a failed conversion."""
        self.errors.append({
            "input": input_str,
            "output": output_str,
            "error_type": error_type
        })

    def generate_report(self):
        """Returns a DataFrame summarizing error types."""
        error_counts = Counter(e["error_type"] for e in self.errors)
        return pd.DataFrame(error_counts.items(), columns=["Error", "Count"])
