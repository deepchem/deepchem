# QM9 DTNN Example

## Expected Results
- Training R²: > 0.90
- Test R²: > 0.85

## Requirements
```bash
pip install deepchem tensorflow==2.x rdkit
```

## Troubleshooting

### Poor Performance (R² < 0.1)
1. Check TensorFlow version compatibility
2. Verify data loaded correctly
3. Ensure random seeds are set
4. Try adjusting learning rate

### Common Issues
- **NaN losses**: Reduce learning rate
- **No improvement**: Increase model capacity
- **Overfitting**: Add dropout or reduce complexity
