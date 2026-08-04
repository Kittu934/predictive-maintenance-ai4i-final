# Genuine Implementation Results

Dataset: AI4I 2020 (`ai4i2020.csv`)

- Rows: 10,000
- Columns: 14
- Missing values: 0
- Duplicate rows: 0
- Normal observations: 9,661 (96.61%)
- Machine failures: 339 (3.39%)
- Train/test split: 8,000 / 2,000, stratified
- Failure cases in test set: 68

## Predictive results

| Model         |   Accuracy |   Precision |   Recall |   F1-score |   AUC-ROC |   Decision threshold |
|:--------------|-----------:|------------:|---------:|-----------:|----------:|---------------------:|
| Random Forest |     0.9800 |      0.7188 |   0.6765 |     0.6970 |    0.9614 |               0.4340 |
| SVM           |     0.9710 |      0.5862 |   0.5000 |     0.5397 |    0.9550 |               0.3431 |
| LSTM          |     0.9710 |      0.5862 |   0.5000 |     0.5397 |    0.9581 |               0.9117 |

## Computational-efficiency results

| Model         |   Training time (s) |   Inference time for 2,000 cases (s) |   Peak memory increase (MB) |
|:--------------|--------------------:|-------------------------------------:|----------------------------:|
| Random Forest |              0.4018 |                               0.0463 |                      9.1050 |
| SVM           |              1.3755 |                               0.0389 |                     34.3980 |
| LSTM          |              7.6744 |                               0.0006 |                    261.9610 |

## Selected hyperparameters

- Random Forest: 200 trees, unrestricted depth, minimum leaf size 2, square-root feature sampling, balanced class weights.
- SVM: RBF kernel, C=100, gamma=scale, balanced class weights, probability estimation enabled.
- LSTM: PyTorch, one time step, 8 inputs, 16 hidden units, dropout 0.2, learning rate 0.001, batch size 64, 60 epochs and positive-class-weighted binary cross-entropy.
- Validation-optimised thresholds: Random Forest 0.4340; SVM 0.3431; LSTM 0.9117.

## Main conclusion

Random Forest gave the strongest overall trade-off. It had the highest F1-score and AUC-ROC, detected more failures than the other models, trained fastest and used the least additional memory. The LSTM result must be interpreted cautiously because AI4I does not contain genuine machine-specific temporal sequences.
