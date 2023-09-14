
# Model training results
    
Overall decision: **PASSED**

## Summary of checks

### Model accuracy on training dataset

Description: Checks how well the model performs on the training dataset.
The model accuracy on the training set should be at least 0.9.

Result: **PASSED**

- min. accuracy: 0.9
- actual accuracy: 0.9912087912087912

### Model accuracy on evaluation dataset

Description: Checks how well the model performs on the evaluation dataset.
The model accuracy on the evaluation set should be at least 0.9.

Result: **PASSED**

- min. accuracy: 0.9
- actual accuracy: 0.956140350877193

### Data integrity checks 

Description: A set of data quality checks that verify whether the input data is
valid and can be used for training (no duplicate samples, no missing values or
problems with string or categorical features, no significant outliers, no
inconsistent labels, etc.).

Results: **PASSED**

- passed checks: 9
- failed checks: 0  []
- checks with warnings: 0  []
- skipped checks: 2

### Train-test data drift checks 

Description: Compares the training and evaluation datasets to verify that their
distributions are similar and there is no potential data leakage that may
contaminate your model or perceived results.

Results: **PASSED**

- passed checks: 9
- failed checks: 0  []
- checks with warnings: 0  []
- skipped checks: 3

### Model evaluation checks 

Description: Runs a set of checks to evaluate the model performance, detect
overfitting, and verify that the model is not biased.

Results: **PASSED**

- passed checks: 2
- failed checks: 0  []
- checks with warnings: 0  []
- skipped checks: 3

### Train-test model comparison checks 

Description: Runs a set of checks to compare the model performance between the
test dataset and the evaluation dataset.

Results: **PASSED**

- passed checks: 3
- failed checks: 0  []
- checks with warnings: 1  [Unused Features]
- skipped checks: 2

## Other results

- [experiment tracker run](http://localhost:8185/#/experiments/0/runs/aa90a969a6bc4e93bafbae92a8d056c3)

