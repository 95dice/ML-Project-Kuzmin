# Double Descent in Linear and Ridge Regression

This project explores the **double descent phenomenon** in machine learning by simulating and visualizing how test and training errors evolve as model complexity (dimensionality `d`) increases. Both **synthetic data** and **real-world MNIST data** are used to demonstrate the effect under varying noise conditions, data density, and model types (Linear vs. Ridge regression).

---

## Project Structure

- `generateData(...)`: Function to generate synthetic datasets with configurable density and noise level.
- `computeMSEs(...)`: Runs experiments for different dimensionalities `d`, returning MSE metrics and condition number of the design matrix.
- `runExperiment(...)`: High-level function to configure and run an experiment, including plotting results.
- `RidgeLinearRegression`: Subclass of a course-provided `LinearRegression` model that adds L2 linear regularization.
- Uses the course's `courselib` package for model training, optimization, data splitting, and evaluation.

---

## Running Experiments

You can run an experiment directly from a script or Jupyter notebook:

```python
runExperiment(
    numberOfSamples=100,
    rangeForDim=range(50, 300),
    sigma=5,
    epochs=500,
    isDense=False,
    isMnistData=True,
    isRidge=False
)
```

This will:

- Train a model over a range of feature dimensions
- Compute and plot test/train MSE curves 
- Display condition number of the design matrix
- Show evidence (or lack thereof) of the double descent behavior

## Dependencies
Install required packages using pip:

```
pip install -r requirements.txt
```