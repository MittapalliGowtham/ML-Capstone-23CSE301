# models/

This folder is **generated at runtime**, not shipped with the submission.

Running `notebooks/09_regression_final_analysis_and_conclusion.ipynb` writes:

    regression_final_model.joblib

That file is the complete fitted `Pipeline` (preprocessing + the tuned Random
Forest), so it can be applied to raw data with a single call:

```python
import joblib
model = joblib.load("models/regression_final_model.joblib")
predictions = model.predict(new_raw_dataframe)
```

There is no separate scaler or encoder to remember to apply — the pipeline
carries them.

**Why it is not included in the repository:** the tuned forest has 400
unpruned trees and serialises to roughly 233 MB, which is far too large for a
Git repository. It is fully reproducible in about 30 seconds by running
notebook 09, because `random_state=42` is fixed throughout.
