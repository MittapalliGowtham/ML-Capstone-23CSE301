# Machine Learning Capstone Project — Review 1

**Regression: Seoul Bike Sharing Demand · Classification Part A: Adult Census Income**

Course: **23CSE301 Machine Learning** — B.Tech. Computer Science and Engineering, III Year
Academic Year 2026–27 · Review 1 submission

---

## Team

| Role | Name | Roll number | Track ownership |
|---|---|---|---|
| Member 1 | *[Your name here]* | *[Roll no.]* | Regression track (Notebooks 01–09) |
| Member 2 | *[Your name here]* | *[Roll no.]* | Classification track (Notebooks 10–14) |
| Member 3 | *[Your name here]* | *[Roll no.]* | Shared utilities, viva prep, documentation (Notebook 15, `src/`) |

> **Fill these in before submitting.** The guidelines require that each member owns
> at least one track and can explain every line of code in it during the viva.
> Note that ownership means *primary* responsibility — the reviewer may ask any
> member about any part of the work, so everyone should read all 15 notebooks.

---

## 1. Project description

This project builds two complete, end-to-end machine-learning pipelines for two
**entirely separate** problems. The datasets are never merged, joined, or mixed;
each track has its own notebooks, train/test split, preprocessing pipeline,
metrics and conclusions.

**Track 1 — Regression.** Predict the number of public bicycles rented in Seoul
in a given hour, from calendar and weather conditions. Ten regression algorithms
are trained on one identical split, compared on R² / RMSE / MAE, two are tuned by
cross-validated search, and the final model is analysed for residual behaviour,
feature importance and failure modes.

**Track 2 — Classification (Part A).** Predict whether a US census respondent's
annual income exceeds \$50,000. Five classification algorithms are trained on one
identical stratified split and compared on accuracy, weighted F1, precision,
recall and confusion matrices.

The work is split across **15 focused notebooks** rather than one long file. Each
notebook owns one stage, runs top-to-bottom on its own, and hands its output to
the next through the `artifacts/` folder.

---

## 2. Headline results

### Regression — all ten algorithms (same test set, n = 1,693)

| Rank | Model | R² | RMSE | MAE |
|---:|---|---:|---:|---:|
| 1 | Random Forest Regressor | 0.9236 | 173.19 | 92.45 |
| 2 | Gradient Boosting Regressor | 0.8964 | 201.70 | 129.99 |
| 3 | Polynomial Regression (degree 2) | 0.8843 | 213.13 | 143.19 |
| 4 | Decision Tree Regressor | 0.8560 | 237.75 | 126.69 |
| 5 | KNN Regressor | 0.8513 | 241.66 | 146.98 |
| 6 | Lasso Regression | 0.7108 | 336.95 | 254.97 |
| 7 | Ridge Regression | 0.7100 | 337.40 | 255.69 |
| 8 | Linear Regression | 0.7100 | 337.42 | 255.71 |
| 9 | ElasticNet Regression | 0.6575 | 366.69 | 272.48 |
| 10 | Support Vector Regressor (SVR) | 0.3738 | 495.82 | 331.78 |

**After tuning** (cross-validated search on training data only):

| Model | Untuned R² | Tuned R² | Gain | Search |
|---|---:|---:|---:|---|
| SVR | 0.3738 | **0.9235** | **+0.5497** | GridSearchCV, 18 candidates |
| Random Forest | 0.9236 | 0.9298 | +0.0062 | RandomizedSearchCV, 20 candidates |

**5-fold cross-validation** of the two strongest models:

| Model | CV mean R² | CV std | Fold range | Test R² |
|---|---:|---:|---|---:|
| Random Forest (tuned) | 0.9433 | 0.0046 | 0.9388 – 0.9511 | 0.9298 |
| SVR (tuned) | 0.9228 | 0.0068 | 0.9138 – 0.9316 | 0.9235 |

**Final model: tuned Random Forest — R² 0.9298, RMSE 166.0 bikes, MAE 88.1 bikes**
(against a mean hourly demand of 720, i.e. a typical error of ~12%, and a 50.8%
reduction in RMSE versus the Linear Regression baseline).

### Classification Part A — all five algorithms (same stratified test set, n = 9,769)

| Rank | Model | Accuracy | Weighted F1 | Recall (>50K) |
|---:|---|---:|---:|---:|
| 1 | Logistic Regression | 0.8589 | **0.8538** | 0.6129 |
| 2 | Support Vector Classifier | **0.8591** | 0.8520 | 0.5838 |
| 3 | K-Nearest Neighbors | 0.8410 | 0.8385 | 0.6249 |
| 4 | Decision Tree | 0.8265 | 0.8258 | 0.6253 |
| 5 | Gaussian Naive Bayes | 0.6752 | 0.6983 | **0.9239** |
| — | *Baseline: always predict ≤50K* | *0.7607* | *0.6573* | *0.0000* |

> **No overall winner is declared for the classification track.** The top two are
> separated by 0.0002 in accuracy and **swap rank** depending on the metric;
> neither has been tuned or cross-validated. Gaussian Naive Bayes is last on
> accuracy yet finds the most high earners by a wide margin. Which model is
> "best" depends entirely on the cost of each error type. See Notebook 14 §12.7.

---

## 3. Datasets

### 3.1 Regression — Seoul Bike Sharing Demand

| | |
|---|---|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand |
| File | `dataset/SeoulBikeData.csv` |
| Rows | 8,760 (one per hour, 1 Dec 2017 – 30 Nov 2018) |
| Target | `Rented Bike Count` (continuous, 0–3,556) |
| Features | 13 — hour, temperature, humidity, wind, visibility, dew point, solar radiation, rainfall, snowfall, season, holiday, functioning day, date |
| Rows used | 8,465 after removing 295 non-operating hours (Notebook 04 §6.1) |

> **Encoding note:** the file is **CP-1252 / Latin-1**, not UTF-8, because the
> headers contain `°C`. It must be read with `encoding="cp1252"` or pandas raises
> a `UnicodeDecodeError`.

### 3.2 Classification — Adult / Census Income

| | |
|---|---|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/2/adult |
| Files | `dataset/adult.data`, `dataset/adult.test`, `dataset/adult.names` |
| Rows | 48,842 combined (32,561 + 16,281) |
| Target | `income` — `<=50K` (0) or `>50K` (1) |
| Features | 14 demographic and employment attributes |
| Class balance | 76.07% / 23.93% — ratio 3.18 : 1 |

> **Parsing notes:** the files have no header row, use `", "` as the separator,
> and encode missing values as `?`. All three must be handled explicitly
> (`header=None`, `skipinitialspace=True`, `na_values="?"`) or 6,465 missing
> values become invisible. `adult.test` additionally has a junk first line and a
> trailing full stop on every label.

### 3.3 Ethical note on the Adult dataset

This dataset contains `race`, `sex` and `native-country`, and a model trained on
it will learn the earnings disparities of 1994 American society, including those
produced by discrimination. **Predictive accuracy is not fairness** — a model can
be accurate precisely *because* it has reproduced a historical injustice.

This project uses the dataset as an academic benchmark only. **The resulting
models must not be used for consequential decisions about individuals** (lending,
hiring, insurance). The issue is addressed with measured evidence in Notebook 10
§1.7 and Notebook 14 §11.3.

---

## 4. Where to put the datasets

Place the four raw files in the `dataset/` folder exactly as shown:

```
ml-capstone-review1/
└── data/
    ├── SeoulBikeData.csv     <- from seoul_bike_sharing_demand.zip
    ├── adult.data            <- from adult.zip
    ├── adult.test            <- from adult.zip
    └── adult.names           <- from adult.zip (reference only, not loaded)
```

**If you downloaded the ZIP archives**, extract them into `dataset/`:

```bash
cd ml-capstone-review1
unzip -o seoul_bike_sharing_demand.zip -d data/
unzip -o adult.zip -d data/
```

**Verify the files are in place** before running anything:

```bash
ls -l data/
# Expect: SeoulBikeData.csv (~590 KB), adult.data (~3.8 MB), adult.test (~1.9 MB)
```

The notebooks locate this folder automatically by walking up from wherever they
are run, so you do not need to edit any paths. If a file is missing, Notebook 01
or 10 will fail immediately with a clear `FileNotFoundError` rather than
producing wrong results silently.

---

## 5. Project structure

```
ml-capstone-review1/
├── README.md                    This file
├── requirements.txt             Pinned Python dependencies
│
├── data/                        Raw datasets (see section 4)
│
├── src/
│   └── ml_utils.py              Shared helper functions used by all 15 notebooks
│
├── notebooks/                   15 notebooks, run in numerical order
│   ├── 01_regression_introduction_and_data_loading.ipynb
│   ├── 02_regression_data_quality_audit.ipynb
│   ├── 03_regression_eda.ipynb
│   ├── 04_regression_outlier_analysis.ipynb
│   ├── 05_regression_feature_engineering.ipynb
│   ├── 06_regression_preprocessing_and_baseline.ipynb
│   ├── 07_regression_all_10_models.ipynb
│   ├── 08_regression_tuning_and_cross_validation.ipynb
│   ├── 09_regression_final_analysis_and_conclusion.ipynb
│   ├── 10_classification_introduction_and_data_loading.ipynb
│   ├── 11_classification_data_quality_audit.ipynb
│   ├── 12_classification_eda.ipynb
│   ├── 13_classification_cleaning_feature_engineering_split.ipynb
│   ├── 14_classification_models_and_conclusion.ipynb
│   └── 15_viva_preparation_and_project_checklist.ipynb
│
├── artifacts/                   Generated: state passed between notebooks (.joblib)
├── outputs/
│   ├── *.csv                    Generated: 13 result tables
│   └── figures/                 Generated: 47 figures (.png)
└── models/                      Generated by NB 09: the saved final pipeline (~233 MB,
                                 not shipped — see models/README.md)
```

The `artifacts/`, `outputs/` and `models/` folders are **created automatically**
when the notebooks run. They are not needed to start.

### What each notebook does

| # | Notebook | Covers |
|---:|---|---|
| **Regression track** | | |
| 01 | Introduction & data loading | Problem statement, libraries, `head`/`tail`/`shape`/`dtypes`/`info`/`describe` |
| 02 | Data quality audit | Missing values, hidden placeholders, duplicates, time-series completeness, physical validity, target statistics |
| 03 | Exploratory data analysis | Target distribution, feature distributions, categorical analysis, correlation heatmap, 4 feature-target plots, hour/season/weekday/holiday/time-series analysis |
| 04 | Outlier analysis | Boxplots, IQR, percentiles, z-scores — and the argument for keeping them |
| 05 | Feature engineering | 14 features: calendar, cyclical sin/cos, rush-hour, weather flags, temperature bands |
| 06 | Preprocessing & baseline | X/y, **split first**, `ColumnTransformer`, leakage proof, Dummy + Linear baselines |
| 07 | All 10 models | Train, predict, evaluate; one ranked comparison table; 4 comparison figures |
| 08 | Tuning & cross-validation | `GridSearchCV` (SVR) + `RandomizedSearchCV` (RF); 5-fold CV of the two strongest |
| 09 | Final analysis & conclusion | Actual-vs-predicted, residuals, heteroscedasticity test, feature importance, error analysis, limitations, conclusion |
| **Classification track** | | |
| 10 | Introduction & data loading | Problem statement, file parsing quirks, combining `adult.data` + `adult.test` |
| 11 | Data quality audit | Hidden `?` missing values, missingness pattern analysis, duplicates, top-coding, **class imbalance** |
| 12 | Exploratory data analysis | Class distribution, numeric and categorical distributions, correlation, feature-target analyses, interaction heatmaps |
| 13 | Cleaning, FE, split, preprocessing | `"Unknown"` encoding, 8 engineered features, `fnlwgt` dropped, **stratified split**, pipeline, leakage proof |
| 14 | Five models & conclusion | Train all 5, comparison table, confusion matrices (raw + normalised), scaling experiment, feature importance, conclusion |
| **Shared** | | |
| 15 | Viva prep & checklist | Results dashboard, **38 viva Q&A**, full Review 1 self-audit, presentation story arc |

---

## 6. Algorithms implemented

### Regression (all 10 required)

| # | Algorithm | scikit-learn class |
|---:|---|---|
| 1 | Linear Regression | `LinearRegression` |
| 2 | Ridge Regression (L2) | `Ridge` |
| 3 | Lasso Regression (L1) | `Lasso` |
| 4 | ElasticNet (L1 + L2) | `ElasticNet` |
| 5 | Polynomial Regression | `PolynomialFeatures` + `LinearRegression` |
| 6 | Decision Tree Regressor | `DecisionTreeRegressor` |
| 7 | Random Forest Regressor | `RandomForestRegressor` |
| 8 | Gradient Boosting Regressor | `GradientBoostingRegressor` |
| 9 | Support Vector Regressor | `SVR` |
| 10 | K-Nearest Neighbors Regressor | `KNeighborsRegressor` |

### Classification Part A (all 5 required)

| # | Algorithm | scikit-learn class |
|---:|---|---|
| 1 | Logistic Regression | `LogisticRegression` |
| 2 | K-Nearest Neighbors | `KNeighborsClassifier` |
| 3 | Gaussian Naive Bayes | `GaussianNB` |
| 4 | Decision Tree Classifier | `DecisionTreeClassifier` |
| 5 | Support Vector Classifier | `SVC` |

---

## 7. Preprocessing approach

Both tracks follow the same discipline, with one difference in the split.

**Order of operations — this order is the point:**

1. Clean and engineer features (**row-wise operations only** — nothing uses a
   dataset-wide statistic, so nothing can leak).
2. **Split into train and test.** No `.fit()` call happens before this line.
3. Define a `ColumnTransformer`, fitted only inside each model's `Pipeline`.

**The pipeline:**

| Step | Applied to | Purpose |
|---|---|---|
| `SimpleImputer(strategy="median")` | numeric | Robustness; median because several features are strongly skewed |
| `StandardScaler()` | numeric | Required by SVR/SVC (RBF kernel), KNN (Euclidean distance) and the regularised linear models |
| `SimpleImputer(strategy="most_frequent")` | categorical | Robustness |
| `OneHotEncoder(handle_unknown="ignore")` | categorical | No false numeric ordering; `ignore` handles categories unseen in training (real risk — some have only 10 rows) |

**Resulting feature counts:** regression 25 raw → **33** after encoding;
classification 20 raw → **88** after encoding.

**Splits:**

* Regression — random 80/20, `random_state=42`. No stratification (a continuous
  target has no classes); train/test distributions were verified instead.
* Classification — **stratified** 80/20, `random_state=42`. Class proportions
  match the full dataset to within **0.006 percentage points**.

---

## 8. Evaluation metrics

| Track | Metrics | Why |
|---|---|---|
| Regression | **R², RMSE, MAE** (all mandatory) + train-set scores as an overfitting diagnostic + 5-fold CV R² for the two strongest models | R² is comparable across problems; MAE is directly readable as "bikes"; RMSE penalises the costly large errors |
| Classification | **Accuracy, weighted F1, confusion matrix** (all mandatory) + weighted and per-class precision/recall | Accuracy alone is misleading at 76/24 imbalance — a model that learns nothing scores 0.7607 |

All metrics are computed by a **single shared function** in `src/ml_utils.py`
(`regression_metrics` / `classification_metrics`), so every model in a track is
scored by identical code.

---

## 9. Installation

### Prerequisites

* **Python 3.10 or newer** (3.12 was used). Check with `python --version`.
* About 500 MB of free disk space.

### Step 1 — get the project and create a virtual environment

```bash
cd ml-capstone-review1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

### Step 2 — install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — place the datasets

See **section 4** above.

### Step 4 — verify the setup

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy, joblib; print('All packages OK')"
python -c "from pathlib import Path; print('Data OK' if Path('dataset/SeoulBikeData.csv').exists() and Path('dataset/adult.data').exists() else 'DATA MISSING - see README section 4')"
```

---

## 10. How to run

> ### Run the notebooks in numerical order the first time.
>
> Each notebook loads the artifact written by the previous one. If you run one
> out of order it will stop with a clear message telling you which earlier
> notebook to run first — it will not produce wrong results silently.
>
> The two tracks are independent of each other: notebooks 01–09 and 10–14 can be
> run in either order as blocks. Notebook 15 needs both tracks finished.

### Option A — Jupyter Lab (recommended)

```bash
cd ml-capstone-review1
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1
jupyter lab
```

A browser tab opens. Then:

1. In the file browser on the left, open the `notebooks/` folder.
2. Double-click `01_regression_introduction_and_data_loading.ipynb`.
3. Run everything with **Run → Run All Cells**, or step through with
   **Shift + Enter**.
4. Wait for it to finish, then open notebook `02`, and so on through `15`.

### Option B — Classic Jupyter Notebook

```bash
jupyter notebook
```

Then navigate to `notebooks/` and open each file in turn. Use
**Cell → Run All**.

### Option C — VS Code

1. Install the **Python** and **Jupyter** extensions from the Extensions panel.
2. **File → Open Folder** and select `ml-capstone-review1`.
3. Open `notebooks/01_regression_introduction_and_data_loading.ipynb`.
4. Click **Select Kernel** in the top right and choose the `venv` interpreter
   (it will be listed as `Python 3.x ('venv')`).
5. Click **Run All** in the toolbar.
6. Repeat for notebooks 02 through 15.

### Option D — run everything from the command line

To execute all 15 notebooks in order without opening a browser:

```bash
pip install nbconvert
cd notebooks
for nb in *.ipynb; do
    echo "Running $nb ..."
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done
```

On Windows PowerShell:

```powershell
cd notebooks
Get-ChildItem *.ipynb | Sort-Object Name | ForEach-Object {
    Write-Host "Running $($_.Name) ..."
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $_.Name
}
```

### Expected runtimes

On a modest laptop (single core), start to finish is roughly **15 minutes**.

| Notebook | Approx. time | Note |
|---|---:|---|
| 01–06 | < 30 s each | |
| **07** | **~4 min** | Polynomial degree 3 fits 7,139 features — see below |
| **08** | **~7 min** | 60 + 54 model fits across the two searches |
| 09 | ~30 s | Includes permutation importance |
| 10–13 | < 15 s each | |
| **14** | **~2 min** | SVC on 39,073 rows is the bottleneck |
| 15 | < 10 s | |

**If notebook 07 is too slow or runs out of memory**, edit the
`POLYNOMIAL_DEGREES` list in the Polynomial Regression section from `[1, 2, 3]`
to `[1, 2]`. Degree 3 is included only because its catastrophic failure is
instructive; removing it changes no conclusion.

---

## 11. Reproducibility

* **`random_state=42`** is used for every train/test split, every model with a
  random component, and both hyperparameter searches. It is defined once in
  `src/ml_utils.py` as `RANDOM_STATE` and imported everywhere.
* **Splits are frozen and shared.** Notebook 06 saves the regression split and
  notebooks 07–09 load it unchanged. Notebook 13 does the same for the
  classification split. All models within a track therefore see identical data.
* **All metrics in this README were produced by executing the notebooks.**
  Nothing is estimated or typed from memory. Notebook 15 re-derives the headline
  tables from the saved artifacts and runs 10 automated consistency checks.
* **Package versions are pinned** in `requirements.txt`.

Re-running the notebooks in order on a fresh machine should reproduce every
number reported here exactly.

---

## 12. Data leakage prevention

This is the requirement most likely to cost marks, so it is worth stating
precisely what was done.

| Requirement | How it is satisfied | Where |
|---|---|---|
| Split before any fitting | `train_test_split` runs before the first `.fit()` call anywhere | NB 06 §8.3, NB 13 §7.2 |
| Scaler fitted on train only | Lives inside a `Pipeline`; `.fit()` sees training rows, `.transform()` handles the test set | NB 06 §8.4, NB 13 §8 |
| Encoder fitted on train only | Same mechanism, with `handle_unknown="ignore"` for unseen categories | NB 06 §8.4, NB 13 §8 |
| Imputer fitted on train only | Same mechanism | Both |
| Cross-validation respects the pipeline | `cross_val_score` / search objects receive the **whole pipeline**, so preprocessing is refitted per fold | NB 07, NB 08 |
| Tuning never uses the test set | Both searches score candidates by CV on the training data only | NB 08 §11 |
| No target-derived features | Every target/mean-encoding candidate was explicitly rejected | NB 05 §7.6, NB 13 §6.8 |
| No lag features in the time series | Under a random split the lagged row often lands in the other split | NB 05 §7.6 |
| Cleaning did not leak | `NaN → "Unknown"` is a constant, not a statistic computed from data | NB 13 §5.3 |
| Feature engineering did not leak | Every engineered feature is computed row-wise from that row's own values | NB 05, NB 13 §6 |
| Test set untouched until final evaluation | It appears only in `.predict()` calls | Both tracks |

### The leakage proof we print

Notebook 06 §8.5 and Notebook 13 §8.3 print a scaling sanity check showing:

* **training** columns: mean exactly `0.0000`, std exactly `1.0000`
* **test** columns: close to, but **not exactly**, 0 and 1

That discrepancy is the evidence the scaling was done correctly — the test set
was standardised using the *training* statistics. **A test set that scaled to
exactly mean 0 and std 1 would indicate the scaler had been fitted on it.**

---

## 13. Key decisions and their justifications

Every non-obvious choice in the project, with where to find the reasoning.

| Decision | Justification | Where |
|---|---|---|
| Removed 295 rows from the bike data | `Functioning Day == "No"` → target is deterministically 0 with zero variance. Not demand, an administrative rule. **The only row removal in the regression track.** | NB 02 §4.6, NB 04 §6.1 |
| **Kept** all 152 statistical outliers | They are warm, dry, evening rush hours — genuine peak demand, and the most operationally important hours. Removing them would cause optimistic bias. | NB 04 §6.5 |
| Did not apply an IQR filter to `Rainfall`/`Snowfall` | When a variable is zero >75% of the time, Q1 = Q3 = 0, so IQR flags *every* non-zero value. The method detects that it rains, not anomalies. | NB 04 §6.2 |
| No log transform of the target | R²/RMSE in log space are not comparable to count space; keeps all ten models comparable and MAE readable as "bikes". | NB 03 §5.A |
| Kept `Dew point` despite r = 0.91 with `Temperature` | Ridge/ElasticNet exist to handle collinearity, and trees are unaffected. Interpret linear coefficients cautiously instead. | NB 03 §5.D, NB 05 §7.6 |
| Adult missing values → `"Unknown"` | Missingness is **informative**: those people earn >50K at 9.5% vs 24.8%. Dropping deletes a population; mode-imputing destroys the signal. | NB 13 §5 |
| Kept 52 duplicate rows | With 15 coarse attributes and 49,000 people, identical rows are expected coincidences between distinct respondents. No identifier exists. | NB 11 §3.2 |
| Kept top-coded values (99999, age 90) | Genuine encoded information ("at or above the ceiling"), not errors. Removing them would delete the highest earners. | NB 11 §3.4 |
| Dropped `fnlwgt` | It is the census **sampling weight** — a property of the survey design, not the person. Correlation with income: −0.006. | NB 13 §6.1 |
| Did **not** apply SMOTE / class weighting | At 3.18 : 1 the imbalance is moderate, not severe. Resampling adds complexity and distorts calibration without fixing a problem this data has. | NB 11 §3.5 |
| Kept `race` and `sex` | Removing a protected attribute does not remove bias — the model reconstructs it from proxies, and we lose the ability to *measure* the disparity. | NB 13 §6.8 |
| Random (not chronological) split on time-series data | Required by the assignment. **Declared as the project's most important limitation**, not quietly benefited from. | NB 03 §5.F.4, NB 09 §13.8 |

---

## 14. Notable findings

1. **Tuning is not a uniform improvement.** SVR gained +0.55 R²; Random Forest
   gained +0.006 from a larger search. Its value depends on how sensitive the
   algorithm is and how badly the defaults mismatch the problem. *(NB 08 §11.3)*
2. **Polynomial degree 3 catastrophically overfits** — 7,139 features for 6,772
   training rows gives a CV R² near −6 × 10⁸. Cross-validation on the training
   set alone was enough to reject it; the test set was never needed. *(NB 07)*
3. **Low correlation ≠ low importance.** Rainfall correlates −0.12 with demand,
   yet mean demand falls from 766 to 251 bikes at the first drop of rain. Pearson's
   r cannot see a threshold effect. *(NB 03 §5.E)*
4. **Feature engineering can substitute for model complexity.** Logistic
   Regression matched a kernel method on the income data, because the banded
   features handed it piecewise approximations of the non-linearities it cannot
   otherwise express. *(NB 14 §10.1)*
5. **A single metric cannot rank models.** Gaussian Naive Bayes is *last* on
   accuracy yet finds 92.4% of high earners — more than any other model. *(NB 14 §10.2)*
6. **A measurement contradicted our own prediction, and we report it.** We
   expected scaling to help KNN; removing the scaler made it slightly *better*,
   because `StandardScaler` amplifies rare one-hot columns by dividing by their
   tiny standard deviation. *(NB 14 §11.2)*
7. **The planted control feature stayed silent.** `Day` (day of month) was added
   deliberately as a feature with no theoretical reason to matter. It ranked 18th
   of 33 — evidence the model learned structure, not noise. *(NB 05 §7.1, NB 09 §13.4)*

---

## 15. Known limitations

Set out in full in Notebook 09 §13.8 (regression) and Notebook 14 §12.5
(classification). The three most important:

1. **The random split on autocorrelated time-series data** means the regression
   R² answers "how well does this explain demand given conditions?", not "how
   well would it forecast next month?"
2. **Systematic underprediction at peak demand** is the operationally expensive
   failure mode — the model is least accurate exactly when a shortage costs most.
3. **No causal claims are supportable.** Feature importance shows what a model
   relies on, not what causes an outcome. This matters most in the income track,
   where `marital-status` ranks first but is confounded with age and career stage.

---

## 16. Academic integrity

* All code in this repository was written for this project.
* **Generative AI assistance was used for code scaffolding**, as permitted by
  guidelines §7.5. The analysis, interpretation, feature-engineering decisions
  and conclusions are the team's own, and every quantitative claim is traceable
  to executed output visible in the notebooks.
* No code was copied from Stack Overflow, blogs or other external sources. Where
  a standard technique is used (cyclical time encoding, Tukey's IQR rule), it is
  explained in the notebook rather than cited, because it is textbook material.
* The datasets are public research datasets from the UCI Machine Learning
  Repository and are cited in section 3.

---

## 17. What comes next (Review 2)

| Track | Planned work |
|---|---|
| Classification Part B | Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP — then the consolidated 10-algorithm table with Precision, Recall, F1 and ROC-AUC |
| Clustering | K-Means (elbow curve) and Agglomerative Hierarchical (dendrogram); Silhouette, Davies-Bouldin, Calinski-Harabasz; PCA and t-SNE visualisation |
| Improvements identified in Review 1 | Cross-validate all five Part-A models; tune the Decision Tree's `max_depth`; threshold tuning via precision–recall curves; a fairness audit computing metrics per subgroup |

Notebook 14 §11.4 makes the case for Part B directly: the Decision Tree's large
train/test gap is precisely the variance problem that ensembles exist to solve.
