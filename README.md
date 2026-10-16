# Employee Attrition Prediction

A notebook that explores an HR dataset of 1,470 employees and trains tree-based classifiers to predict which employees leave the company.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0)

> Type: Jupyter notebook (tabular binary classification)
>
> Note: this repository is named `Vineeth_project` for historical reasons. Its content is the employee attrition notebook described here.

## Overview

Understanding why employees leave helps an organisation plan retention measures. This project works through a standard supervised-learning pipeline on an employee attrition table: exploratory analysis of who leaves, feature preparation (scaling and one-hot encoding), a Random Forest baseline, and a Gradient Boosting model tuned with a grid search.

Both models reach roughly 87% test accuracy, but the data is imbalanced (237 of 1,470 employees left, 16.12%) and both models find only about one in five leavers (recall 0.21 and 0.23 on the leaver class). See [Results](#results) for the numbers and what they do and do not show.

## Contents

| File | What it is |
|---|---|
| [employee-attrition-prediction.ipynb](employee-attrition-prediction.ipynb) | The full analysis (48 cells), committed with its saved outputs and 16 embedded plots |
| [requirements.txt](requirements.txt) | Python packages imported by the notebook |
| `HR_Employee_Attrition-1.csv` | The input data. **Not committed** - see [Data](#data) |

## Data

- **File the notebook expects:** `HR_Employee_Attrition-1.csv`, read with `pd.read_csv('HR_Employee_Attrition-1.csv')`. Place it **next to the notebook, in the repository root**, because the path is relative.
- **Content:** one row per employee. The notebook's outputs show 1,470 employees (see [Limitations](#limitations-and-known-issues) for one inconsistent print) and 35 columns, including `Attrition` (the target, `Yes` or `No`), `Age`, `BusinessTravel`, `DailyRate`, `Department`, `DistanceFromHome`, `Education`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `MonthlyIncome`, `OverTime`, `YearsAtCompany` and others. 237 employees have `Attrition = Yes` and 1,233 have `No`.
- **Source:** not stated in the repository. The column layout matches the publicly distributed IBM HR Analytics Employee Attrition dataset, but this is an inference and should be confirmed by whoever supplies the file. Check the provider's terms of use before reusing the data.
- **Committed?** No. The file is listed in `.gitignore`.

## Approach

1. **Load and explore** - basic statistics, then plots of attrition overall, by age, by distance from home, and by gender, job role, education field, marital status, business travel, overtime and department (bar charts with the within-category percentages), plus a correlation heatmap.
2. **Clean** - drop `EmployeeCount`, `EmployeeNumber`, `StandardHours` and `Over18` (identifiers and constant columns).
3. **Prepare features** - split the remaining columns into 23 numeric columns, scaled with `StandardScaler`, and 7 categorical columns, one-hot encoded with `pd.get_dummies` into 28 columns, giving 51 features. `Attrition` is mapped to 1 (`Yes`) and 0 (`No`).
4. **Split** - stratified 80/20 train/test split with `random_state=50`: 1,176 training rows and 294 test rows.
5. **Baseline** - `RandomForestClassifier(random_state=42, oob_score=True)`.
6. **Tuned model** - `GradientBoostingClassifier(max_features=0.7, learning_rate=0.3)` tuned with `GridSearchCV` (accuracy, 2-fold cross-validation) over `max_depth` (6 values), `min_samples_leaf` (4 values) and `n_estimators` (5 values): 120 candidates, 240 fits.
7. **Evaluate** - accuracy and a per-class classification report on the test set, ROC curves, and the feature importances of the tuned model.

## Results

All figures are from the saved notebook outputs (test set of 294 employees: 247 stayed, 47 left).

| Model | Test accuracy | Leaver class: precision | Leaver class: recall | Leaver class: F1 | Notebook section |
|---|---|---|---|---|---|
| Random Forest baseline | 87.07% | 0.91 | 0.21 | 0.34 | Base Model / Model Evaluation |
| Gradient Boosting, as evaluated right after the grid search | 86.73% (derived, see below) | 0.79 | 0.23 | 0.36 | Hyper-parameter tuning |
| Same Gradient Boosting estimator, refitted in the next cell | 87.76% | not printed | not printed | not printed | Final scoring cell (after the tuning section) |

The 86.73% is not printed directly: it is the baseline accuracy (87.07%) plus the notebook's printed "Improvement of -0.34%", and it matches the leaver-class precision and recall above (11 of 47 leavers found, 3 false alarms, 255 of 294 test employees correct). The classification report is printed only for this first evaluation; after the estimator is refitted (`rf_best.fit(x_train, y_train)`) only its accuracy of 87.76% is printed.

Other printed values: out-of-bag score of the baseline 85.54%; best grid-search cross-validation accuracy 86.82% with `learning_rate=0.3`, `max_depth=12`, `max_features=0.7`, `min_samples_leaf=20`, `n_estimators=25`; training accuracy 100.00% for both models. The most important features of the tuned model are `MonthlyIncome` (0.102), `OverTime_Yes` (0.071), `DistanceFromHome` (0.071), `Age` (0.068), `DailyRate` (0.056) and `JobLevel` (0.045).

How to read these numbers:

- **Accuracy is close to a trivial baseline.** Predicting "stays" for every employee would score 84.01% on this test set (247 of 294). The models add roughly three points of accuracy over that baseline (2.7 to 3.7 points, depending on which printed figure is used).
- **Most leavers are missed.** Recall of 0.21 and 0.23 means only about 10 to 11 of the 47 leavers are found; the high precision applies to the few employees the models do flag.
- **The two models are not clearly different.** The notebook prints "Improvement of -0.34%" for the tuned model at grid-search time (one test employee fewer correct than the baseline), while the later refit of the same estimator scores 87.76% (258 of 294 correct, three more than at grid-search time). The Gradient Boosting model has no fixed random seed, so repeated fits can differ by this much; the models cannot be ranked from a single run.
- **Overfitting.** Both models reach 100% training accuracy.

## How to run

1. Create a Python environment. The notebook's saved metadata records Python 3.8.3.
2. Install the dependencies and Jupyter:
   ```bash
   pip install -r requirements.txt
   pip install notebook
   ```
3. Put `HR_Employee_Attrition-1.csv` in the repository root, next to the notebook.
4. Open and run the notebook:
   ```bash
   jupyter notebook employee-attrition-prediction.ipynb
   ```

The notebook was left unmodified and uses APIs that were removed or changed in later releases: `plot_roc_curve` (removed in scikit-learn 1.2), `sns.distplot` (deprecated in seaborn), and `df.corr()` on a table that still contains text columns (raises an error from pandas 2.0). On current versions, either install older releases of these packages or adjust those calls.

## Project structure

```text
.
├── README.md                              # this file
├── employee-attrition-prediction.ipynb    # the analysis, with saved outputs
├── requirements.txt                       # packages imported by the notebook
├── .gitignore                             # keeps the dataset and notebook checkpoints out of git
└── HR_Employee_Attrition-1.csv            # NOT committed - supply it and place it here
```

## Limitations and known issues

- **Class imbalance is not addressed.** The notebook notes the imbalance but does not use class weights, resampling, a different decision threshold or a recall-oriented metric; the low recall on leavers is the main weakness of the results.
- **Single split, small grid.** One train/test split and 2-fold cross-validation are used, so the reported figures carry sampling noise (47 leavers in the test set).
- **Scaling before splitting.** `StandardScaler` is fitted on all 1,470 rows before the train/test split.
- **Inconsistent output.** The first cell that loads the data prints a shape of `(2940, 35)`, while every other output shows 1,470 rows, so the saved outputs may come from more than one run or version of the data file.
- **Labelling.** The section titled "Hyper-parameter tuning for the Random Forest" actually tunes a Gradient Boosting model, and the last cell is empty. `map` is used as a variable name, shadowing the built-in.
- **Data not included**, so the notebook cannot be re-run from this repository alone.

## Credits

- **Data:** source not recorded in the repository (see [Data](#data)).
- **Notebook provenance:** the repository does not record the origin of the analysis. The notebook's embedded metadata contains a Colab notebook name (`employee prediction new.ipynb`) and a papermill execution record dated 2022-02-07; no authorship is claimed here.
- **Libraries:** pandas, NumPy, scikit-learn, seaborn and Matplotlib.

## License

No license file - all rights reserved by the respective authors. The dataset is subject to its own terms of use.
