# Neonatal Feeding Intolerance Risk Calculator

This folder contains a Streamlit web calculator for the fixed final 5-feature LightGBM model.

## Model inputs

1. Delayed feeding (`feeddelay`, 0/1)
2. Blood transfusion (`bloodtransfusion`, 0/1)
3. Gestational age (`gestationalgeweek`)
4. Total bilirubin (`tbil`)
5. Birth weight (`birthweight`)

The exact model feature order is preserved in `clinical_feature_order.csv` and
`model_matrix_column_order.csv`.

## Fixed model

The calculator loads `final_5feature_lightgbm_model.txt` directly. It does not retrain the model.

Exported configuration:
- num_leaves = 7
- learning_rate = 0.05
- min_data_in_leaf = 5
- feature_fraction = 0.8
- bagging_fraction = 0.8
- nrounds = 31
- prespecified threshold = 0.4823497017

The exported-model verification file reported a maximum absolute difference from the
original R predictions of approximately 5e-16.

## Run locally

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

A local browser window should open automatically.

## Deploy with GitHub + Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files in this folder, keeping `streamlit_app.py`,
   `requirements.txt`, and `final_5feature_lightgbm_model.txt` in the repository root.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Create a new app and select the repository.
5. Set the main file path to `streamlit_app.py`.
6. Deploy.
7. After deployment, test several patients against the R verification probabilities before
   citing the calculator URL in the manuscript.

## Important unit check before public release

The exported model confirms the feature names and development-data ranges. The final public
calculator should use exactly the same clinical definitions, units, and time windows as the
manuscript/data dictionary.

The current interface labels gestational age in weeks and birth weight in grams. The total
bilirubin field is intentionally labeled without a unit because the deployment export itself
does not encode the laboratory unit. Confirm the final TBIL unit from the study data dictionary
or manuscript before public release, then add that unit to the field label.

## Interpretation

The calculator displays:
- predicted FI probability;
- the prespecified threshold;
- whether the model probability is above or below that threshold;
- individual TreeSHAP contributions for the five predictors.

Positive SHAP contributions push the fitted model output toward a higher FI probability and
negative values push it toward a lower FI probability. They are model explanations, not causal
effects.

## Research disclaimer

This tool is intended for research and model reporting. It is not a substitute for clinical
judgment or a validated medical-device workflow.
