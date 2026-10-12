# Student Grade Prediction

This project predicts student grades with an early-prediction ensemble model. The
app uses a balanced Random Forest because it performs better than the original
single C4.5-style tree on this dataset. The model intentionally excludes `Final_Score` and `Total_Score` to avoid
data leakage and to simulate a prediction made before the final assessment. It
also excludes extracurricular activities, home internet access, parent
education, and family income so the prediction focuses on academic and
well-being signals. Department is also excluded to keep the model
department-neutral.

## Streamlit app

Install dependencies and start the interactive dashboard:

```powershell
uv sync
uv run streamlit run app.py
```

The app includes:

- A student grade prediction form
- Model accuracy and confusion matrix
- Grade distribution and score comparisons
- A dataset preview

The app runs from `app.py` and reads `Students Performance Dataset.csv` from the
project root.# dmdw-el
