# Machine-Learning-for-Healthcare
Unsupervised learning project that segments patients into meaningful groups based on demographics, clinical conditions, insurance, and visit history — using distance-based clustering to identify natural patient cohorts.

## Motivation
 
Healthcare organizations — from small clinics to large hospital networks —
benefit from understanding their patient population as distinct groups
rather than as one undifferentiated pool. This project explores whether
clustering can reveal meaningful patient segments in a real-world dataset.
 
## Dataset
 
[Patient Segmentation Dataset](https://www.kaggle.com/) (Kaggle) — 2,000
patient records with demographics, clinical conditions, insurance type, and
visit information.
 
## Approach
 
1. **Data cleaning** — identified and handled "hidden" missing values
   (e.g. `"Unknown"` in `City`, `"None"` in `Primary_Condition`) that don't
   surface with a standard `isnull()` check, and made a deliberate choice
   about how to treat each (dropped vs. imputed as `"No Condition"`).
2. **Exploratory data analysis** — distributions of numeric and categorical
   features, correlation structure between clinical/demographic variables.
3. **Clustering** — [fill in: algorithm(s) used, how many clusters, how you
   chose k, how you handled mixed categorical/numeric data]
4. **Interpretation** — [fill in: what each cluster represents in plain
   terms, and any clinical or operational implication]
## Key findings
 
- [fill in 2–3 concrete takeaways once you've run your own analysis]
## What I extended beyond the guided exercise
 
- [fill in: e.g. tried a second clustering algorithm and compared results,
  evaluated cluster stability, added a visualization not in the original
  walkthrough, wrote up clinical interpretation of each segment]
## Tech stack
 
Python, pandas, NumPy, scikit-learn, seaborn/matplotlib
 
## Running it
 
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
 
## Attribution
 
Built while working through a guided exercise on patient clustering
(DataCamp, "Machine Learning for Healthcare"), using the dataset above.
