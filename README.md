# Task 2 — Exploratory Data Analysis (EDA)

## Overview
This task performs a structured Exploratory Data Analysis on the **Amazon Product Reviews** dataset (`amazon_reviews.csv`). The goal is to understand the data structure, detect patterns and anomalies, test hypotheses, and surface issues before any modelling work.

---

## Dataset
| Property | Value |
|---|---|
| File | `amazon_reviews.csv` (one level up from this folder) |
| Rows | 4 915 |
| Columns | 12 |
| Period | January 2012 – December 2014 |
| Product | Single ASIN — `B007WTAJTO` |

### Column Reference
| Column | Type | Description |
|---|---|---|
| `reviewerID` | string | Unique reviewer identifier |
| `asin` | string | Amazon product ID |
| `reviewerName` | string | Display name (1 missing) |
| `helpful` | string | Raw helpfulness array `[yes, total]` |
| `reviewText` | string | Full review body (1 missing) |
| `overall` | float | Star rating (1–5) |
| `summary` | string | Review headline |
| `unixReviewTime` | int | UNIX timestamp |
| `reviewTime` | string | Human-readable date |
| `day_diff` | int | Days since a reference date |
| `helpful_yes` | int | Helpful votes received |
| `total_vote` | int | Total votes received |

---

## Questions Asked Before Analysis
1. What is the overall distribution of star ratings? Is it skewed?
2. How does review volume change over time (by year / month)?
3. Do lower-rated reviews tend to be longer (more detailed complaints)?
4. How helpful are reviews? What fraction receive any votes?
5. Is there a correlation between review length and helpfulness votes?
6. Do newer reviews (smaller `day_diff`) have higher ratings?
7. Are there extreme outliers in `total_vote` or `review_length`?
8. Are there missing or duplicate records?
9. Do reviews written closer to the purchase date differ in rating?
10. What are the statistical properties of key numeric variables?

---

## Key Findings
- **Ratings are heavily skewed**: 79.8 % of reviews are 5-star.
- **Low ratings → longer reviews**: 1-star reviews average 559 chars vs 234 for 5-star.
- **Helpfulness is sparse**: only 11.3 % of reviews received any votes.
- **Review length predicts votes**: Pearson r = 0.45 between length and total_vote.
- **Temporal decline in ratings**: newer reviews rate slightly higher than older ones.
- **Outliers exist**: max total_vote = 2 020, max review_length = 8 638 chars.
- **Data quality is good**: only 1 missing reviewerName and 1 missing reviewText; zero duplicates.

---

## Hypothesis Test Results
| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | 5-star reviews are shorter than 1-star reviews | Mann-Whitney U | ✅ Supported (p < 0.001) |
| H2 | Voted reviews are longer than unvoted reviews | Mann-Whitney U | ✅ Supported (p < 0.001) |
| H3 | Review length differs across all star ratings | Kruskal-Wallis | ✅ Supported (p < 0.001) |

---

## Project Structure
```
Task_2_EDA/
├── eda_analysis.py      ← Main EDA script
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── report.html          ← Visual summary report
└── plots/               ← Generated charts (auto-created on run)
    ├── 01_rating_distribution.png
    ├── 02_review_volume_over_time.png
    ├── 03_review_length_by_rating.png
    ├── 04_helpfulness_analysis.png
    ├── 05_correlation_heatmap.png
    ├── 06_daydiff_vs_rating.png
    └── 07_outlier_detection.png
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the EDA script
```bash
# From inside Task_2_EDA/
python eda_analysis.py
```

All charts are saved to `Task_2_EDA/plots/`. Printed output covers statistics, hypothesis tests, and a key-findings summary.

### 3. View the report
Open `report.html` in any browser for a visual one-page summary.

---

## Dependencies
| Package | Purpose |
|---|---|
| `pandas` | Data loading, wrangling, groupby |
| `numpy` | Numeric operations, regression |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scipy` | Hypothesis testing (Mann-Whitney U, Kruskal-Wallis) |
