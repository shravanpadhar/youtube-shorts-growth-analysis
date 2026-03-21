# 📈 YouTube Shorts: Engagement & Growth Velocity
 
> **Predicting algorithmic momentum and audience interaction depth in viral short-form content using EDA, NLP, Clustering, and Regression.**
 
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
 
---
 
## 📌 Project Overview
 
This end-to-end data science project analyzes **799 viral YouTube Shorts** to uncover the statistical and linguistic signals that drive algorithmic growth. Rather than relying on raw lifetime metrics like total views, the analysis focuses on two engineered targets:
 
| Metric | Formula | What it measures |
|---|---|---|
| `Views_Per_Day` | Views ÷ Age_In_Days | **Algorithmic momentum / Velocity** |
| `Engagement_Rate_%` | (Likes + Comments) ÷ Views × 100 | **Audience interaction depth** |
 
The project walks through a complete data science workflow — from raw inspection to predictive regression — and concludes with an **"Ideal Short" profile** backed by data.
 
---
 
## 🗂️ Repository Structure
 
```
youtube-shorts-growth-analysis/
│
├── 📓 youtube-shorts-growth-regressor-nlp-insights.ipynb   ← Main analysis notebook
├── 📄 README.md                                            ← You are here
├── 📊 data/
│   └── YouTube_Shorts_Engagement_and_Growth_Velocity.csv  ← Dataset (799 rows, 0 nulls)
│
├── 📸 outputs/                                             ← All generated charts
│   ├── 01_univariate_histograms.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_velocity_decay.png
│   ├── 04_boxplots_outliers.png
│   ├── 05_tfidf_keywords.png
│   ├── 06_sentiment_vs_engagement.png
│   ├── 07_wordclouds.png
│   ├── 08_kmeans_clusters.png
│   ├── 09_channel_benchmarking.png
│   ├── 10_model_comparison.png
│   ├── 11_feature_importance.png
│   └── 12_title_length_vs_engagement.png
│
├── 🐍 youtube_shorts_analysis.py                           ← Standalone Python script
├── 📋 requirements.txt                                     ← All dependencies
└── 📄 LICENSE
```
 
---
 
## 📊 Dataset
 
| Property | Value |
|---|---|
| Source | Kaggle / Apify extraction |
| Rows | 799 |
| Columns | 11 |
| Missing values | 0 (zero-null) |
 
### Column Reference
 
| Column | Type | Description |
|---|---|---|
| `Video_ID` | string | Unique YouTube video identifier |
| `Title` | string | Raw video title text |
| `Channel_Name` | string | Publishing creator |
| `Views` | int | Total lifetime views |
| `Likes` | int | Total lifetime likes |
| `Comments` | int | Total lifetime comments |
| `Age_In_Days` | int | Days since publication |
| `Engagement_Rate_%` | float | *(Likes + Comments) / Views × 100* |
| `Views_Per_Day` | float | *Views / Age_In_Days* (velocity) |
| `Video_URL` | string | Direct YouTube link |
| `Description_Length` | int | Character count of video description |
 
---
 
## 🔬 Analysis Workflow
 
### 1. 📥 Data Loading & Inspection
- Verified zero-null status via `.isnull().sum()`
- Reviewed data types with `.info()` and `.describe()`
- Inspected `Title` and `Channel_Name` column structure
 
### 2. 📊 Exploratory Data Analysis (EDA)
- **Univariate:** Histograms (log-scaled) for `Views`, `Engagement_Rate_%`, `Views_Per_Day` — confirmed heavy right skew
- **Bivariate:** Correlation heatmap across all numeric features
- **Velocity Decay:** Scatter plot of `Age_In_Days` vs `Views_Per_Day` with linear trend line
- **Outlier Detection:** Box plots to surface "Super-Viral" outliers
 
### 3. 🔤 NLP & Text Mining
- **Feature Engineering:** `Title_Length` (character count) and `Emoji_Count` per video
- **TF-IDF:** Top 20 keywords and bigrams most associated with high-engagement titles
- **Sentiment Analysis:** TextBlob polarity + VADER compound scores correlated with `Engagement_Rate_%`
- **Word Clouds:** High-velocity vs Low-velocity title vocabulary comparison
 
### 4. 🔵 Clustering & Channel Benchmarking
- **K-Means (k=4):** Segmented videos into algorithmic tiers — *Viral Hits, Dark Horses, Moderate, Slow Burns*
- **Channel Benchmarking:** Ranked creators by average engagement consistency (min. 2 videos)
 
### 5. 🤖 Predictive Modeling
- **Target:** `log1p(Views_Per_Day)` (log-transformed to handle skew)
- **Features:** Title Length, Emoji Count, Description Length, Age In Days, Likes, Comments, Has Emoji, Has Hashtag
- **Models Trained:**
  - `RandomForestRegressor` (200 estimators)
  - `XGBRegressor` (300 estimators, lr=0.05)
 
---
 
## 🏆 Model Results
 
| Model | MAE | R² Score |
|---|---|---|
| **Random Forest** | **0.6931** | **0.8720** ✅ Best |
| XGBoost | 0.7160 | 0.8627 |
 
> Both models were trained on an 80/20 train-test split with `StandardScaler` preprocessing.
 
---
 
## 💡 Key Findings — The "Ideal Short" Profile
 
```
✦ Best title length   →  50–70 characters
✦ Description length  →  ~505 characters (high-engagement average)
✦ Age effect          →  Newer videos strongly dominate velocity
                         (Age_In_Days is negatively correlated with Views_Per_Day)
✦ Emoji impact        →  Videos with 2+ emojis show measurable engagement difference
✦ Top model           →  Random Forest (R² = 0.872)
```
 
---
 
## 📸 Sample Visualizations
 
| Velocity Decay | K-Means Clusters | Feature Importance |
|---|---|---|
| Age vs Views/Day scatter with trend | 4-tier algorithmic segmentation | RF/XGBoost feature rankings |
 
*(All 12 charts are saved in the `/outputs` folder)*
 
---
 
## ⚙️ Getting Started
 
### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/youtube-shorts-growth-analysis.git
cd youtube-shorts-growth-analysis
```
 
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Run the notebook
```bash
jupyter notebook youtube-shorts-growth-regressor-nlp-insights.ipynb
```
 
Or run the standalone script:
```bash
python youtube_shorts_analysis.py
```
 
---
 
## 📦 Requirements
 
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
textblob
vaderSentiment
wordcloud
emoji
jupyter
```
 
*(Full pinned versions in `requirements.txt`)*
 
---
 
## 🛠️ Tech Stack
 
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![XGBoost](https://img.shields.io/badge/-XGBoost-EC4E20?style=flat)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat)
![NLTK/TextBlob](https://img.shields.io/badge/-TextBlob%20%7C%20VADER-6DB33F?style=flat)
 
---
 
## 👤 Notebook Author
 
**Shravan Padhar**
- 📅 Project Date: March 19, 2025
- 🔗 [LinkedIn](https://linkedin.com/in/shravanpadhar) · [Kaggle](https://kaggle.com/shravanpadhar) · [GitHub](https://github.com/shravanpadhar)
 
---
## 👤 Dataset Creator 
 
**Kanchana Gajamuthu**
- 🔗 · [Kaggle](https://kaggle.com/kanchana1990) · 
 
---
 ### 🔗 Quick Links
* **Medium Article:** [Read the Artical](https://medium.com/@shravanpadhar/decoding-virality-a-data-science-deep-dive-into-youtube-shorts-307943493163)
* **Kaggle Notebook:** [Interactive Analysis](https://www.kaggle.com/code/shravanpadhar/youtube-shorts-growth-regressor-nlp-insights)
* **Dataset:** [Download via Kaggle](https://www.kaggle.com/datasets/kanchana1990/youtube-shorts-engagement-and-growth-velocity)

---
## 📄 License
 
This project is licensed under the [MIT License](LICENSE). The dataset was sourced from Kaggle and extracted via Apify infrastructure.
 
---
 
## ⭐ If you found this useful, please star the repo!
