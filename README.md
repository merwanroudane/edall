This app empowers you to take a raw Excel dataset all the way through to actionable insights—in just a few clicks. Its core capabilities are:

- **Data Ingestion & Preview**  
  • Upload any `.xlsx` file  
  • Instantly view the first rows, overall shape and column names  

- **Exploratory Data Analysis (EDA)**  
  • **Descriptive statistics** (mean, median, quartiles, etc.)  
  • **Normality checks** (Shapiro–Wilk test per variable)  
  • **Outlier detection** (z-score > |3| counts)  
  • **Variance homogeneity** (boxplots + Levene’s test against a chosen categorical)  
  • **Distribution visuals** (histograms with KDE)  
  • **Correlation analysis** (heatmap + pairwise scatterplots)  
  • **Categorical summaries** (bar charts of value counts)  

- **Linear Regression Modeling**  
  • Select any numeric column as **y** (dependent) and one or more as **X** (independent)  
  • Fit an OLS model in real time  
  • Display **coefficient table** (estimates, SE, t-stats, p-values)  
  • Show **model summary** (R², adjusted R², F-stat, AIC/BIC, Durbin–Watson)  

- **Post-Regression Diagnostics**  
  • **Multicollinearity** check via Variance Inflation Factors (VIF)  
  • **Residual analysis**:  
    – Residuals vs. fitted values  
    – Normal Q–Q plot  
    – Scale–location plot  
  • **Statistical tests**:  
    – Shapiro–Wilk on residuals (normality)  
    – Breusch–Pagan (heteroskedasticity)  
  • **Influence detection**: Cook’s distance plot  

- **Automated AI Interpretation**  
  • Builds a detailed natural-language prompt from every computed result  
  • Sends it to **Google Gemini 1.5-flash**  
  • Renders an expert-level narrative explaining findings, checking assumptions, and recommending next analytical steps  

All of this runs in a lightweight Streamlit interface—no manual coding needed once the app is running.
