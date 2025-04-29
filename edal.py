import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, levene, zscore
import google.generativeai as genai

# ⚠️ Hard-coded API key for demonstration
# In production, secure this key appropriately (e.g., environment variable)
genai.configure(api_key="AIzaSyCRZGcQV9YiXqB6mtQjii9a3Lm2jSw_STo")

# Streamlit Page Setup
st.set_page_config(page_title="🔍 Comprehensive EDA & Regression with Gemini", layout="wide")
st.title("🧪 Comprehensive EDA, Regression & Diagnostics with Gemini Interpretation")

# File Uploader
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx) for analysis", type=["xlsx"])

if uploaded_file:
    # Load Data
    df = pd.read_excel(uploaded_file)
    st.write("### Dataset Preview", df.head())
    st.write("**Shape:**", df.shape)

    # Identify Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Sidebar Controls
    st.sidebar.header("Controls")
    selected_numeric = st.sidebar.multiselect("Numeric columns for analysis:", numeric_cols, default=numeric_cols[:3])
    selected_categorical = st.sidebar.multiselect("Categorical columns for analysis:", cat_cols)
    dep_var = st.sidebar.selectbox("Dependent variable (y):", numeric_cols)
    indep_vars = st.sidebar.multiselect("Independent variables (X):", [c for c in numeric_cols if c != dep_var])

    # --- Exploratory Data Analysis ---
    if selected_numeric:
        st.subheader("Descriptive Statistics")
        desc = df[selected_numeric].describe().T.round(3)
        st.dataframe(desc)

        # Normality Test
        sw_results = {col: shapiro(df[col].dropna()) for col in selected_numeric}
        sw_series = pd.Series({col: f"W={s[0]:.3f}, p={s[1]:.3f}" for col, s in sw_results.items()}, name="Shapiro-Wilk")
        st.subheader("Shapiro-Wilk Normality Test")
        st.dataframe(sw_series.to_frame())

        # Outlier Detection
        outliers = {col: int((np.abs(zscore(df[col].dropna())) > 3).sum()) for col in selected_numeric}
        st.subheader("Outlier Counts (|z|>3)")
        st.dataframe(pd.Series(outliers, name="#Outliers").to_frame())

    if selected_categorical and selected_numeric:
        st.subheader("Boxplots & Homogeneity of Variance")
        for col in selected_numeric:
            fig, ax = plt.subplots()
            sns.boxplot(x=selected_categorical[0], y=col, data=df, ax=ax)
            ax.set_title(f"{col} by {selected_categorical[0]}")
            st.pyplot(fig)
        # Levene test
        lev_text = []
        groups = [grp[selected_numeric].dropna() for _, grp in df.groupby(selected_categorical[0])]
        # apply Levene per numeric column
        for col in selected_numeric:
            vals = [grp[col].values for _, grp in df.groupby(selected_categorical[0])]
            stat, p = levene(*vals)
            lev_text.append(f"Levene test for {col} by {selected_categorical[0]}: W={stat:.3f}, p={p:.3f}")
        st.write("\n".join(lev_text))

    if selected_numeric:
        st.subheader("Distribution Plots")
        for col in selected_numeric:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    if len(selected_numeric) > 1:
        st.subheader("Correlation Matrix & Pairplot")
        corr = df[selected_numeric].corr().round(3)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.dataframe(corr)
        pairfig = sns.pairplot(df[selected_numeric].dropna())
        st.pyplot(pairfig.fig)

    if selected_categorical:
        st.subheader("Categorical Value Counts")
        for col in selected_categorical:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

    # --- Linear Regression & Diagnostics ---
    if indep_vars:
        st.subheader("Linear Regression")
        X = sm.add_constant(df[indep_vars])
        y = df[dep_var]
        model = sm.OLS(y, X, missing='drop').fit()

        # Coefficients and Model Stats
        coef_df = pd.DataFrame(model.params, columns=['coef']).join(
            pd.DataFrame({'std_err': model.bse, 't': model.tvalues, 'p_value': model.pvalues})
        ).round(4)
        st.write("### Regression Coefficients")
        st.dataframe(coef_df)

        stats = pd.Series({
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'F-statistic': model.fvalue,
            'Prob (F-statistic)': model.f_pvalue,
            'AIC': model.aic,
            'BIC': model.bic,
            'Durbin-Watson': sms.durbin_watson(model.resid)
        }).round(4)
        st.write("### Model Summary")
        st.dataframe(stats)

        # Diagnostics
        st.subheader("Diagnostics Plots & Tests")
        # VIF
        vif = pd.DataFrame({'variable': X.columns,
                            'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
        st.write("**Variance Inflation Factors**")
        st.dataframe(vif.round(2))

        resid = model.resid
        fitted = model.fittedvalues

        # Residuals vs Fitted
        fig, ax = plt.subplots()
        sns.scatterplot(x=fitted, y=resid, ax=ax)
        ax.axhline(0, linestyle='--', color='red'); ax.set(title='Residuals vs Fitted')
        st.pyplot(fig)

        # Normal Q-Q
        fig = sm.qqplot(resid, line='45', fit=True)
        plt.title('Normal Q-Q'); st.pyplot(fig)

        # Scale-Location
        fig, ax = plt.subplots()
        sns.scatterplot(x=fitted, y=np.sqrt(np.abs(resid)), ax=ax)
        ax.set(title='Scale-Location'); st.pyplot(fig)

        # Shapiro-Wilk Residuals
        sw_r, sw_p = shapiro(resid.dropna())
        st.write(f"Shapiro-Wilk on Residuals: W={sw_r:.3f}, p={sw_p:.3f}")

        # Breusch-Pagan
        bp = sms.het_breuschpagan(resid, model.model.exog)
        bp_labels = ['LM Stat', 'LM p-value', 'F Stat', 'F p-value']
        bp_res = dict(zip(bp_labels, bp))
        st.write("**Breusch-Pagan Test**")
        st.write({k: round(v,4) for k,v in bp_res.items()})

        # Influence Plot
        fig, ax = plt.subplots(figsize=(8,6))
        sm.graphics.influence_plot(model, ax=ax, criterion='cooks')
        plt.title('Influence Plot'); st.pyplot(fig)

        # --- Gemini Interpretation ---
        prompt = (
            "You are an expert data analyst. "
            "Interpret the following analysis results in detail, highlighting key findings, assumptions, and next steps.\n\n" +
            "Descriptive Statistics:\n" + desc.to_string() + "\n\n" +
            "Shapiro-Wilk:\n" + sw_series.to_string() + "\n\n" +
            "Outliers:\n" + pd.Series(outliers).to_string() + "\n\n" +
            ("Correlation Matrix:\n" + corr.to_string() + "\n\n" if len(selected_numeric)>1 else "") +
            ("VIF:\n" + vif.round(2).to_string(index=False) + "\n\n") +
            "Regression Coefficients:\n" + coef_df.to_string() + "\n\n" +
            "Model Summary:\n" + stats.to_string() + "\n\n" +
            f"Durbin-Watson: {stats['Durbin-Watson']}\nShapiro Residuals W={sw_r:.3f}, p={sw_p:.3f}\n" +
            f"Breusch-Pagan: {bp_res['LM Stat']:.3f} (p={bp_res['LM p-value']:.3f})\n\n"
        )
        if st.button("🔮 Gemini Insights"):
            with st.spinner("Generating interpretation…"):
                try:
                    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
                    st.subheader("🤖 Gemini Interpretation")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")
else:
    st.info("Upload an Excel (.xlsx) file to start your analysis.")
