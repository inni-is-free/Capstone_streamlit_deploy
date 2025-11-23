#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import gdown

sns.set_style("whitegrid")
st.set_page_config(
    page_title="Lease Renewal Capstone Dashboard",
    layout="wide"
)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
import platform

# ---- Korean font settings ----
system = platform.system()
if system == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")   
elif system == "Darwin":
    matplotlib.rc("font", family="AppleGothic")     
else:
    matplotlib.rc("font", family="NanumGothic")   

matplotlib.rc("axes", unicode_minus=False)  


# In[2]:


# =========================================
# 0. Common cache functions
# =========================================


DF_FINAL_ID = "1vlnmBKaO0GvcOq0m5BWj6DAtQJXS5WPO"
PRED_ID     = "1bYvlJAXtXmaNqcXXOPiafpoKO90Z3Wst"


@st.cache_data(show_spinner="Loading df_final from Google Drive...")
    
def load_df_final():
    url = f"https://drive.google.com/uc?id={DF_FINAL_ID}"
    local_path = "df_final_from_gdrive.csv"


    gdown.download(url, local_path, quiet=True)


    return pd.read_csv(local_path, low_memory=False)


@st.cache_data(show_spinner="Loading final predictions from Google Drive...")
    
def load_final_predictions():
    url = f"https://drive.google.com/uc?id={PRED_ID}"
    local_path = "final_predict_2025_11_12_with_pred_from_gdrive.csv"

    try:
        gdown.download(url, local_path, quiet=True)
        df_pred = pd.read_csv(local_path)
    except Exception:
        df_pred = None

    return df_pred



@st.cache_data
def load_model_metrics():

    rows = [
        # Scenario, Model, Valid, Test
        ("Scenario1", "Logit",                  0.412, 0.388),
        ("Scenario1", "XGB",                    0.783, 0.611),

        ("Scenario2", "Logit α",                0.500, 0.450),
        ("Scenario2", "XGB α",                  0.783, 0.611),

        ("Scenario3", "XGB (native, Tuned)",    0.786, 0.612),
        ("Scenario3", "RF (Tuned)",             0.793, 0.605),
        ("Scenario3", "HGB (Tuned)",            0.794, 0.604),
        ("Scenario3", "RidgeCal",               0.397, 0.367),
        ("Scenario3", "CatBoost (Tuned)",       0.796, 0.612),

        ("Scenario4", "Ensemble - SoftVote",    0.799, 0.611),
        ("Scenario4", "Ensemble - Stacking",    0.799, 0.612),
    ]
    df = pd.DataFrame(
        rows,
        columns=["Scenario", "Model", "Macro-F1 (Valid)", "Macro-F1 (Test)"]
    )
    df["Δ (Valid−Test)"] = df["Macro-F1 (Valid)"] - df["Macro-F1 (Test)"]
    return df





df_all = load_df_final()
df_metrics = load_model_metrics()
df_pred = load_final_predictions()


st.title("Lease Renewal Prediction – Capstone Dashboard")


st.markdown(
"""
This dashboard summarizes my capstone project:

1. **EDA** on lease renewals and external housing indices  
2. **Modeling results** across four scenarios (Logit, XGB, tuned models, ensemble)  
3. **Final predictions** for contracts expiring in **Nov–Dec 2025**
"""
)


# In[8]:


# =========================
# 2. Tab configuration
# =========================
tab1, tab2, tab3 = st.tabs(["1. EDA", "2. Modeling Results", "3. Final Prediction"])

# ---------------------------------
# TAB 1: EDA
# ---------------------------------
with tab1:
    st.header("1. Exploratory Data Analysis")

    if df_all is None:
        st.warning("`df_final.csv` not found. Please place the file in the same folder.")
    else:

        candidate_targets = ["target", "rent_change"]
        tgt_col = None
        for c in candidate_targets:
            if c in df_all.columns:
                tgt_col = c
                break

        if tgt_col is None:
            st.error("No target column (`target` or `rent_change`) found in df_final.csv.")
        else:

            st.subheader("1) Overall Target Distribution")

            tgt_counts = df_all[tgt_col].value_counts(dropna=False).sort_index()
            tgt_share  = (tgt_counts / tgt_counts.sum()).round(3)


            col1, col2 = st.columns([2, 2])

            with col1:
                st.write("**Counts & Share**")
                st.dataframe(
                    pd.DataFrame({
                        "count": tgt_counts,
                        "share": tgt_share
                    }),
                    use_container_width=True
                )

            with col2:
                fig, ax = plt.subplots(figsize=(3, 2))
                sns.barplot(
                    x=tgt_counts.index.astype(str),
                    y=tgt_counts.values,
                    ax=ax
                )
                ax.set_xlabel(tgt_col)
                ax.set_ylabel("Count")
                ax.set_title("Target class counts")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=False)

            st.markdown("---")


            st.subheader("2) Target Share by Year")

            if "yyyymm" in df_all.columns:
                df_eda = df_all.copy()
                df_eda["year"] = df_eda["yyyymm"].astype(str).str[:4]

                tmp = (
                    df_eda
                    .groupby(["year", tgt_col])
                    .size()
                    .reset_index(name="cnt")
                )
                tmp["share"] = tmp["cnt"] / tmp.groupby("year")["cnt"].transform("sum")

                col3, col4 = st.columns([2, 3])

                with col3:
                    st.write("**Yearly target distribution (table)**")
                    st.dataframe(tmp.head(20), use_container_width=True)

                with col4:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.barplot(
                        data=tmp,
                        x="year",
                        y="share",
                        hue=tgt_col,
                        ax=ax
                    )
                    ax.set_ylabel("Share")
                    ax.set_title("Target share by year")
                    ax.legend(title=tgt_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=False)
            else:
                st.info("Column `yyyymm` not found in df_final.csv.")

            st.markdown("---")


            st.subheader("3) External Housing Indices (sample)")

            ext_cols = [
                "apt_sale_index",
                "apt_rent_index",
                "apt_sale_supply_index",
                "apt_rent_supply_index",
            ]
            present_ext = [c for c in ext_cols if c in df_all.columns]

            if ("yyyymm" in df_all.columns) and present_ext:

                df_ext = df_all.copy()
                df_ext["yyyymm"] = df_ext["yyyymm"].astype(str)


                df_ext["year"]  = df_ext["yyyymm"].str.slice(0, 4).astype(int)
                df_ext["month"] = df_ext["yyyymm"].str.slice(4, 6).astype(int)


                grp = (
                    df_ext
                    .groupby(["year", "month"])[present_ext]
                    .mean()
                    .reset_index()
                    .sort_values(["year", "month"])
                )
                grp["ym_str"] = grp["year"].astype(str) + "-" + grp["month"].astype(str).str.zfill(2)

                st.write("**Sample of monthly external indices**")
                st.dataframe(grp[["ym_str"] + present_ext].head(24), use_container_width=True)


                plot_cols = present_ext[:2]
                col5, _ = st.columns([2, 2])

                with col5:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    for c in plot_cols:
                        ax.plot(grp["ym_str"], grp[c], marker="o", label=c)

                    ax.set_xticks(grp["ym_str"][::6]) 
                    ax.tick_params(axis="x", rotation=45)
                    ax.set_xlabel("Year-Month")
                    ax.set_ylabel("Index Level")
                    ax.set_title("External Housing Indices Over Time")
                    ax.legend(fontsize=7)
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=False)

            else:
                st.info("`yyyymm` or external index columns not found in df_final.csv.")


# ---------------------------------
# TAB 2: Modeling Results
# ---------------------------------
with tab2:
    st.header("Modeling Results (Macro-F1)")

    st.subheader("Scenario-wise performance table")
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("### Scenario-wise Macro-F1 (Valid vs Test)")


    col_small = st.columns([1])[0]

    with col_small:

        fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=True) 
        axes = axes.flatten()

        scenarios = df_metrics["Scenario"].unique()

        for i, sc in enumerate(scenarios):
            sub = df_metrics[df_metrics["Scenario"] == sc]
            ax = axes[i]
            sub_plot = sub.melt(
                id_vars="Model",
                value_vars=["Macro-F1 (Valid)", "Macro-F1 (Test)"],
                var_name="Split",
                value_name="Macro-F1"
            )
            sns.barplot(data=sub_plot, x="Model", y="Macro-F1", hue="Split", ax=ax)
            ax.set_title(sc)
            ax.set_ylim(0.35, 0.82)
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

            if i == 0:
                ax.legend(loc="lower left", fontsize=6)
            else:
                ax.legend_.remove()

        plt.tight_layout(pad=0.7)
        st.pyplot(fig, use_container_width=False)

    st.markdown("---")
    st.subheader("Generalization gap Δ(Valid - Test)")


    df_metrics["Δ (Valid - Test)"] = (
        df_metrics["Macro-F1 (Valid)"] - df_metrics["Macro-F1 (Test)"]
    )


    col_small2 = st.columns([1])[0]

    with col_small2:
        heat = df_metrics.pivot(index="Model", columns="Scenario", values="Δ (Valid - Test)")

        fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.32 * heat.shape[0])))

        sns.heatmap(
            heat,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Δ(Valid - Test)"},
            ax=ax2
        )
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=False)

    st.markdown(
    """
    **Interpretation (short):**  
    - Most tuned models and ensembles show a noticeable drop between Valid and Test,  
      which reflects the distribution shift in 2025 data compared to 2015–2024.  
    - I still choose the tuned XGB model as my main model, and plan to keep updating it  
      as more 2025+ data become available so that it can adapt to the new regime.
    """
    )





# In[7]:


# ---------------------------------
# TAB 3: Final Prediction
# ---------------------------------
with tab3:
    st.header("Final Prediction – 2025 Nov–Dec")

    st.markdown(
    """
    Here I apply my final XGB model to contracts ending in **2025-11 ~ 2025-12**.
    """
    )

    if df_pred is None:
        st.warning("`final_predict_2025_11_12_with_pred.csv` not found. Please export your prediction file first.")
    else:
        st.subheader("Sample of prediction results")


        preview_cols = [
            c for c in df_pred.columns
            if c.lower() in ["lease_id", "ym", "yyyymm", "address_1", "pred_target"]
            or c.startswith("proba_")
        ]
        if not preview_cols:
            preview_cols = list(df_pred.columns)  # fallback

        st.dataframe(df_pred[preview_cols].head(30), use_container_width=True)

        st.markdown("### Target distribution for predicted period")


        tgt_counts = None
        tmp_region = None

        if "pred_target" in df_pred.columns:
            tgt_counts = df_pred["pred_target"].value_counts().sort_index()

        if {"pred_target", "address_1"}.issubset(df_pred.columns):
            tmp_region = (
                df_pred.groupby(["address_1", "pred_target"])
                       .size()
                       .reset_index(name="cnt")
            )
            tmp_region["share"] = (
                tmp_region["cnt"] /
                tmp_region.groupby("address_1")["cnt"].transform("sum")
            )


        if (tgt_counts is not None) and (tmp_region is not None):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Predicted target counts**")
                fig3, ax3 = plt.subplots(figsize=(3, 3))
                sns.barplot(x=tgt_counts.index, y=tgt_counts.values, ax=ax3)
                ax3.set_xlabel("Predicted target")
                ax3.set_ylabel("Count")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig3, use_container_width=False)

            with col2:
                st.markdown("**Predicted target share by region**")
                fig4, ax4 = plt.subplots(figsize=(3, 3))
                sns.barplot(
                    data=tmp_region,
                    x="address_1",
                    y="share",
                    hue="pred_target",
                    ax=ax4,
                )
                ax4.set_ylabel("Share")
                ax4.set_xlabel("Region (address_1)")
                ax4.legend(fontsize=6)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig4, use_container_width=False)

        elif tgt_counts is not None:

            fig3, ax3 = plt.subplots(figsize=(3, 3))
            sns.barplot(x=tgt_counts.index, y=tgt_counts.values, ax=ax3)
            ax3.set_xlabel("Predicted target")
            ax3.set_ylabel("Count")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig3, use_container_width=False)

        elif tmp_region is not None:

            fig4, ax4 = plt.subplots(figsize=(3, 3))
            sns.barplot(
                data=tmp_region,
                x="address_1",
                y="share",
                hue="pred_target",
                ax=ax4,
            )
            ax4.set_ylabel("Share")
            ax4.set_xlabel("Region (address_1)")
            ax4.legend(fontsize=6)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig4, use_container_width=False)

        st.markdown(
        """
        **How I intend to use this:**  
        - Monitor where the model predicts more **increase** or **decrease** in rent.  
        - Combine predictions with business rules to prioritize negotiation strategies.
        """
        )



