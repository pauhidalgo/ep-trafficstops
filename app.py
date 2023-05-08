from datetime import datetime

import pandas as pd
import streamlit as st

from analysis import GeoPlots, TimePlots, BiasPlots

cities = {
    "New Orleans, LA": "la_new_orleans_2020_04_01.parquet",
    "San Antonio, TX": "tx_san_antonio_2023_01_26.parquet",
    "Vermont State Patrol": "vt_statewide_2020_04_01.parquet",
}


def prep_data(city, date_range):
    data = pd.read_parquet(f"data/{cities[city]}")
    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True)
    data["time"] = pd.to_datetime(data["time"], infer_datetime_format=True)
    data["year"] = data["date"].dt.year
    data["weekday"] = data["date"].dt.weekday.apply(float)
    data["hour"] = data["time"].dt.hour.astype(float)
    # Special city cases
    if city == "New Orleans, LA" or city == "Vermont State Patrol":
        data = data.loc[data["date"] > datetime(2010, 12, 31)]

    # Filter to user specified date range
    data = data.loc[(data["date"] >= date_range[0]) & (data["date"] <= date_range[1])]
    if len(data) < 1:
        raise ValueError(
            "There is no available data for the dates you selected. Please try another date range."
        )
    date_range_str = f"{data['date'].min().strftime('%b %Y')} to {data['date'].max().strftime('%b %Y')}"
    disp_str = f"The following data for {city} includes **{len(data):,.0f}** traffic stops from **{date_range_str}**."
    st.markdown(disp_str)
    return data


# Build application
st.set_page_config(page_title="Final Project", layout="wide")
st.markdown("# Exploring Police Traffic Stops")
st.markdown("Pauline Hidalgo | Data Visualization Spring 2023")

# Define form for user input options (city and date range)
with st.sidebar:
    st.markdown("# Options")
    form = st.form("my_form")
    city = form.selectbox(
        "Which location would you like to visualize?", list(cities.keys())
    )
    date_range = form.slider(
        "Date Range",
        min_value=datetime(2005, 1, 1),
        max_value=datetime(2023, 5, 1),
        value=(datetime(2005, 1, 1), datetime(2023, 5, 1)),
    )
    form.form_submit_button("Analyze")
    plot_data = prep_data(city, date_range)
    st.markdown(
        f"Data from the [Stanford Open Policing Project](https://openpolicing.stanford.edu/)"
    )
    st.markdown("")
    st.markdown("**References**")
    st.markdown(
        "Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017, August). "
        "Algorithmic decision making and the cost of fairness."
    )
    st.markdown(
        "Pierson, E., Simoiu, C., Overgoor, J., Corbett-Davies, S., Jenson, D., Shoemaker, A., ... & Goel, "
        "S. (2020). A large-scale analysis of racial disparities in police stops across the United States."
    )
    st.markdown(
        "Simoiu, C., Corbett-Davies, S., & Goel, S. (2017). The problem of infra-marginality in outcome "
        "tests for discrimination."
    )

st.markdown("## Data summary")
tab1, tab2 = st.tabs(["**Geography**", "**Temporal Patterns**"])
with tab1:
    st.markdown(
        "Traffic stops tend to concentrate in city centers and extend out along main roads:"
    )
    gp = GeoPlots(plot_data, city)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gp.bubble_map(), use_container_width=True)
    with col2:
        st.plotly_chart(gp.scatter_map(), use_container_width=True)

with tab2:
    st.markdown("The number of traffic stops made trends downwards over time: ")
    tp = TimePlots(plot_data, city)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(tp.yearly_plot(), use_container_width=True)
    with col2:
        st.plotly_chart(tp.yearly_bars(), use_container_width=True)
    st.markdown("There is generally less policing on weekends: ")
    st.plotly_chart(tp.heatmap(), use_container_width=True)

st.markdown("## Decision making and Bias")
bp = BiasPlots(plot_data, city)
st.markdown("### Benchmarking")
st.markdown(
    "Benchmarking is a straightforward test that compares the rates at which different demographics are searched. "
    "Black drivers are often searched at the highest rates."
)
st.plotly_chart(bp.benchmark_plot(), use_container_width=True)
st.markdown("### Outcome test")
st.markdown(
    "The outcome test specifically targets the 'hit rate' of searches, "
    "i.e. whether contraband was found. "
    "When searches for non-white drivers have lower hit rates compared to white drivers, this suggests officers tend "
    "to search those groups with less evidence. "
    "The search hit rates for hispanic drivers are often lower than for white drivers, suggesting hispanic drivers "
    "face increased discrimination."
)
st.plotly_chart(bp.outcome_plot(), use_container_width=True)

st.markdown("### Algorithmic fairness")
st.markdown(
    "Algorithmic fairness is an area of research concerned with how to correct biases in decisions made by machine "
    "learning models. "
    "The table below shows the features and resulting coefficients of a logistic regression classifier on whether a "
    "search will find contraband based on historical data."
)
st.dataframe(bp.get_coeffs())
st.markdown(
    "Popular definitions of algorithmic fairness include statistical parity and predictive equality. "
    "The following sections will apply these metrics to the training set to explore the classifier's fairness with "
    "respect to the race of the stopped person."
)

st.markdown("#### Statistical Parity")
st.markdown(
    "Statistical parity refers to when the class proportions across groups are the same. "
)
st.markdown(
    "When machine learning models are trained on biased data, those biases often become exacerbated, "
    "as illustrated in the plots below."
)
clf_fig, actual_fig = bp.statistical_parity_plots()
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(clf_fig, use_container_width=True)
with col2:
    st.plotly_chart(actual_fig, use_container_width=True)

st.markdown("#### Predictive Equality")
st.markdown(
    "Predictive equality is achieved when the decision accuracy (e.g. false positive rate) is the same across "
    "groups."
)
st.markdown("##### ROC Curve")
st.markdown(
    "The Receiver Operating Characteristic, or ROC curve, plots the true positive rate vs. the false "
    "positive rate to indicate the performance of a classifier compared to random guessing "
    "(indicated by the y = x line)."
)

st.plotly_chart(bp.roc_plot(), use_container_width=True)

st.markdown("##### Confusion Matrices")
st.markdown(
    "Confusion matrices summarize the performance of a classifier as a contingency table of the actual "
    "labels vs. what was predicted. "
)
st.markdown(
    "The top right entry, for example, indicates the false positive rate: the proportion of cases that were "
    "predicted to have contraband, but actually did not. "
    "It is often the case that algorithms have higher false positive rates for minority groups, "
    "such as black and hispanic drivers."
)
fig0, fig1, fig2, fig3 = bp.confusion_matrix_plots()
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig0, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
