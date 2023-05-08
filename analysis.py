import calendar
from abc import ABC
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

import streamlit as st


class BiasPlots(ABC):
    def __init__(self, data, city):
        self.data = data
        self.city = city

        self.data = self.data.loc[
            (~self.data["subject_race"].isna())
            & (~self.data["subject_race"].isin(["other", "unknown"]))
        ]

        # Train a binary classifier based on past search decisions and outcomes
        # Add indicator columns for each race category
        self.races = [
            "black",
            "white",
            "hispanic",
            "asian/pacific islander",
        ]
        for race in self.races:
            self.data[f"subject_is_{race}"] = self.data["subject_race"] == race

        self.feature_cols = [
            "hour",
            "subject_age",
        ]
        self.feature_cols.extend([f"subject_is_{r}" for r in self.races])
        self.pred_col = "contraband_found"

        self.plot_data = self.data.loc[
            :, self.feature_cols + [self.pred_col, "subject_race"]
        ].dropna()
        self.train_data = self.plot_data.drop(columns=["subject_race"])
        for c in self.feature_cols:
            self.train_data[c] = self.train_data[c].astype(int)

        X, y = self.get_xy()
        mod = LogisticRegression(random_state=0, class_weight="balanced")
        self.mod = mod.fit(X, y)

    def get_xy(self):
        return self.train_data.loc[:, self.feature_cols].values, self.train_data[
            self.pred_col
        ].astype(int)

    def get_coeffs(self):
        params = pd.DataFrame(
            dict(zip(self.feature_cols, self.mod.coef_[0])), index=["coefficients"]
        )
        return params

    def statistical_parity_plots(self):
        # Plot stacked bar chart of search decision proportions per race
        X, y_true = self.get_xy()
        plot_data = self.plot_data.copy()
        plot_data["predict_yes"] = self.mod.predict(X)
        plot_data["actual_yes"] = y_true

        plot_data = (
            plot_data.groupby("subject_race")[["predict_yes", "actual_yes"]]
            .mean()
            .reset_index()
        )
        for yes_col, no_col in [
            ("predict_yes", "predict_no"),
            ("actual_yes", "actual_no"),
        ]:
            plot_data[no_col] = 1 - plot_data[yes_col]
        plot_data.sort_values(by=["subject_race"], inplace=True)

        clf_fig = px.bar(
            plot_data,
            x="subject_race",
            y=[
                "predict_no",
                "predict_yes",
            ],
            title="Proportion of Searches Finding Contraband (Predicted)",
        )
        clf_fig.update_layout(xaxis_title="Subject Race", yaxis_title="Proportion")
        actual_fig = px.bar(
            plot_data,
            x="subject_race",
            y=[
                "actual_no",
                "actual_yes",
            ],
            title="Proportion of Searches Finding Contraband (Actual)",
        )
        actual_fig.update_layout(xaxis_title="Subject Race", yaxis_title="Proportion")

        return clf_fig, actual_fig

    def roc_plot(self):
        X, y_true = self.get_xy()
        y_score = self.mod.decision_function(X)
        plot_data = pd.DataFrame()
        for race in self.races:
            race_idx = self.plot_data["subject_race"] == race
            fpr, tpr, _ = metrics.roc_curve(y_true[race_idx], y_score[race_idx])
            df = pd.DataFrame()
            df["False Positive Rate"] = fpr
            df["True Positive Rate"] = tpr
            df["race"] = race
            plot_data = pd.concat([plot_data, df], ignore_index=True)

        # Add special y = x case corresponding to random guessing performance
        df = pd.DataFrame()
        df["False Positive Rate"] = [0, 0.5, 1]
        df["True Positive Rate"] = [0, 0.5, 1]
        df["race"] = "y=x"
        plot_data = pd.concat([plot_data, df], ignore_index=True)

        plot_data.sort_values(by=["False Positive Rate", "race"], inplace=True)

        fig = px.line(
            plot_data, x="False Positive Rate", y="True Positive Rate", color="race"
        )
        fig.update_layout(
            showlegend=True,
            title=f"ROC Curves by Race",
            yaxis_title="True Positive Rate",
        )
        return fig

    def confusion_matrix_plots(self):
        X, y_true = self.get_xy()
        y_score = self.mod.predict(X)
        figs = []
        labels = ["Negative", "Positive"]
        for race in self.races:
            race_idx = self.plot_data["subject_race"] == race
            cm = metrics.confusion_matrix(
                y_true[race_idx], y_score[race_idx], normalize="all"
            )
            cm = np.around(cm, 2)
            fig = px.imshow(
                cm,
                text_auto=True,
                zmin=0,
                zmax=1,
                labels=dict(color="Proportion"),
                x=["Predicted " + c for c in labels],
                y=["Actual " + c for c in labels],
                title=f"Subject Race = {race}",
            )
            fig.update_xaxes(side="top")
            figs.append(fig)
        return figs

    def benchmark_plot(self):
        plot_data = (
            self.data.groupby(["year", "subject_race"])["search_conducted"]
            .mean()
            .reset_index()
        )
        plot_data.sort_values(by=["year", "subject_race"], inplace=True)
        # Convert rate to searches per 100
        plot_data["search_conducted"] = plot_data["search_conducted"] * 100
        fig = px.line(plot_data, x="year", y="search_conducted", color="subject_race")
        fig.update_layout(
            showlegend=True,
            title=f"Search Rates by Race - {self.city} ",
            xaxis_title="Year",
            yaxis_title="Searches per 100 Stops",
        )
        return fig

    def outcome_plot(self):
        plot_data = self.data.loc[self.data["search_conducted"]]
        plot_data = (
            plot_data.groupby(["year", "subject_race"])["contraband_found"]
            .mean()
            .reset_index()
        )
        plot_data.sort_values(by=["year", "subject_race"], inplace=True)
        # Convert rate to hits per 100
        plot_data["contraband_found"] = plot_data["contraband_found"] * 100
        fig = px.line(plot_data, x="year", y="contraband_found", color="subject_race")
        fig.update_layout(
            showlegend=True,
            title=f"Search Hit Rates by Race - {self.city} ",
            xaxis_title="Year",
            yaxis_title="Hits per 100 Searches",
        )
        return fig


class GeoPlots(ABC):
    def __init__(self, data, city):
        self.data = data
        self.city = city

        self.data = self.data.rename(columns={"lng": "lon"})
        self.data["lon"] = self.data["lon"].astype(float)
        self.data = self.data.loc[
            (~self.data["lat"].isna()) & (~self.data["lon"].isna())
        ]
        px.set_mapbox_access_token(st.secrets["mapbox_token"])
        self.mapbox_style = "light"

    def bubble_map(self):
        # Use K-means clustering to find centroids for bubbles
        X = self.data.loc[:, ["lat", "lon"]]
        kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto").fit(X)
        cluster_data = pd.DataFrame(kmeans.cluster_centers_).rename(
            columns={0: "lat", 1: "lon"}
        )
        counts = sorted(Counter(kmeans.labels_).items(), key=lambda item: item[0])
        cluster_data["Number of Stops"] = [p[1] for p in counts]

        fig = px.scatter_mapbox(
            cluster_data,
            lat="lat",
            lon="lon",
            size="Number of Stops",
            zoom=10,
            opacity=0.5,
            mapbox_style=self.mapbox_style,
            size_max=100,
        )
        fig.update_layout(
            height=1000,
            title="Traffic Stops Bubble Map",
        )
        return fig

    def scatter_map(self):
        plot_data = self.data
        if len(plot_data) > 200000:  # resample large datasets to preserve memory
            plot_data = self.data.sample(n=200000, random_state=1)
        fig = px.scatter_mapbox(
            plot_data,
            lat="lat",
            lon="lon",
            color="outcome",
            opacity=0.5,
            zoom=10,
            mapbox_style=self.mapbox_style,
        )
        fig.update_layout(
            height=1000,
            title="Traffic Stops Scatter Plot",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig


class TimePlots(ABC):
    def __init__(self, data, city):
        self.data = data
        self.city = city

    def yearly_bars(self):
        plot_data = self.data.groupby("year")["raw_row_number"].count().reset_index()
        fig = px.bar(
            plot_data,
            x="year",
            y="raw_row_number",
        )
        fig.update_layout(
            showlegend=True,
            title="Total Stops per Year",
            xaxis_title="Year",
            yaxis_title="Number of stops",
        )
        return fig

    def yearly_plot(self):
        plot_data = self.data.groupby("date")["raw_row_number"].count().reset_index()

        plot_data["12-week rolling average"] = (
            plot_data["raw_row_number"].rolling(12 * 7, center=True).mean()
        )
        fig = px.line(
            plot_data,
            x="date",
            y=["12-week rolling average"],
        )
        fig.update_layout(
            showlegend=True,
            title="Average Daily Traffic Stops over Time",
            xaxis_title="Date",
            yaxis_title="Number of stops",
            legend=dict(
                orientation="h",
            ),
        )
        return fig

    def heatmap(self):
        self.data["one"] = 1
        pivot = pd.pivot_table(
            self.data, values="one", index="hour", columns="weekday", aggfunc=sum
        )
        fig = px.imshow(
            pivot.values,
            aspect="auto",
            x=list(calendar.day_name),
            labels=dict(x="Day of Week", y="Hour of Day", color="Number of Stops"),
            title="Traffic Stops by Weekday and Time",
        )
        return fig
