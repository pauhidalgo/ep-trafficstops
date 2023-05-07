from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st
from abc import ABC
from typing import List, Dict, Optional
import calendar
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go
from tokens import mapbox_token
from collections import Counter


class BiasPlots(ABC):
    def __init__(self, data, city):
        self.data = data
        self.city = city

        # Add indicator columns for each race
        self.data = self.data.loc[
            (~self.data["subject_race"].isna())
            & (~self.data["subject_race"].isin(["other", "unknown"]))
        ]

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
        px.set_mapbox_access_token(mapbox_token)
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
            title="Traffic Stops over Time",
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
