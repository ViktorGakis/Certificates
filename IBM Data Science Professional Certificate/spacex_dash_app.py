import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px


spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df["Payload Mass (kg)"].max()
min_payload = spacex_df["Payload Mass (kg)"].min()


app = dash.Dash(__name__)
server = app.server

dropmenu = [{"label": "All Sites", "value": "All Sites"}] + [
    {"label": x, "value": x} for x in spacex_df["Launch Site"].unique().tolist()
]


app.layout = html.Div(
    children=[
        html.H1(
            "SpaceX Launch Records Dashboard",
            style={"textAlign": "center", "color": "#503D36", "font-size": 40},
        ),
        dcc.Dropdown(
            id="site_dropdown",
            options=dropmenu,
            placeholder="Select a Launch Site here",
            searchable=True,
            value="All Sites",
        ),
        html.Br(),
        html.Div(dcc.Graph(id="success-pie-chart")),
        html.Br(),
        html.P("Payload range (Kg):"),
        dcc.RangeSlider(
            id="payload_slider",
            min=0,
            max=10000,
            step=1000,
            marks={i: f"{i} kg" for i in range(0, 10000, 1000)},
            value=[min_payload, max_payload],
        ),
        html.Div(dcc.Graph(id="success-payload-scatter-chart")),
    ]
)


@app.callback(
    Output(component_id="success-pie-chart", component_property="figure"),
    [Input(component_id="site_dropdown", component_property="value")],
)
def update_pie(site_dropdown):
    if site_dropdown == "All Sites":
        df = spacex_df[spacex_df["class"] == 1]
        fig = px.pie(
            df, names="Launch Site", title="Total Success Launches By all sites"
        )
    else:
        df = spacex_df.loc[spacex_df["Launch Site"] == site_dropdown]
        fig = px.pie(
            df, names="class", title="Total Success Launches for site " + site_dropdown
        )
    return fig


@app.callback(
    Output(component_id="success-payload-scatter-chart", component_property="figure"),
    [
        Input(component_id="site_dropdown", component_property="value"),
        Input(component_id="payload_slider", component_property="value"),
    ],
)
def update_scatter(site_dropdown, payload_slider):
    if site_dropdown == "All Sites":
        low, high = payload_slider
        df = spacex_df
        mask = (df["Payload Mass (kg)"] > low) & (df["Payload Mass (kg)"] < high)
        fig = px.scatter(
            df[mask],
            x="Payload Mass (kg)",
            y="class",
            color="Booster Version",
            size="Payload Mass (kg)",
            hover_data=["Payload Mass (kg)"],
        )
    else:
        low, high = payload_slider
        df = spacex_df.loc[spacex_df["Launch Site"] == site_dropdown]
        mask = (df["Payload Mass (kg)"] > low) & (df["Payload Mass (kg)"] < high)
        fig = px.scatter(
            df[mask],
            x="Payload Mass (kg)",
            y="class",
            color="Booster Version",
            size="Payload Mass (kg)",
            hover_data=["Payload Mass (kg)"],
        )
    return fig


if __name__ == "__main__":
    app.run_server(debug=False)
