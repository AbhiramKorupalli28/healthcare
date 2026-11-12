"""Plotly charts for the dashboard: modern clean light theme with Forest Green accents."""
import plotly.express as px
import plotly.graph_objects as go
from config import THEME

PALETTE = ["#0E4D2A", "#10B981", "#059669", "#34D399", "#2563EB", "#D97706"]

def style(fig, height: int = 360):
    """Clean light SaaS style with dark text and forest green accents."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        colorway=PALETTE,
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#4B5563")),
    )
    fig.update_xaxes(
        gridcolor="#F3F4F6",
        linecolor="#E5E7EB",
        tickfont=dict(color="#4B5563"),
        title_font=dict(color="#111827")
    )
    fig.update_yaxes(
        gridcolor="#F3F4F6",
        linecolor="#E5E7EB",
        tickfont=dict(color="#4B5563"),
        title_font=dict(color="#111827")
    )
    return fig

def bar(df, x, y, title, color=None, orientation="v", text_auto=True, x_label=None, y_label=None):
    fig = px.bar(df, x=x, y=y, color=color, title=title, orientation=orientation,
                 text_auto=text_auto if text_auto else False, barmode="group",
                 labels={x: x_label or pretty_label(x), y: y_label or pretty_label(y)})
    fig.update_traces(
        marker_color="#0E4D2A" if color is None else None,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color="#111827", size=11)
    )
    fig.update_yaxes(tickformat=",")
    return style(fig)

def pretty_label(name: str) -> str:
    return str(name).replace("_", " ").capitalize()

def histogram(series, title, nbins=30, x_title=""):
    fig = px.histogram(x=series, nbins=nbins, title=title, labels={"x": x_title})
    fig.update_traces(marker_color="#0E4D2A")
    return style(fig)

def heatmap(matrix, labels, title, x_title="Predicted", y_title="Actual"):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels,
                               colorscale=[[0, "#EBF5EE"], [1, "#0E4D2A"]],
                               text=matrix, texttemplate="%{text}", showscale=False))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, height=340)

def importance_bar(imp_df, title, top=10):
    d = imp_df.head(top).iloc[::-1]
    fig = px.bar(d, x="importance", y="feature", orientation="h", title=title)
    fig.update_traces(marker_color="#10B981")
    return style(fig, height=max(300, 26 * len(d) + 60))
