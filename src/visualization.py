"""Plotly charts for the dashboard and matplotlib PNGs for the visual report.

Every chart takes data that the caller computed from the records; nothing here
creates numbers of its own.
"""
import plotly.express as px
import plotly.graph_objects as go

from config import THEME

PALETTE = [THEME["violet"], THEME["blue"], THEME["lilac"], THEME["green"], THEME["amber"], THEME["red"]]


def style(fig, height: int = 380):
    """Navy background, light text, violet/blue accents."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"], family="Inter, Segoe UI, sans-serif"),
        colorway=PALETTE, height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(154,164,199,0.15)", zerolinecolor="rgba(154,164,199,0.2)")
    fig.update_yaxes(gridcolor="rgba(154,164,199,0.15)", zerolinecolor="rgba(154,164,199,0.2)")
    return fig


def bar(df, x, y, title, color=None, orientation="v", text_auto=True, x_label=None, y_label=None):
    fig = px.bar(df, x=x, y=y, color=color, title=title, orientation=orientation,
                 text_auto=text_auto if text_auto else False, barmode="group",
                 labels={x: x_label or pretty_label(x), y: y_label or pretty_label(y)})
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=THEME["text"])
    fig.update_yaxes(tickformat=",")
    return style(fig)


def pretty_label(name: str) -> str:
    return str(name).replace("_", " ").capitalize()


def histogram(series, title, nbins=30, x_title=""):
    fig = px.histogram(x=series, nbins=nbins, title=title, labels={"x": x_title})
    fig.update_traces(marker_color=THEME["violet"])
    return style(fig)


def heatmap(matrix, labels, title, x_title="Predicted", y_title="Actual"):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels,
                               colorscale=[[0, THEME["panel2"]], [1, THEME["violet"]]],
                               text=matrix, texttemplate="%{text}", showscale=False))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, height=360)


def importance_bar(imp_df, title, top=12):
    d = imp_df.head(top).iloc[::-1]
    fig = px.bar(d, x="importance", y="feature", orientation="h", title=title)
    fig.update_traces(marker_color=THEME["blue"])
    return style(fig, height=max(320, 26 * len(d) + 80))


def rate_bars(df, group, title):
    fig = px.bar(df, x=group, y="positive_pct", color="condition" if "condition" in df else None,
                 title=title, barmode="group", labels={"positive_pct": "Positive (%)"})
    return style(fig)


def save_png_bar(df, x, y, title, path):
    """Matplotlib PNG for the visual report (same palette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["panel"])
    ax.bar(df[x].astype(str), df[y], color=THEME["violet"])
    ax.set_title(title, color=THEME["text"])
    ax.tick_params(colors=THEME["text"], rotation=30)
    for s in ax.spines.values():
        s.set_color(THEME["muted"])
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
