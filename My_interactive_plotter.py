import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from dataclasses import dataclass


@dataclass
class PlotlyStyle:
    ax_label_size: int = 18
    tick_label_size: int = 18
    legend_size: int = 14

    line_width: float = 4
    line_style: str = "solid"

    grid: bool = True

    cmap: str = "Viridis"

    width: int | int = 1000
    height: int |int = 600


class PlotlyPlotter:
    def __init__(self, style: PlotlyStyle):
        self.style = style
        self.fig = go.Figure()

        self.p = None
        self.p_min = None
        self.p_max = None

        self.legend_loc = None

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    def set_xlabel(self, xlabel):
        self.fig.update_xaxes(
            title_text=xlabel,
            title_font=dict(size=self.style.ax_label_size),
        )
        return self

    def set_ylabel(self, ylabel):
        self.fig.update_yaxes(
            title_text=ylabel,
            title_font=dict(size=self.style.ax_label_size),
        )
        return self

    def set_xlim(self, xlim):
        self.fig.update_xaxes(range=xlim)
        return self

    def set_ylim(self, ylim):
        self.fig.update_yaxes(range=ylim)
        return self

    def set_title(self, title):
        self.fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=self.style.ax_label_size),
            )
        )
        return self

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    def set_legend(self, legend_loc="top right"):
        self.legend_loc = legend_loc
        return self

    # --------------------------------------------------
    # Parameter colormap
    # --------------------------------------------------

    def set_p(self, p):
        self.p = np.asarray(p)

        self.p_min = np.min(self.p)
        self.p_max = np.max(self.p)

        return self

    def _get_color(self, p_i):
        if self.p is None:
            return None

        if self.p_max == self.p_min:
            value = 0.5
        else:
            value = (p_i - self.p_min) / (self.p_max - self.p_min)

        return px.colors.sample_colorscale(
            self.style.cmap,
            [value],
        )[0]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    def plot(
        self,
        x,
        y,
        label="",
        linestyle=None,
        line_width=None,
    ):
        lw = (
            line_width
            if line_width is not None
            else self.style.line_width
        )

        ls = (
            linestyle
            if linestyle is not None
            else self.style.line_style
        )

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line=dict(
                    width=lw,
                    dash=ls,
                ),
                showlegend=bool(label),
            )
        )

        return self

    def multiple_plot(
        self,
        xi,
        yi,
        p_i,
        label="",
        linestyle=None,
        line_width=None,
    ):
        lw = (
            line_width
            if line_width is not None
            else self.style.line_width
        )

        ls = (
            linestyle
            if linestyle is not None
            else self.style.line_style
        )

        color = self._get_color(p_i)

        self.fig.add_trace(
            go.Scatter(
                x=xi,
                y=yi,
                mode="lines",
                name=label,
                line=dict(
                    color=color,
                    width=lw,
                    dash=ls,
                ),
                showlegend=bool(label),
            )
        )

        return self

    # --------------------------------------------------
    # Finalize
    # --------------------------------------------------

    def finalize(self):
        self.fig.update_xaxes(
            tickfont=dict(size=self.style.tick_label_size),
            showgrid=self.style.grid,
        )

        self.fig.update_yaxes(
            tickfont=dict(size=self.style.tick_label_size),
            showgrid=self.style.grid,
        )

        self.fig.update_layout(
            width=self.style.width,
            height=self.style.height,

            legend=dict(
                font=dict(size=self.style.legend_size),
            ),

            template="plotly_white",

            margin=dict(
                l=70,
                r=30,
                t=60,
                b=60,
            ),
        )

        self._apply_legend_position()

        return self

    def _apply_legend_position(self):
        positions = {
            "top right": dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
            ),
            "top left": dict(
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
            "bottom right": dict(
                x=1,
                y=0,
                xanchor="right",
                yanchor="bottom",
            ),
            "bottom left": dict(
                x=0,
                y=0,
                xanchor="left",
                yanchor="bottom",
            ),
        }

        if self.legend_loc is not None:
            position = positions.get(self.legend_loc)

            if position is not None:
                self.fig.update_layout(
                    legend=position
                )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    def show(self):
        self.finalize()
        self.fig.show(config={'scrollZoom': True})

    def get_figure(self):
        self.finalize()
        return self.fig