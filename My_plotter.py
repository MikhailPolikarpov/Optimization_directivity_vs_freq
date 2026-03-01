
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Style:
    ax_label_size: int = 14
    tick_label_size: int = 12
    legend_size: int = 10
    grid_alpha: float = 0.3
    grid_linestyle: str = '--'
    grid_linewidth: int = 1 
    line_width: int = 2.5
    line_style: str = '-'
    cmap: str = 'viridis'

class Plotter:
    def __init__(self, ax, style: Style):
        self.ax = ax
        self.style = style
    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel, fontsize=self.style.ax_label_size)
        return self
    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel, fontsize=self.style.ax_label_size)
        return self
    def set_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        return self
    def set_title(self, title):
        self.ax.set_title(title, fontsize=self.style.ax_label_size)
        return self
    def set_legend(self, legend_loc = 'best'):
        self.legend_loc = legend_loc
        return self
    def set_p(self, p):
        self.p = p
        self.cmap = plt.get_cmap(self.style.cmap)
        self.norm = plt.Normalize(min(self.p), max(self.p))
        return self
    
    def plot(self, x, y, label='', linestyle=None, line_width=None):
        lw = line_width if line_width is not None else self.style.line_width
        ls = linestyle if linestyle is not None else self.style.line_style
        self.ax.plot(x, y, label=label, linewidth=lw, linestyle=ls)

    def multiple_plot(self, xi, yi, p_i, label=''):
        self.ax.plot(xi, yi, label=label, color=self.cmap(self.norm(p_i)), linewidth=self.style.line_width, linestyle=self.style.line_style)

    def finalize(self):
        self.ax.tick_params(axis='both', which='major', labelsize=self.style.tick_label_size)
        self.ax.grid(alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle, linewidth=self.style.grid_linewidth)
        
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            if not hasattr(self, 'legend_loc'):
                self.legend_loc = 'best'
            self.ax.legend(fontsize=self.style.legend_size, loc=self.legend_loc)