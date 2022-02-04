import matplotlib as mpl
from matplotlib import cm
import collections
import seaborn as sns
import numpy as np
from sklearn.preprocessing import QuantileTransformer
sns.set_style("ticks")

mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams.update({'font.size': 10})
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({"axes.grid": True})


class Colors(object):

    def __init__(self, cmap_name="Spectral", empty="lightgray"):
        self.cmap_name = cmap_name
        self.empty = empty
        self.cmap = cm.get_cmap(self.cmap_name)
        self.transformer = None
        self.set_bokeh()

    def set_bokeh(self):
        self.red = '#EC1557'
        self.orange = '#F05223'
        self.yellow = '#F6A91B'
        self.lightgreen = '#A5CD39'
        self.green = '#20B254'
        self.lightblue = '#00AAAE'
        self.blue = '#4998D3'
        self.purple = '#892889'

    def ideas(self):
        ideas = {
            "diverging": ["Spectral", "coolwarm"],
            "uniform": ["viridis", "plasma"],
            "sequential": ["YlGnBu"]
        }
        return ideas

    def sample(self, n):
        values = np.linspace(0, 1, n)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        values = [norm(x) for x in values]
        return [self.cmap(x) for x in values]

    def from_categories(self, categories, spread_by_counts=True, empty_category=-1):
        categories_counts = collections.defaultdict(int)
        for c in categories:
            categories_counts[c] += 1
        cats = sorted(categories_counts.keys())
        colors = self.sample(len(cats))
        cat2col = {}
        for cat, col in zip(cats, colors):
            if cat == empty_category:
                cat2col[cat] = self.empty
            else:
                cat2col[cat] = col
        return [cat2col[c] for c in categories]

    def from_values(self, values, method="uniform"):
        values = np.array(values).reshape(-1,1)
        if self.transformer is None:
            self.transformer = QuantileTransformer(output_distribution=method)
            self.transformer.fit(values)
            if method == "uniform":
                vmin = 0
                vmax = 1
            else:
                values = self.transformer.transform(values).ravel()
                vmin = np.percentile(values, 1)
                vmax = np.percentile(values, 99)
            self.from_values_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        values = self.transformer.transform(values).ravel()
        values = [self.from_values_norm(x) for x in values]
        colors = self.cmap(values)
        return colors