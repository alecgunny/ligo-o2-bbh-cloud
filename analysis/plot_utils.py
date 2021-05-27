import glob
import os
import pickle
import sys
from functools import wraps, lru_cache

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2_6 as palette
from bokeh.plotting import figure


# need to insert the path so that we can unpickle
# our config objects
sys.path.insert(0, "..")
output_notebook()


def make_run_dir(run_id):
    return os.path.join("data", run_id)


run_dirs = glob.glob(make_run_dir("*"))
configs = []
for d in run_dirs:
    try:
        with open(os.path.join(d, "config.pkl"), "rb") as f:
            configs.append(pickle.load(f))
    except FileNotFoundError:
        continue


@lru_cache(10)
def load_stats_for_config(config):
    run_dir = make_run_dir(config.id)
    df = pd.read_csv(os.path.join(run_dir, "server-stats.csv"))

    df["average_time"] = df["time (us)"] / df["count"]
    df["throughput"] = df["count"] / df["interval"]

    ds = []
    for stuff, d in df.groupby(["ip", "gpu_id", "model", "process"]):
        d["time_since_start"] = d.interval.cumsum()
        ds.append(d)
    df = pd.concat(ds, axis=0, ignore_index=True)
    return df


def _has_value(config, key, value):
    try:
        return config.__dict__[key] == value
    except KeyError:
        raise ValueError(f"Invalid config parameter {key}")


def get_configs(**kwargs):
    relevant = [config for config in configs]
    for key, value in kwargs.items():
        relevant = [
            config for config in relevant if _has_value(config, key, value)
        ]
    return relevant


def df_or_config(f):
    @wraps(f)
    def wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            df = load_stats_for_config(df)
        return f(df, *args, **kwargs)
    return wrapper


@df_or_config
def get_num_inferences(df):
    print(
        df[(df.process == "request") & (df.model == "bbh")]["count"].sum()
    )


def make_figure(title, x_axis_label, y_axis_label, **kwargs):
    p = figure(
        height=400,
        width=800,
        title=title,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        **kwargs
    )
    p.outline_line_color = None
    p.toolbar_location = None
    p.background_fill_color = "#efefef"
    return p


def make_step_hist(grouped, bins):
    try:
        x = [bins[0]]
    except TypeError:
        max_, min_ = grouped.agg("max").max(), grouped.agg("min").min()
        bins = np.linspace(min_, max_, bins)
        x = [bins[0]]

    for b0, b1 in zip(bins[:-1], bins[1:]):
        x.extend([b0, b1])
    x.append(bins[-1])

    output = {"x": x}
    for name, x in grouped:
        hist, _ = np.histogram(x, bins)
        y = [0]
        for h in hist:
            y.extend([h, h])
        y.append(0)
        output[name] = y
    return ColumnDataSource(output)


@df_or_config
def plot_queue_histograms(df, bins=50):
    queue = df[df.process == "queue"][["model", "average_time"]]
    queue["average_time"] *= 10**-6
    tmin, tmax = queue["average_time"].min(), queue["average_time"].max()
    bins = np.geomspace(tmin, tmax, bins)

    p = make_figure(
        title="Queue Times Per Model",
        x_axis_label="Time (s)",
        y_axis_label="Count",
        x_axis_type="log"
    )
    source = make_step_hist(
        queue.groupby("model")["average_time"], bins
    )
    for model, color in zip(df.model.unique(), palette):
        p.line(
            x="x",
            y=model,
            line_width=1.8,
            line_alpha=0.8,
            line_color=color,
            legend_label=model,
            source=source
        )
    show(p)


@df_or_config
def plot_throughput_histograms(df, aggregate=False, bins=50):
    df = df[df.process == "request"]
    models = df.model.unique()

    if not aggregate:
        df = df.groupby("model")["throughput"]
        x_axis_label = "Throughput (frames / s)"
    else:
        df = df.groupby(["model", "step"])["throughput"].agg("sum")
        df = df.reset_index().groupby("model")["throughput"]
        x_axis_label = "Aggregate throughput (frames / s)"

    p = make_figure(
        title="Throughput Breakdowns by Model",
        x_axis_label=x_axis_label,
        y_axis_label="Count"
    )
    source = make_step_hist(df, bins)
    for model, color in zip(models, palette):
        p.line(
            x="x",
            y=model,
            line_width=1.8,
            line_alpha=0.8,
            line_color=color,
            legend_label=model,
            source=source
        )
    show(p)


@df_or_config
def plot_throughput_vs_time(df, models=None):
    if isinstance(models, str):
        models = [models]
    elif models is None:
        models = df.model.unique()

    mask = df.process == "request"
    model_mask = False
    for model in models:
        model_mask |= df.model == model
    mask &= model_mask

    p = make_figure(
        title="Throughput vs. Time",
        x_axis_label="Time (s)",
        y_axis_label="Aggregate throughput (frames / s)",
    )
    for color, (model, d) in zip(palette, df[mask].groupby("model")):
        grouped = d.groupby("step")
        p.line(
            x=grouped["time_since_start"].agg("mean"),
            y=grouped["throughput"].agg("sum"),
            line_width=1.8,
            line_alpha=0.7,
            line_color=color,
            legend_label=model
        )
    show(p)


cost_per_n1_cpu_per_hour = 0.04749975
cpus_per_client = 8
cost_per_gpu_per_hour = 0.35


def map_to_cost(seconds_per_second, config):
    cost_per_server_cpus = cost_per_n1_cpu_per_hour * config.vcpus_per_gpu
    cost_per_gpu = cost_per_gpu_per_hour + cost_per_server_cpus
    cost_per_server = config.gpus_per_node * cost_per_gpu

    cost_per_client = cost_per_n1_cpu_per_hour * cpus_per_client
    client_costs_per_server = cost_per_client * config.clients_per_node

    total_cost = (cost_per_server + client_costs_per_server) * config.num_nodes
    cost_in_usd = seconds_per_second * total_cost / 3600
    cost_per_cpu_hour = cost_in_usd / cost_per_n1_cpu_per_hour
    return cost_per_cpu_hour


def make_violin_patch(config, y_axis=None, percentile=5):
    df = load_stats_for_config(config)
    df = df[(df.process == "request") & (df.model == "bbh")]
    inferences_per_second = df.groupby("step")["throughput"].agg("sum")

    y_time = 1 / (inferences_per_second * config.kernel_stride)
    y_cost = map_to_cost(y_time, config)

    if y_axis is None:
        ys = [y_time, y_cost]
    elif y_axis == "time":
        ys = [y_time]
    elif y_axis == "cost":
        ys = [y_cost]
    else:
        raise ValueError(f"Can't plot y-axis {y_axis}")

    outputs = []
    for metric in ys:
        min_, max_ = np.percentile(metric, [percentile, 100 - percentile])
        diff = (max_ - min_) / (101)

        observations = metric[(metric >= min_) & (metric <= max_)]
        kernel = gaussian_kde(observations)
        y = np.linspace(min_ - diff, max_ + diff, 102)
        x = kernel(y)
        x[0] = x[-1] = 0
        x /= x.max() * 1.05 * 2
        outputs.append([x, y])

    if len(outputs) == 1:
        x, y = outputs[0]
        x = list(x) + list(-x[::-1])
        y = list(y) + list(y[::-1])
        return x, y

    outputs = [(x, y) for x, y in outputs]
    outputs = [(x * (-1)**(i + 1), y) for i, (x, y) in enumerate(outputs)]
    return outputs
