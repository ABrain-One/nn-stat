import os
import matplotlib.pyplot as plt
import ab.nn.api as api
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.image as mpimg
import math
from scipy.stats import t




OUTDIR = "ab/stat/docs/figures"

#to load the data
def load_data():
    df = api.data()
    return df

#to print all the available tasks
def print_tasks(df):
    print("Available tasks:")
    for t in sorted(df["task"].astype(str).unique()):
        print(" -", t)

#to compute best accuracy and best epoch
def compute_best_runs(df):
    df = df.copy()
    df["prm_str"] = df["prm"].astype(str)

    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    idx = df.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
    best = df.loc[idx].copy().rename(
        columns={"accuracy": "best_accuracy", "epoch": "best_epoch"}
    )
    return best

#to print available tasks and used datasets
def datasets_per_task(df):
    datasets_per_task = df.groupby("task")["dataset"].nunique()
    print(datasets_per_task)
    for task, sub in df.groupby("task"):
        datasets = sorted(sub["dataset"].astype(str).unique())
        print(f"Task: {task}: {len(datasets)} datasets")
        for d in datasets:
            print(f"  - {d}")

#save dataset information as table
def save_tasks_datasets_table_markdown(df, outpath="ab/stat/docs/tasks_datasets_table.md"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    rows = []
    for task, sub in df.groupby("task"):
        datasets = sorted(sub["dataset"].astype(str).unique())
        rows.append((task, len(datasets), datasets))

    rows.sort(key=lambda r: r[1], reverse=True)

    lines = []
    lines.append("| Task | Number of Datasets | Datasets |")
    lines.append("|------|--------------------|----------|")

    for task, n, datasets in rows:
        ds_text = "<br>".join(f"• {d}" for d in datasets)
        lines.append(f"| {task} | {n} | {ds_text} |")

    table_md = "\n".join(lines)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(table_md)

    print(f"Saved dataset table markdown to {outpath}")

#to print all the available tasks and used datasets names as table
def save_tasks_datasets_table_image(df, outpath="ab/stat/docs/figures/tasks_datasets_table.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    rows = []
    for task, sub in df.groupby("task"):
        datasets = sorted(sub["dataset"].astype(str).unique())

        bullets = [f"• {d}" for d in datasets]
        ds_text = "\n".join(bullets)

        rows.append([task, str(len(datasets)), ds_text])

    rows.sort(key=lambda r: int(r[1]), reverse=True)

    fig_h = max(3, 0.7 * len(rows) + 0.4 * max(int(r[1]) for r in rows))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Task", "Number of Datasets", "Datasets"],
        loc="center",
        cellLoc="left",
        colWidths=[0.25, 0.12, 0.63],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, row in enumerate(rows, start=1):
        n_lines = int(row[1])
        height = 0.10 + 0.03 * (n_lines - 1)

        for j in range(3):
            table[(i, j)].set_height(height)
            table[(i, j)].get_text().set_va("center")

    for j in range(3):
        table[(0, j)].set_height(0.12)
        table[(0, j)].get_text().set_va("center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved dataset table image to {outpath}")

#to plot best accuracy per task
def plot_per_task_mean_best_accuracy(best):
    task_stats = (
        best.groupby("task")["best_accuracy"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    task_stats.plot(kind="bar")
    plt.ylabel("Mean best accuracy")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/per_task_mean_best_accuracy.png", dpi=200)
    plt.close()

#plot top models acc vs duration for each task
def plot_task_metric_vs_duration_topk_models_per_dataset_grid(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    top_k=5,
    duration_col="duration",
    value_col="accuracy",         
    metric_name=None,            
    higher_is_better=True,        

    max_cols=3,
    use_lines=True,
    figsize_per_col=6.5,
    figsize_per_row=4.8,
    ylabel=None,

    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
):

    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data found for task '{task_name}'")
        return

    if duration_col not in d.columns:
        raise KeyError(f"'{duration_col}' column not found. Available: {list(d.columns)}")

    if value_col not in d.columns:
        raise KeyError(f"'{value_col}' column not found. Available: {list(d.columns)}")

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name was provided but df has no 'metric' column.")
        d = d[d["metric"].astype(str) == str(metric_name)].copy()
        if d.empty:
            print(f"No data found for task '{task_name}' with metric '{metric_name}'")
            return

    d["epoch"] = d["epoch"].astype(int)
    d["prm_str"] = d["prm"].astype(str)

    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    datasets = sorted(d["dataset"].astype(str).unique())
    n = len(datasets)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
    )

    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    chooser = "idxmax" if higher_is_better else "idxmin"

    for ax_i, dataset in enumerate(datasets):
        ax = axes[ax_i]
        sub = d[d["dataset"].astype(str) == dataset].copy()
        if sub.empty:
            ax.axis("off")
            continue

        gb = sub.groupby(run_cols)[value_col]
        idx = getattr(gb, chooser)().dropna().astype(int)
        best_rows = sub.loc[idx].copy()  

        model_rank = best_rows.groupby("nn")[value_col].mean()
        model_rank = model_rank.sort_values(ascending=not higher_is_better)
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            ax.axis("off")
            continue

        for nn in top_models:
            m = sub[sub["nn"] == nn].copy()

            curve = (
                m.groupby("epoch")
                 .agg(val=(value_col, "mean"), dur=(duration_col, "mean"))
                 .dropna()
                 .sort_values("dur")
            )
            if curve.empty:
                continue

            if use_lines:
                ax.plot(curve["dur"], curve["val"], marker="o", linewidth=1.8, label=str(nn))
            else:
                ax.scatter(curve["dur"], curve["val"], s=18, alpha=0.8, label=str(nn))

        ax.set_xlabel("Duration", fontsize=label_fontsize)
        ax.set_ylabel(ylabel or f"{value_col} (mean)", fontsize=label_fontsize)

        metric_part = f" ({metric_name})" if metric_name is not None else ""
        # ax.set_title(
        #     f"{task_name}{metric_part} — {dataset} — {value_col} vs Duration (Top {len(top_models)} models)",
        #     fontsize=title_fontsize,
        # )
        title_line1 = f"{task_name}{metric_part} — {dataset}"
        title_line2 = f"{value_col} vs duration (best {len(top_models)} models)"

        ax.set_title(title_line1 + "\n" + title_line2, fontsize=title_fontsize)


        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    metric_tag = f"_{metric_name}" if metric_name is not None else ""
    fname = os.path.join(
        outdir,
        f"{task_name}{metric_tag}_{value_col}_vs_{duration_col}_top{top_k}_models_grid.png"
    )

    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Saved: {fname}")

#plot epoch vs accuracy for best models for each task
def plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,         
    top_k=5,
    value_col="accuracy",     
    higher_is_better=True,     
    max_cols=3,
    figsize_per_col=6.5,
    figsize_per_row=4.8,
    ylabel=None,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    palette=None,  
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data found for task '{task_name}'")
        return

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name provided but df has no 'metric' column.")
        d = d[d["metric"].astype(str) == str(metric_name)].copy()
        if d.empty:
            print(f"No data found for task '{task_name}' with metric '{metric_name}'")
            return

    if value_col not in d.columns:
        raise KeyError(f"'{value_col}' column not found. Available: {list(d.columns)}")

    d["epoch"] = d["epoch"].astype(int)
    d["prm_str"] = d["prm"].astype(str)

    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    datasets = sorted(d["dataset"].astype(str).unique())
    n = len(datasets)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
    )

    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    chooser = "idxmax" if higher_is_better else "idxmin"

    cmap = plt.get_cmap(palette)
    

    for ax_i, dataset in enumerate(datasets):
        ax = axes[ax_i]
        sub = d[d["dataset"].astype(str) == dataset].copy()
        if sub.empty:
            ax.axis("off")
            continue

        gb = sub.groupby(run_cols)[value_col]
        idx = getattr(gb, chooser)().dropna().astype(int)
        best_rows = sub.loc[idx].copy()

        model_rank = best_rows.groupby("nn")[value_col].mean()
        model_rank = model_rank.sort_values(ascending=not higher_is_better)
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            ax.axis("off")
            continue

        for i, nn in enumerate(top_models):
            m = sub[sub["nn"] == nn]
            curve = m.groupby("epoch")[value_col].mean().sort_index()

            color = cmap(i / max(len(top_models)-1, 1))

            ax.plot(
                curve.index,
                curve.values,
                marker="o",
                linewidth=1.8,
                label=str(nn),
                color=color
            )

        ax.set_xlabel("Epoch", fontsize=label_fontsize)
        ax.set_ylabel(ylabel or f"{value_col} (mean)", fontsize=label_fontsize)

        metric_part = f" ({metric_name})" if metric_name is not None else ""
        ax.set_title(
            f"{task_name}{metric_part} — {dataset} — best {len(top_models)} models",
            fontsize=title_fontsize
        )

        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    metric_tag = f"_{metric_name}" if metric_name is not None else ""
    fname = os.path.join(outdir, f"{task_name}{metric_tag}_top{top_k}_metric_vs_epoch_grid.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    print(f"Saved: {fname}")

#plot best per run distribution
def plot_task_best_metric_distribution(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,            
    value_col="accuracy",
    top_k=10,
    higher_is_better=True,
    dataset=None,
    figsize=(6.5, 4.8),
    bw_adjust=1.0,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data for task '{task_name}'")
        return

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name provided but df has no 'metric' column")
        d = d[d["metric"].astype(str) == str(metric_name)].copy()
        if d.empty:
            print(f"No data for metric '{metric_name}'")
            return

    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)].copy()
        if d.empty:
            print(f"No data for dataset '{dataset}'")
            return

    if value_col not in d.columns:
        raise KeyError(f"'{value_col}' column not found in dataframe")

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    chooser = "idxmax" if higher_is_better else "idxmin"
    gb = d.groupby(run_cols)[value_col]
    idx = getattr(gb, chooser)().dropna().astype(int)
    best = d.loc[idx].copy()

    model_rank = best.groupby("nn")[value_col].mean()
    model_rank = model_rank.sort_values(ascending=not higher_is_better)
    top_models = model_rank.head(top_k).index.tolist()

    best = best[best["nn"].isin(top_models)].copy()

    if best.empty:
        print("No data after filtering top models.")
        return


    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    for color, nn in zip(colors, top_models):
        vals = best[best["nn"] == nn][value_col].dropna().values
        if len(vals) < 3:
            continue

        kde = gaussian_kde(vals, bw_method=bw_adjust)
        x = np.linspace(vals.min(), vals.max(), 300)
        y = kde(x)

        plt.plot(x, y, color=color, linewidth=2, label=str(nn))
        plt.fill_between(x, y, color=color, alpha=0.25)

        median = np.median(vals)
        plt.axvline(median, color=color, linestyle="--", linewidth=1, alpha=0.8)

    metric_label = metric_name.upper() if metric_name else value_col

    title_parts = [task_name]
    if dataset:
        title_parts.append(dataset)
    if metric_name:
        title_parts.append(metric_name.upper())


    plt.title(
    " — ".join(title_parts) + " — Best-per-run Distribution",
    fontsize=title_fontsize
)
    plt.xlabel(metric_label, fontsize=label_fontsize)
    plt.ylabel("Density", fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(title="Model", fontsize=6)
    plt.tight_layout()


    metric_tag = f"_{metric_name}" if metric_name else ""
    dataset_tag = f"_{dataset}" if dataset else ""
    fname = os.path.join(
        outdir,
        f"{task_name}{dataset_tag}{metric_tag}_best_metric_distribution_top{top_k}.png"
    )

    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved: {fname}")


#select best per run 
def best_per_run(df, task_name, metric_name=None, value_col="accuracy", higher_is_better=True):
    d = df[df["task"] == task_name].copy()
    if d.empty:
        return d

    if metric_name is not None:
        d = d[d["metric"].astype(str) == str(metric_name)].copy()
        if d.empty:
            return d

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    chooser = "idxmax" if higher_is_better else "idxmin"
    idx = getattr(d.groupby(run_cols)[value_col], chooser)().dropna().astype(int)

    best = d.loc[idx].copy()
    return best

#plot best vs duration
def plot_pareto_best_vs_duration(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,
    value_col="accuracy",
    duration_col="duration",
    higher_is_better=True,
    top_k_models=12,     
    dataset=None,
    figsize=(10.5, 6.5),
    logx=False,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
):
    os.makedirs(outdir, exist_ok=True)

    best = best_per_run(df, task_name, metric_name, value_col, higher_is_better)
    if best.empty:
        print(f"No data for {task_name} / {metric_name}")
        return

    if duration_col not in best.columns:
        raise KeyError(f"'{duration_col}' not found in df")

    if dataset is not None:
        best = best[best["dataset"].astype(str) == str(dataset)].copy()
        if best.empty:
            print(f"No data for dataset {dataset}")
            return

    best = best.dropna(subset=[value_col, duration_col])

    rank = best.groupby("nn")[value_col].mean().sort_values(ascending=not higher_is_better)
    top_models = rank.head(top_k_models).index.tolist()
    best = best[best["nn"].isin(top_models)].copy()

    pts = best[[duration_col, value_col]].sort_values(duration_col).values
    frontier = []
    best_so_far = -np.inf if higher_is_better else np.inf
    for dur, val in pts:
        better = (val > best_so_far) if higher_is_better else (val < best_so_far)
        if better:
            frontier.append((dur, val))
            best_so_far = val
    frontier = np.array(frontier) if frontier else None

    plt.figure(figsize=figsize)

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(top_models)))
    for nn, c in zip(top_models, colors):
        sub = best[best["nn"] == nn]
        plt.scatter(sub[duration_col], sub[value_col], s=35, alpha=0.85, label=str(nn), color=c)

    if frontier is not None and len(frontier) >= 2:
        plt.plot(frontier[:, 0], frontier[:, 1], linewidth=2.2, color="black", alpha=0.9)

    if logx:
        plt.xscale("log")

    metric_label = metric_name.upper() if metric_name else value_col
    title = f"{task_name} — Best-per-run {metric_label} vs {duration_col} (Pareto)"
    if dataset is not None:
        title += f" — {dataset}"
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(duration_col, fontsize=label_fontsize)
    plt.ylabel(metric_label, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()


    metric_tag = f"_{metric_name}" if metric_name else ""
    dataset_tag = f"_{str(dataset).replace('/','_').replace(' ','_')}" if dataset is not None else ""
    fname = os.path.join(outdir, f"{task_name}{metric_tag}{dataset_tag}_pareto_best_vs_{duration_col}.png")
    plt.savefig(fname, dpi=220)
    plt.close()
    print(f"Saved: {fname}")

#create heatmap
def plot_winner_rank_heatmap(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,             
    value_col="accuracy",       
    higher_is_better=True,
    top_k_models=10,            
    figsize=(12, 7),
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data for task '{task_name}'")
        return

    if metric_name is not None:
        d = d[d["metric"].astype(str) == str(metric_name)].copy()
        if d.empty:
            print(f"No data for metric '{metric_name}'")
            return

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]
    chooser = "idxmax" if higher_is_better else "idxmin"
    idx = getattr(d.groupby(run_cols)[value_col], chooser)().dropna().astype(int)
    best = d.loc[idx].copy()

    agg = (
        best.groupby(["dataset", "nn"])[value_col]
        .mean()
        .reset_index()
    )

   
    coverage = agg.groupby("nn")["dataset"].nunique()
    perf = agg.groupby("nn")[value_col].mean()
    model_pick = (
        pd.DataFrame({"coverage": coverage, "perf": perf})
        .sort_values(["coverage", "perf"], ascending=[False, not higher_is_better])
    )

    top_models = model_pick.head(top_k_models).index.tolist()
    agg = agg[agg["nn"].isin(top_models)]


    mat = agg.pivot_table(index="dataset", columns="nn", values=value_col, aggfunc="mean")

    if mat.empty:
        print("Nothing to plot after filtering.")
        return

    ranks = mat.rank(axis=1, ascending=not higher_is_better, method="min")

    plt.figure(figsize=figsize)
    im = plt.imshow(ranks.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Rank (1 = best)")

    plt.title(
        f"{task_name}" +
        (f" ({str(metric_name).upper()})" if metric_name else "") +
        " — Model Rank Heatmap (mean best-per-run)"
    )
    plt.xlabel("Model")
    plt.ylabel("Dataset")

    plt.xticks(np.arange(ranks.shape[1]), ranks.columns.astype(str), rotation=45, ha="right")
    plt.yticks(np.arange(ranks.shape[0]), ranks.index.astype(str))

    plt.tight_layout()

    metric_tag = f"_{metric_name}" if metric_name else ""
    fname = os.path.join(outdir, f"{task_name}{metric_tag}_rank_heatmap_top{top_k_models}.png")
    plt.savefig(fname, dpi=220)
    plt.close()
    print(f"Saved: {fname}")

#combine some pictures into one grid
def combine_pngs_to_grid(
    image_paths,
    out_path,
    ncols=3,
    figsize_per_cell=(6.5, 3.8),
    titles=None,  
    title_fontsize=16,            
    dpi=220,
    pad=0.02,
    bg_color="white",
):
    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError("None of the provided image paths exist.")

    n = len(paths)
    nrows = math.ceil(n / ncols)

    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), facecolor=bg_color)
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        img = mpimg.imread(paths[i])
        ax.imshow(img)
        ax.axis("off")

        if titles is not None and i < len(titles) and titles[i]:
            ax.set_title(titles[i], fontsize=title_fontsize)

    plt.subplots_adjust(wspace=pad, hspace=pad)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined grid: {out_path}")

def plot_topk_models_bar(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,           
    value_col="accuracy",       
    top_k=10,
    higher_is_better=True,
    dataset=None,               
    figsize=(10.5, 6.5),
    title=None,
    xlabel="Score",
    palette="viridis",          
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
):


    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"].astype(str) == str(task_name)].copy()
    if d.empty:
        print(f"No data found for task '{task_name}'")
        return

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name provided but df has no 'metric' column.")
        d = d[d["metric"].astype(str).str.lower() == str(metric_name).lower()].copy()
        if d.empty:
            print(f"No data found for task '{task_name}' with metric '{metric_name}'")
            return

    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)].copy()
        if d.empty:
            print(f"No data found for dataset '{dataset}' in task '{task_name}'")
            return

    if value_col not in d.columns:
        raise KeyError(f"'{value_col}' column not found. Available: {list(d.columns)}")

    d["prm_str"] = d["prm"].astype(str) if "prm" in d.columns else ""
    run_cols = ["task", "dataset", "nn", "prm_str", "transform_code"]
    if "metric" in d.columns:
        run_cols.insert(2, "metric")

    chooser = "idxmax" if higher_is_better else "idxmin"

    gb = d.groupby(run_cols)[value_col]
    idx = getattr(gb, chooser)().dropna().astype(int)
    if idx.empty:
        print("No best-per-run indices found (check value_col / data).")
        return
    best = d.loc[idx].copy()

    model_scores = best.groupby("nn")[value_col].mean().sort_values(ascending=not higher_is_better)
    top = model_scores.head(top_k)

    if top.empty:
        print("No models found after ranking.")
        return

    labels = top.index.astype(str).tolist()[::-1]       
    scores = top.values[::-1]

    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0.05, 0.95, len(scores)))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, scores, color=colors)

    for b, s in zip(bars, scores):
        ax.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"{s:.4f}",
            va="center",
            ha="left",
            fontsize=tick_fontsize,
            clip_on=False,
        )

    if title is None:
        metric_part = f" ({str(metric_name).upper()})" if metric_name else ""
        ds_part = f" — {dataset}" if dataset else ""
        title = f"{task_name}{metric_part}{ds_part} — Top {len(scores)} models"

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.06)

    fig.tight_layout()

    metric_tag = f"_{metric_name}" if metric_name else ""
    ds_tag = f"_{str(dataset).replace('/','_').replace(' ','_')}" if dataset else ""
    fname = os.path.join(outdir, f"{task_name}{metric_tag}{ds_tag}_top{top_k}_models_bar.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    print(f"Saved: {fname}")

def mean_ci95_t(values):
    """
    Mean ± 95% CI using Student-t interval.
    Returns (mean, low, high, n). For n<2 -> (nan, nan, nan, n).
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan, np.nan, n)

    m = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / (n ** 0.5)
    h = t.ppf(0.975, df=n - 1) * se
    return (m, m - h, m + h, n)


def plot_bar_ci95_top_models(
    df,
    out_path,
    task_name,
    top_k=10,
    metric_name=None,          
    value_col="accuracy",
    dataset=None,
    higher_is_better=True,
    min_n_ci=5,               

    figsize=(10.5, 6.5),
    title_fontsize=18,
    label_fontsize=14,
    tick_fontsize=12,

    palette="Set2",          
    ci_color="blue",         
):
   

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    d = df[df["task"].astype(str) == str(task_name)].copy()
    if d.empty:
        print(f"No data for task '{task_name}'")
        return

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name provided but df has no 'metric' column.")
        d = d[d["metric"].astype(str).str.lower() == str(metric_name).lower()].copy()
        if d.empty:
            print(f"No data for task '{task_name}' metric '{metric_name}'")
            return

    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)].copy()
        if d.empty:
            print(f"No data for dataset '{dataset}'")
            return

    if value_col not in d.columns:
        raise KeyError(f"Missing '{value_col}' column")

    run_cols = ["dataset", "nn", "prm_id", "transform_id"]
    chooser = "idxmax" if higher_is_better else "idxmin"
    idx = getattr(d.groupby(run_cols)[value_col], chooser)().dropna().astype(int)
    best = d.loc[idx].copy()
    if best.empty:
        print("No best-per-run rows found.")
        return

    rank = best.groupby("nn")[value_col].mean().sort_values(ascending=not higher_is_better)
    top_models = rank.head(top_k).index.tolist()
    best = best[best["nn"].isin(top_models)].copy()

    rows = []
    for nn in top_models:
        vals = best.loc[best["nn"] == nn, value_col].dropna().to_numpy()
        mu, lo, hi, n = mean_ci95_t(vals)
        if n >= min_n_ci and np.isfinite(mu):
            rows.append((str(nn), mu, lo, hi, n))

    if not rows:
        print(f"No models with n>={min_n_ci} runs to compute CI.")
        return

    labels = [r[0] for r in rows]
    means  = np.array([r[1] for r in rows], float)
    lows   = np.array([r[2] for r in rows], float)
    highs  = np.array([r[3] for r in rows], float)

    yerr = np.vstack([means - lows, highs - means])

    cmap = plt.get_cmap(palette)
    bar_colors = cmap(np.linspace(0.1, 0.9, len(means)))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, means, color=bar_colors)
    ax.errorbar(
        x, means, yerr=yerr,
        fmt="none",
        ecolor=ci_color,
        capsize=6,
        linewidth=1.5
    )

    metric_label = metric_name.upper() if metric_name else value_col
    title1 = f"{task_name}" + (f" ({metric_label})" if metric_name else "")
    title2 = f"Top {len(labels)} models — mean ± 95% CI" + (f" — {dataset}" if dataset else "")

    ax.set_title(title1 + "\n" + title2, fontsize=title_fontsize)
    ax.set_ylabel(metric_label, fontsize=label_fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_line_ci95_over_epochs(
    df,
    out_path,
    task_name,
    top_k_models=10,          
    metric_name=None,          
    value_col="accuracy",
    dataset=None,
    higher_is_better=True,
    min_n_ci=5,               

    figsize=(10.5, 6.5),
    title_fontsize=18,
    label_fontsize=14,
    tick_fontsize=12,
    line_width=2.0,
    marker_size=5,
    capsize=6,

    line_color="blue",
    ci_color="red",
):
   

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    d = df[df["task"].astype(str) == str(task_name)].copy()
    if d.empty:
        print(f"No data for task '{task_name}'")
        return

    if metric_name is not None:
        if "metric" not in d.columns:
            raise KeyError("metric_name provided but df has no 'metric' column.")
        d = d[d["metric"].astype(str).str.lower() == str(metric_name).lower()].copy()
        if d.empty:
            print(f"No data for task '{task_name}' metric '{metric_name}'")
            return

    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)].copy()
        if d.empty:
            print(f"No data for dataset '{dataset}'")
            return

    for col in ["epoch", "nn", "prm_id", "transform_id", value_col]:
        if col not in d.columns:
            raise KeyError(f"Missing column '{col}'")

    d["epoch"] = d["epoch"].astype(int)

    run_cols = ["dataset", "nn", "prm_id", "transform_id"]
    chooser = "idxmax" if higher_is_better else "idxmin"
    idx = getattr(d.groupby(run_cols)[value_col], chooser)().dropna().astype(int)
    best = d.loc[idx].copy()
    if best.empty:
        print("No best-per-run rows found.")
        return

    rank = best.groupby("nn")[value_col].mean().sort_values(ascending=not higher_is_better)
    top_models = rank.head(top_k_models).index.tolist()
    d = d[d["nn"].isin(top_models)].copy()

    epochs = sorted(d["epoch"].unique())
    xs, means, lo, hi = [], [], [], []

    for e in epochs:
        vals = d.loc[d["epoch"] == e, value_col].dropna().to_numpy()
        mu, l, h, n = mean_ci95_t(vals)
        if np.isfinite(mu) and n >= min_n_ci:
            xs.append(e); means.append(mu); lo.append(l); hi.append(h)

    if len(xs) < 2:
        print(f"Not enough epochs with n>={min_n_ci} to plot CI.")
        return

    means = np.asarray(means, float)
    yerr = np.vstack([means - np.asarray(lo), np.asarray(hi) - means])

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        xs, means, yerr=yerr,
        marker="o",
        markersize=marker_size,
        linewidth=line_width,
        capsize=capsize,
        color=line_color,
        ecolor=ci_color
    )

    metric_label = metric_name.upper() if metric_name else value_col
    title1 = f"{task_name}" + (f" ({metric_label})" if metric_name else "")
    title2 = f"Top {len(top_models)} models — mean ± 95% CI" + (f" — {dataset}" if dataset else "")

    ax.set_title(title1 + "\n" + title2, fontsize=title_fontsize)
    ax.set_xlabel("Epoch", fontsize=label_fontsize)
    ax.set_ylabel(metric_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = load_data()
    print_tasks(df)
    save_tasks_datasets_table_markdown(df)
    save_tasks_datasets_table_image(df)
    datasets_per_task(df)
    best = compute_best_runs(df)
    plot_per_task_mean_best_accuracy(best)

    plot_task_metric_vs_duration_topk_models_per_dataset_grid(
    df,
    task_name="img-classification",
    top_k=5,
    value_col="accuracy",
    higher_is_better=True,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    )

    plot_task_metric_vs_duration_topk_models_per_dataset_grid(
    df,
    task_name="txt-generation",
    top_k=2,
    value_col="accuracy",
    higher_is_better=True,
    max_cols=2,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    )

    plot_task_metric_vs_duration_topk_models_per_dataset_grid(
    df,
    outdir=OUTDIR,
    task_name="img-segmentation",   
    top_k=5,
    value_col="accuracy",           
    metric_name="iou",              
    higher_is_better=True,
    ylabel="IoU (mean)",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-classification",
    value_col="accuracy",
    top_k=5,
    ylabel="Accuracy (mean)",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    palette="Set2"
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-segmentation",
    metric_name="iou",
    top_k=5,
    value_col="accuracy",
    ylabel="IoU (mean)",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    palette="tab10"
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-captioning",
    metric_name="bleu",
    top_k=5,
    value_col="accuracy",
    ylabel="BLEU (mean)",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    palette="Set2"
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="txt-generation",
    top_k=2,
    value_col="accuracy",
    ylabel="Accuracy (mean)",
    max_cols=2,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    palette="Dark2"
    )

    plot_task_best_metric_distribution(
    df,
    task_name="img-classification",
    top_k=10,
    )

    plot_task_best_metric_distribution(
    df,
    task_name="img-segmentation",
    metric_name="iou",
    top_k=10,
    )

    plot_task_best_metric_distribution(
    df,
    task_name="img-captioning",
    metric_name="bleu",
    top_k=10,
    )

    plot_task_best_metric_distribution(
    df,
    task_name="txt-generation",
    top_k=10,
    )


    plot_pareto_best_vs_duration(df, task_name="img-classification", top_k_models=12, duration_col="duration")

    plot_pareto_best_vs_duration(df, task_name="img-segmentation", metric_name="iou", top_k_models=12, duration_col="duration")

    plot_pareto_best_vs_duration(df, task_name="img-captioning", metric_name="bleu", top_k_models=12, duration_col="duration")

    plot_pareto_best_vs_duration(df, task_name="txt-generation", top_k_models=2, duration_col="duration")

    
    plot_winner_rank_heatmap(df, task_name="img-classification", top_k_models=10)
    plot_winner_rank_heatmap(df, task_name="img-segmentation", metric_name="iou", top_k_models=10)
    plot_winner_rank_heatmap(df, task_name="img-captioning", metric_name="bleu", top_k_models=10)


    group1 = [
        "ab/stat/docs/figures/img-captioning_bleu_top5_metric_vs_epoch_grid.png",
        "ab/stat/docs/figures/img-segmentation_iou_top5_metric_vs_epoch_grid.png",
        "ab/stat/docs/figures/txt-generation_top2_metric_vs_epoch_grid.png"
    ]

    group2 = [
        "ab/stat/docs/figures/img-classification_best_metric_distribution_top10.png",
        "ab/stat/docs/figures/img-segmentation_iou_best_metric_distribution_top10.png",
        "ab/stat/docs/figures/img-captioning_bleu_best_metric_distribution_top10.png",
        "ab/stat/docs/figures/txt-generation_best_metric_distribution_top10.png",
    ]

    group3 = [
        "ab/stat/docs/figures/img-classification_pareto_best_vs_duration.png",
        "ab/stat/docs/figures/img-segmentation_iou_pareto_best_vs_duration.png",
        "ab/stat/docs/figures/img-captioning_bleu_pareto_best_vs_duration.png",
        "ab/stat/docs/figures/txt-generation_pareto_best_vs_duration.png",
    ]

    group4 = [
        "ab/stat/docs/figures/img-segmentation_iou_accuracy_vs_duration_top5_models_grid.png",
        "ab/stat/docs/figures/txt-generation_accuracy_vs_duration_top2_models_grid.png"
    ]
    group5 = [
        "ab/stat/docs/figures/img-classification_top10_models_bar.png",
        "ab/stat/docs/figures/img-segmentation_iou_top10_models_bar.png",
        "ab/stat/docs/figures/txt-generation_top2_models_bar.png"
    ]
    group6 = [
        "ab/stat/docs/figures/bar_ci95_img_classification.png",
        "ab/stat/docs/figures/bar_ci95_img_segmentation_iou.png",
        "ab/stat/docs/figures/bar_ci95_txt_generation.png"
    ]
    group7 = [
        "ab/stat/docs/figures/line_ci95_img_classification.png",
        "ab/stat/docs/figures/line_ci95_img_segmentation_iou.png",
        "ab/stat/docs/figures/line_ci95_txt_generation.png"
    ]
  


    combine_pngs_to_grid(
        group1,
        out_path="ab/stat/docs/figures/Figure_A_accuracy_vs_epoch.png",
        ncols=2,
    )

    combine_pngs_to_grid(
        group2,
        out_path="ab/stat/docs/figures/Figure_B_best_metric_distributions.png",
        ncols=2,
    )

    combine_pngs_to_grid(
        group3,
        out_path="ab/stat/docs/figures/Figure_C_pareto_frontiers.png",
        ncols=2,
    )

    combine_pngs_to_grid(
        group4,
        out_path="ab/stat/docs/figures/Figure_D_acc_vs_duration.png",
        ncols=2,
    )

    combine_pngs_to_grid(
        group5,
        out_path="ab/stat/docs/figures/Figure_E.png",
        ncols=2,
    )
    combine_pngs_to_grid(
        group6,
        out_path="ab/stat/docs/figures/Figure_F.png",
        ncols=2,
    )
    combine_pngs_to_grid(
        group7,
        out_path="ab/stat/docs/figures/Figure_G.png",
        ncols=2,
    )
    plot_topk_models_bar(
    df,
    outdir=OUTDIR,
    task_name="img-classification",
    metric_name=None,
    value_col="accuracy",
    top_k=10,
    xlabel="Score",
    palette="viridis",
    )
    plot_topk_models_bar(
    df,
    outdir=OUTDIR,
    task_name="txt-generation",
    metric_name=None,
    value_col="accuracy",
    top_k=2,
    xlabel="Score",
    palette="viridis",
    )
    plot_topk_models_bar(
    df,
    outdir=OUTDIR,
    task_name="img-segmentation",
    metric_name="iou",
    value_col="accuracy",  
    top_k=10,
    xlabel="Score",
    palette="viridis",
    )


    plot_bar_ci95_top_models(
        df,
        out_path="ab/stat/docs/figures/bar_ci95_img_classification.png",
        task_name="img-classification",
        top_k=10,
        palette="Set2",
    )
    plot_line_ci95_over_epochs(
        df,
        out_path="ab/stat/docs/figures/line_ci95_img_classification.png",
        task_name="img-classification",
        top_k_models=10,
        line_color="blue",
        ci_color="red",
    )

    plot_bar_ci95_top_models(
        df,
        out_path="ab/stat/docs/figures/bar_ci95_txt_generation.png",
        task_name="txt-generation",
        top_k=2,
        palette="Set2",
    )
    plot_line_ci95_over_epochs(
        df,
        out_path="ab/stat/docs/figures/line_ci95_txt_generation.png",
        task_name="txt-generation",
        top_k_models=2,
        line_color="blue",
        ci_color="red",
    )

    plot_bar_ci95_top_models(
        df,
        out_path="ab/stat/docs/figures/bar_ci95_img_segmentation_iou.png",
        task_name="img-segmentation",
        metric_name="iou",
        top_k=10,
        palette="Set2",
    )
    plot_line_ci95_over_epochs(
        df,
        out_path="ab/stat/docs/figures/line_ci95_img_segmentation_iou.png",
        task_name="img-segmentation",
        metric_name="iou",
        top_k_models=10,
        line_color="blue",
        ci_color="red",
    )
   



if __name__ == "__main__":
    main()
