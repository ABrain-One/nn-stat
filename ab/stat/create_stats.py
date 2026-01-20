import os
import matplotlib.pyplot as plt
import ab.nn.api as api
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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
                ax.plot(curve["dur"], curve["val"], marker="o", linewidth=1, label=str(nn))
            else:
                ax.scatter(curve["dur"], curve["val"], s=18, alpha=0.8, label=str(nn))

        ax.set_xlabel("Duration")
        ax.set_ylabel(ylabel or f"{value_col} (mean)")
        metric_part = f" ({metric_name})" if metric_name is not None else ""
        ax.set_title(f"{task_name}{metric_part} — {dataset} — {value_col} vs Duration (Top {len(top_models)} models)")
        ax.legend(fontsize=7)

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
            m = sub[sub["nn"] == nn]
            curve = m.groupby("epoch")[value_col].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1, label=str(nn))

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel or f"{value_col} (mean)")
        metric_part = f" ({metric_name})" if metric_name is not None else ""
        ax.set_title(f"{task_name}{metric_part} — {dataset} — Top {len(top_models)} models")
        ax.legend(fontsize=7)

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
    figsize=(10, 6),
    bw_adjust=1.0,
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

    plt.title(" — ".join(title_parts) + " — Best-per-run Distribution")
    plt.xlabel(metric_label)
    plt.ylabel("Density")
    plt.legend(title="Model", fontsize=9)
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
    plt.title(title)
    plt.xlabel(duration_col)
    plt.ylabel(metric_label)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()

    metric_tag = f"_{metric_name}" if metric_name else ""
    dataset_tag = f"_{str(dataset).replace('/','_').replace(' ','_')}" if dataset is not None else ""
    fname = os.path.join(outdir, f"{task_name}{metric_tag}{dataset_tag}_pareto_best_vs_{duration_col}.png")
    plt.savefig(fname, dpi=220)
    plt.close()
    print(f"Saved: {fname}")


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
    )

    plot_task_metric_vs_duration_topk_models_per_dataset_grid(
    df,
    task_name="txt-generation",
    top_k=2,
    value_col="accuracy",
    higher_is_better=True,
    max_cols=2,
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
    )
   
    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-classification",
    value_col="accuracy",
    top_k=5,
    ylabel="Accuracy (mean)",
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-segmentation",
    metric_name="iou",
    top_k=5,
    value_col="accuracy",
    ylabel="IoU (mean)",
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="img-captioning",
    metric_name="bleu",
    top_k=5,
    value_col="accuracy",
    ylabel="BLEU (mean)",
    )

    plot_task_metric_vs_epoch_topk_models_per_dataset_grid(
    df,
    task_name="txt-generation",
    top_k=2,
    value_col="accuracy",
    ylabel="Accuracy (mean)",
    max_cols=2,
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




if __name__ == "__main__":
    main()
