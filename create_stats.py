import os
import matplotlib.pyplot as plt
import ab.nn.api as api
import numpy as np
import textwrap

OUTDIR = "docs/figures"

#to load the data
def load_data():
    os.makedirs(OUTDIR, exist_ok=True)
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

#to print all the available tasks and used datasets names as table
def save_tasks_datasets_table_image(df, outpath="docs/figures/tasks_datasets_table.png"):
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

#to plot accuracy vs epoch for text generation task
def plot_txt_generation_acc_vs_epoc(df):
    task_name = "txt-generation"

    tg = df[df["task"] == task_name].copy()

    if tg.empty:
        print(f"No data found for task '{task_name}'")
        return

    tg["epoch"] = tg["epoch"].astype(int)

    plt.figure(figsize=(8, 5))

    for nn, sub in tg.groupby("nn"):
        s = sub.groupby("epoch")["accuracy"].mean().sort_index()
        plt.plot(s.index, s.values, marker="o", linewidth=1, label=str(nn))

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Text generation: accuracy vs epoch")
    plt.legend(title="Model", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/txt_generation_accuracy_vs_epoch_by_model.png", dpi=200)
    plt.close()

#to plot duration vs accuracy for text generation task
def plot_txt_generation_duration_vs_accuracy(df):
    task_name = "txt-generation"
    tg = df[df["task"] == task_name].copy()

    if tg.empty:
        print(f"No data found for task '{task_name}'")
        return

    tg["duration"] = tg["duration"].astype(float)

    plt.figure(figsize=(7, 5))

    for nn, sub in tg.groupby("nn"):
        plt.scatter(sub["duration"], sub["accuracy"], label=str(nn), alpha=0.7)

    plt.xlabel("Duration")
    plt.ylabel("Accuracy")
    plt.title("Text generation: duration vs accuracy")
    plt.legend(title="Model", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/txt_generation_duration_vs_accuracy.png", dpi=200)
    plt.close()

#to plot best 10 models for each dataset for the image classification task
def plot_img_classification_top10_models_per_dataset(
    df,
    outdir="docs/figures",
    task_name="img-classification",
    top_k=10,
    agg="mean", 
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data found for task '{task_name}'")
        return

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    idx = d.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
    best = d.loc[idx].copy().rename(columns={"accuracy": "best_accuracy"})

    for dataset, sub in best.groupby("dataset"):
        if agg == "mean":
            model_scores = sub.groupby("nn")["best_accuracy"].mean()
            agg_label = "Mean best accuracy"
        elif agg == "max":
            model_scores = sub.groupby("nn")["best_accuracy"].max()
            agg_label = "Max best accuracy"
        else:
            raise ValueError("agg must be 'mean' or 'max'")

        top = model_scores.sort_values(ascending=False).head(top_k)

        if top.empty:
            continue

        plt.figure(figsize=(9, 4))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top)))
        top.sort_values().plot(kind="barh", color=colors)

        plt.xlabel(agg_label)
        plt.title(f"{task_name} — {dataset} — Top {min(top_k, len(top))} models")
        plt.tight_layout()

        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{outdir}/{task_name}_{safe_ds}_top{top_k}_{agg}.png", dpi=200)
        plt.close()

        print(f"Saved: {outdir}/{task_name}_{safe_ds}_top{top_k}_{agg}.png")

#to plot best 5 accuracy vs epoch for each dataset for image classification task
def plot_img_classification_top5_acc_vs_epoch_per_dataset(
    df,
    outdir="docs/figures",
    task_name="img-classification",
    top_k=5,
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()
    if d.empty:
        print(f"No data found for task '{task_name}'")
        return

    d["epoch"] = d["epoch"].astype(int)

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    for dataset, sub_ds in d.groupby("dataset"):
        idx = sub_ds.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
        best_rows = sub_ds.loc[idx].copy()  

        model_rank = (
            best_rows.groupby("nn")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            continue

        plt.figure(figsize=(8, 5))

        for nn in top_models:
            m = sub_ds[sub_ds["nn"] == nn]
            curve = m.groupby("epoch")["accuracy"].mean().sort_index()
            plt.plot(curve.index, curve.values, marker="o", linewidth=1, label=str(nn))

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{task_name} — {dataset} — Top {len(top_models)} models: accuracy vs epoch")
        plt.legend(title="Model", fontsize=8)
        plt.tight_layout()

        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        fname = f"{outdir}/{task_name}_{safe_ds}_top{top_k}_acc_vs_epoch.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")

#to plot best 5 iou values for each dataset for image segmentation task
def plot_img_segmentation_top5_iou_vs_epoch_per_dataset(
    df,
    outdir="docs/figures",
    task_name="img-segmentation",
    metric_name="iou",
    top_k=5,
):
    os.makedirs(outdir, exist_ok=True)

    d = df[(df["task"] == task_name) & (df["metric"] == metric_name)].copy()
    if d.empty:
        print(f"No data found for task='{task_name}' and metric='{metric_name}'")
        return

    d["epoch"] = d["epoch"].astype(int)

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    for dataset, sub_ds in d.groupby("dataset"):
        idx = sub_ds.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
        best_rows = sub_ds.loc[idx].copy()  
        model_rank = (
            best_rows.groupby("nn")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            continue

        plt.figure(figsize=(8, 5))
        for nn in top_models:
            m = sub_ds[sub_ds["nn"] == nn]
            curve = m.groupby("epoch")["accuracy"].mean().sort_index()
            plt.plot(curve.index, curve.values, marker="o", linewidth=1, label=str(nn))

        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title(f"{task_name} — {dataset} — Top {len(top_models)} models: IoU vs epoch")
        plt.legend(title="Model", fontsize=8)
        plt.tight_layout()

        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        fname = f"{outdir}/{task_name}_{safe_ds}_top{top_k}_iou_vs_epoch.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")

#to plot best 5 bleu vs epoch for each dataset for image captioning task
def plot_img_captioning_top5_metric_vs_epoch_per_dataset(
    df,
    outdir="docs/figures",
    task_name="img-captioning",
    metric_name="accuracy",  
    top_k=5,
):
    os.makedirs(outdir, exist_ok=True)

    d = df[(df["task"] == task_name) & (df["metric"] == metric_name)].copy()
    if d.empty:
        print(f"No data found for task='{task_name}' and metric='{metric_name}'")
        return

    d["epoch"] = d["epoch"].astype(int)

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    for dataset, sub_ds in d.groupby("dataset"):
        idx = sub_ds.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
        best_rows = sub_ds.loc[idx].copy()

        model_rank = (
            best_rows.groupby("nn")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            continue

        plt.figure(figsize=(8, 5))
        for nn in top_models:
            m = sub_ds[sub_ds["nn"] == nn]
            curve = m.groupby("epoch")["accuracy"].mean().sort_index()
            plt.plot(curve.index, curve.values, marker="o", linewidth=1, label=str(nn))

        plt.xlabel("Epoch")
        plt.ylabel(metric_name.upper())
        plt.title(f"{task_name} — {dataset} — Top {len(top_models)} models: {metric_name} vs epoch")
        plt.legend(title="Model", fontsize=8)
        plt.tight_layout()

        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        fname = f"{outdir}/{task_name}_{safe_ds}_top{top_k}_{metric_name}_vs_epoch.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")

#to plot best 5 accuracy vs epoch for each dataset for text generation task
def plot_txt_generation_top5_acc_vs_epoch_per_dataset(
    df,
    outdir="docs/figures",
    task_name="txt-generation",
    metric_name="accuracy",  
    top_k=2,
):
    os.makedirs(outdir, exist_ok=True)

    d = df[df["task"] == task_name].copy()

    if d.empty:
        print(f"No data found for task='{task_name}' and metric='{metric_name}'")
        return

    d["epoch"] = d["epoch"].astype(int)

    d["prm_str"] = d["prm"].astype(str)
    run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

    for dataset, sub_ds in d.groupby("dataset"):
        idx = sub_ds.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
        best_rows = sub_ds.loc[idx].copy()
        model_rank = (
            best_rows.groupby("nn")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        top_models = model_rank.head(top_k).index.tolist()

        if not top_models:
            continue

        plt.figure(figsize=(8, 5))
        for nn in top_models:
            m = sub_ds[sub_ds["nn"] == nn]
            curve = m.groupby("epoch")["accuracy"].mean().sort_index()
            plt.plot(curve.index, curve.values, marker="o", linewidth=1, label=str(nn))

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{task_name} — {dataset} — Top {len(top_models)} models: accuracy vs epoch")
        plt.legend(title="Model", fontsize=8)
        plt.tight_layout()

        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        fname = f"{outdir}/{task_name}_{safe_ds}_top{top_k}_acc_vs_epoch.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")

def main():
    df = load_data()
    print_tasks(df)
    save_tasks_datasets_table_image(df)
    datasets_per_task(df)
    best = compute_best_runs(df)
    plot_per_task_mean_best_accuracy(best)
    plot_txt_generation_acc_vs_epoc(df)
    plot_txt_generation_duration_vs_accuracy(df)
    plot_img_classification_top10_models_per_dataset(df, top_k=10, agg="mean")
    plot_img_classification_top5_acc_vs_epoch_per_dataset(df, top_k=5)
    plot_img_segmentation_top5_iou_vs_epoch_per_dataset(df)
    plot_img_captioning_top5_metric_vs_epoch_per_dataset(df, metric_name="bleu")
    plot_txt_generation_top5_acc_vs_epoch_per_dataset(df, top_k=2)

if __name__ == "__main__":
    main()
