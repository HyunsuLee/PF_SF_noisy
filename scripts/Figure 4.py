import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from predecessor.visualize import return_plot_data_for_noise
from predecessor.utils import make_df_from_exp

save_file_name = "./results/processed_pickles/noise1D_2023-12-22-11-09.pickle"

with open(save_file_name, "rb") as f:
    exp_dic = pickle.load(f)

algorithms = ["Q", "SF", "Q($\\lambda$)", "PF"]

noise_level = list(exp_dic["Q"].keys())

lambda_list = list(exp_dic["PF"][0.05].keys())

fig = plt.figure(constrained_layout=True, figsize=(12, 7))
subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1, 1])

axsUpper = subfigs[0].subplots(1, len(noise_level), sharey=True, sharex=True)

for i, noise in enumerate(noise_level):
    return_plot_data_for_noise(
        axsUpper[i],
        exp_dic,
        "mean_ma_step",
        "std_ma_step",
        lambda_list,
        algorithms,
        noise,
    )

    # plt.ylabel(kwargs["ylabel"])
    # plt.xlim(kwargs["xlim"])


axsUpper[0].ticklabel_format(
    style="scientific", useMathText=True, axis="x", scilimits=(0, 0)
)
axsUpper[0].set_ylabel("Average episode length", fontsize=12)
axsUpper[0].set_xlabel("")
axsUpper[1].set_xlabel("Episode", fontsize=12)
axsUpper[2].set_xlabel("")
axsUpper[2].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
subfigs[0].text(0, 0.96, "A", fontsize=14, fontweight="bold")


axsMiddle = subfigs[1].subplots(1, 1)
df_step = make_df_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, "last_ma_step"
)

sns.boxplot(
    x="noise",
    y="last_ma_step",
    hue="algorithm",
    data=df_step,
    ax=axsMiddle,
)
axsMiddle.set_xlabel("$\sigma$", fontsize=12)
# axsLower.set_xticklabels([0, 1, 2], noise_level)
axsMiddle.set_ylabel("Final episode lengths", fontsize=12)
axsMiddle.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
subfigs[1].text(0, 0.98, "B", fontsize=14, fontweight="bold")


plt.savefig("./results/images/Figure 4.pdf")
plt.close()


### for new figure 5(previous figure 4C)
figC = plt.figure(constrained_layout=True, figsize=(12, 8))

axsLower = figC.subplots(3, 2, sharey=True, sharex=True, width_ratios=[1, 1])
count_df = pd.read_csv("./results/csvs/noise1D_2023-12-22-11-09_count_epi_length.csv")
count_df_small = count_df[count_df["trials_step"] <= 29]
count_df_small_grouped = (
    count_df_small.groupby(["algorithm", "noise", "trial"]).sum().reset_index()
)
low_noise = count_df_small_grouped[count_df_small_grouped["noise"] == 0.05]
mid_noise = count_df_small_grouped[count_df_small_grouped["noise"] == 0.25]
high_noise = count_df_small_grouped[count_df_small_grouped["noise"] == 0.5]

count_df_large = count_df[count_df["trials_step"] == 100]
count_df_large_grouped = (
    count_df_large.groupby(["algorithm", "noise", "trial"]).sum().reset_index()
)
low_noise_large = count_df_large_grouped[count_df_large_grouped["noise"] == 0.05]
mid_noise_large = count_df_large_grouped[count_df_large_grouped["noise"] == 0.25]
high_noise_large = count_df_large_grouped[count_df_large_grouped["noise"] == 0.5]


sns.boxplot(
    data=low_noise,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[0, 0],
    legend=False,
)


sns.boxplot(
    data=low_noise_large,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[0, 1],
    legend=True,
)

sns.boxplot(
    data=mid_noise,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[1, 0],
    legend=False,
)

sns.boxplot(
    data=mid_noise_large,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[1, 1],
    legend=False,
)

sns.boxplot(
    data=high_noise,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[2, 0],
    legend=False,
)

sns.boxplot(
    data=high_noise_large,
    x="algorithm",
    y="count",
    order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    hue="algorithm",
    hue_order=[
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ],
    ax=axsLower[2, 1],
    legend=False,
)

for i in range(3):
    # axsLower[i, 0].set_ylabel("Number of episode in a trial", fontsize=12)
    axsLower[i, 0].ticklabel_format(
        style="scientific", useMathText=True, axis="y", scilimits=(0, 0)
    )

axsLower[0, 0].set_ylabel("$\sigma=0.05$")
axsLower[1, 0].set_ylabel("$\sigma=0.25$")
axsLower[2, 0].set_ylabel("$\sigma=0.5$")
axsLower[0, 1].legend()

plt.setp(axsLower[2, 0].get_xticklabels(), rotation=45)
plt.setp(axsLower[2, 1].get_xticklabels(), rotation=45)

for i in range(3):
    for j in range(2):
        axsLower[i, j].set_xlabel("")
axsLower[2, 0].set_xlabel("Episode length less than 30")
axsLower[2, 1].set_xlabel("Episode length 100")


figC.text(0.5, 0.98, "Average number of episodes in a trial", ha="center", fontsize=14)
# figC.text(0.5, 0.01, "Episode length", ha="center", fontsize=12)
figC.text(0, 0.98, "A", fontsize=14, fontweight="bold")
figC.text(0, 0.685, "B", fontsize=14, fontweight="bold")
figC.text(0, 0.385, "C", fontsize=14, fontweight="bold")

plt.savefig("./results/images/Figure 4C.pdf")
plt.close()


### for description of summary of figure 4C(now figure 5 -> A2?5?)
def mean_sem_count(df):
    df_summary = (
        df.drop(columns=["trials_step", "trial", "Unnamed: 0"])
        .groupby(["noise", "algorithm"])
        .describe()
    )
    df_summary = df_summary["count"][["mean", "std"]]
    df_summary["SEM"] = df_summary["std"] / np.sqrt(100)
    df_summary = df_summary.drop(columns="std")
    # Sort algorithms in desired order
    algorithm_order = [
        "Q",
        "SF",
        "Q($\\lambda$ = 0.7)",
        "Q($\\lambda$ = 0.8)",
        "Q($\\lambda$ = 0.9)",
        "PF($\\lambda$ = 0.7)",
        "PF($\\lambda$ = 0.8)",
        "PF($\\lambda$ = 0.9)",
    ]
    df_summary = df_summary.reindex(algorithm_order, level=1)
    print(df_summary.round(2))


mean_sem_count(count_df_small_grouped)
mean_sem_count(count_df_large_grouped)
