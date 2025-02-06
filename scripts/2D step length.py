import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from predecessor.visualize import return_plot_data_for_noise
from predecessor.utils import make_df_from_exp

save_file_name = "./results/processed_pickles/noise2D_2024-01-31-11-41.pickle"

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


plt.savefig("./results/images/2D step length.pdf")
plt.close()
