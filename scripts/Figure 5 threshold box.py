import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from predecessor.visualize import return_plot_data_for_noise
from predecessor.utils import make_df_from_exp

save_file_name = "./results/processed_pickles/noise1D_2023-12-22-11-09.pickle"

with open(save_file_name, "rb") as f:
    exp_dic = pickle.load(f)

algorithms = ["Q", "SF", "Q($\\lambda$)", "PF"]

noise_level = list(exp_dic["Q"].keys())

lambda_list = list(exp_dic["PF"][0.05].keys())

df_step_20 = make_df_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, "threshold_20"
)
df_step_40 = make_df_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, "threshold_40"
)
df_step_60 = make_df_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, "threshold_60"
)

fig = plt.figure(constrained_layout=True, figsize=(12, 12))

axs = fig.subplots(3, 1, sharey=True)

sns.boxplot(
    x="noise",
    y="threshold_20",
    hue="algorithm",
    data=df_step_20,
    ax=axs[0],
    legend=False,
)


axs[0].set_ylabel("Episode ($\\theta <= 20$)", fontsize=12)


sns.boxplot(
    x="noise",
    y="threshold_40",
    hue="algorithm",
    data=df_step_40,
    ax=axs[1],
    legend=False,
)

axs[1].set_ylabel("Episode ($\\theta <= 40$)", fontsize=12)

sns.boxplot(
    x="noise",
    y="threshold_60",
    hue="algorithm",
    data=df_step_60,
    ax=axs[2],
)

axs[2].set_ylabel("Episode ($\\theta <= 60$)", fontsize=12)
axs[2].legend(loc="right", borderaxespad=0.0)

axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("$\sigma$", fontsize=12)

axs[0].ticklabel_format(
    style="scientific", useMathText=True, axis="y", scilimits=(0, 0)
)


fig.text(0, 0.985, "A", fontsize=14, fontweight="bold")
fig.text(0, 0.655, "B", fontsize=14, fontweight="bold")
fig.text(0, 0.32, "C", fontsize=14, fontweight="bold")

plt.savefig("./results/images/Figure 5 threshold box.pdf")
plt.close()
