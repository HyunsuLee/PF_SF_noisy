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

fig = plt.figure(constrained_layout=True, figsize=(12, 8))
subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1, 1])

axsUpper = subfigs[0].subplots(1, len(noise_level), sharey=True, sharex=True)

for i, noise in enumerate(noise_level):
    return_plot_data_for_noise(
        axsUpper[i],
        exp_dic,
        "mean_rewards",
        "std_rewards",
        lambda_list,
        algorithms,
        noise,
    )

    # plt.ylabel(kwargs["ylabel"])
    # plt.xlim(kwargs["xlim"])

axsUpper[0].set_ylabel("Average cumulative reward", fontsize=12)
axsUpper[0].set_xlabel("")
axsUpper[1].set_xlabel("Episode", fontsize=12)
axsUpper[2].set_xlabel("")
axsUpper[0].ticklabel_format(style="sci", useMathText=True, axis="y", scilimits=(0, 0))
axsUpper[0].ticklabel_format(style="sci", useMathText=True, axis="x", scilimits=(0, 0))


axsUpper[2].legend()
subfigs[0].text(0, 0.96, "A", fontsize=14, fontweight="bold")


axsLower = subfigs[1].subplots(1, 1)
df_reward = make_df_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, "last_reward"
)

sns.boxplot(
    x="noise",
    y="last_reward",
    hue="algorithm",
    data=df_reward,
    ax=axsLower,
)
axsLower.set_xlabel("$\sigma$")
# axsLower.set_xticklabels([0, 1, 2], noise_level)
axsLower.set_ylabel("Cumulative rewards distribution", fontsize=12)
axsLower.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
axsLower.ticklabel_format(style="sci", useMathText=True, axis="y", scilimits=(0, 0))
subfigs[1].text(0, 0.96, "B", fontsize=14, fontweight="bold")


plt.savefig("./results/images/Figure 3.pdf")
plt.close()
