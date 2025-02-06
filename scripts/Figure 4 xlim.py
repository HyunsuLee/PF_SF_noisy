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

# fig = plt.figure(constrained_layout=True, figsize=(12, 8))
# subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1, 1])
# fig = plt.subplots
fig, axsUpper = plt.subplots(
    2,
    len(noise_level),
    sharey=True,
    sharex=True,
    constrained_layout=True,
    figsize=(12, 6),
)


# axsUpper = plt.subplots(1, len(noise_level), sharey=True)

for i, noise in enumerate(noise_level):
    return_plot_data_for_noise(
        axsUpper[0, i],
        exp_dic,
        "mean_step_length",
        "std_step_length",
        lambda_list,
        algorithms,
        noise,
        xlim=(0, 450),
        xlabel_vis=False,
    )

for i, noise in enumerate(noise_level):
    return_plot_data_for_noise(
        axsUpper[1, i],
        exp_dic,
        "mean_ma_step",
        "std_ma_step",
        lambda_list,
        algorithms,
        noise,
        xlim=(0, 450),
        title=False,
    )
    # plt.ylabel(kwargs["ylabel"])
    # plt.xlim(kwargs["xlim"])
axsUpper[0, 0].set_ylabel("Average episode length", fontsize=12)
axsUpper[1, 0].set_ylabel("Average episode length", fontsize=12)
axsUpper[1, 0].set_xlabel("")
axsUpper[1, 1].set_xlabel("Episode", fontsize=12)
axsUpper[1, 2].set_xlabel("")
axsUpper[0, 2].legend()
fig.text(0, 0.96, "A", fontsize=14, fontweight="bold")
fig.text(0, 0.49, "B", fontsize=14, fontweight="bold")


plt.savefig("./results/images/Figure 4 xlim.pdf")
plt.close()
