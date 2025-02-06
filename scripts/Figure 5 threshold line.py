import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from predecessor.visualize import plot_threshold_for_noise

save_raw_file_name = "./results/pickles/noise1D_2023-12-22-11-09.pickle"
with open(save_raw_file_name, "rb") as f:
    raw_exp_dic = pickle.load(f)

save_file_name = "./results/processed_pickles/noise1D_2023-12-22-11-09.pickle"

with open(save_file_name, "rb") as f:
    exp_dic = pickle.load(f)

algorithms = ["Q", "SF", "Q($\\lambda$)", "PF"]

noise_level = list(exp_dic["Q"].keys())

lambda_list = list(exp_dic["PF"][0.05].keys())

fig, axs = plt.subplots(
    2,
    2,
    # sharey=True,
    # sharex=True,
    constrained_layout=True,
    figsize=(10, 7),
)
q_whole_step = np.array(raw_exp_dic["Q"][0.05][0]["trials_step"])
for i in range(100):
    axs[0, 0].plot(q_whole_step[i], alpha=0.02, color="blue")
axs[0, 0].hlines(20, -50, 3100, color="red", linestyles="dashed")
axs[0, 0].hlines(40, -50, 3100, color="red", linestyles="dashed")
axs[0, 0].hlines(60, -50, 3100, color="red", linestyles="dashed")
axs[0, 0].set_ylabel("Episode length", fontsize=12)
axs[0, 0].set_title("Q learning ($\\sigma$ = 0.05)", fontsize=12)

plot_threshold_for_noise(
    axs[0, 1], exp_dic, lambda_list, algorithms, 0.05, xlim=(-50, 3100)
)
plot_threshold_for_noise(
    axs[1, 0], exp_dic, lambda_list, algorithms, 0.25, xlim=(-50, 3100)
)
plot_threshold_for_noise(
    axs[1, 1], exp_dic, lambda_list, algorithms, 0.5, xlim=(-50, 3100)
)

axs[0, 1].set_ylabel("$\\theta$", fontsize=12)
axs[1, 0].set_ylabel("$\\theta$", fontsize=12)
axs[1, 1].set_ylabel("$\\theta$", fontsize=12)
axs[0, 1].set_yticks([20, 40, 60])
axs[0, 1].set_yticklabels(["20", "40", "60"])
axs[1, 0].set_yticks([20, 40, 60])
axs[1, 0].set_yticklabels(["20", "40", "60"])
axs[1, 1].set_yticks([20, 40, 60])
axs[1, 1].set_yticklabels(["20", "40", "60"])
axs[0, 1].set_xlabel("")
axs[1, 0].set_xlabel("Episode", fontsize=12)
axs[1, 1].set_xlabel("Episode", fontsize=12)
axs[0, 0].ticklabel_format(style="sci", useMathText=True, axis="x", scilimits=(0, 0))
axs[0, 1].ticklabel_format(style="sci", useMathText=True, axis="x", scilimits=(0, 0))
axs[1, 0].ticklabel_format(style="sci", useMathText=True, axis="x", scilimits=(0, 0))
axs[1, 1].ticklabel_format(style="sci", useMathText=True, axis="x", scilimits=(0, 0))

axs[0, 1].legend()

fig.text(0.01, 0.98, "A", ha="center", fontsize=14, fontweight="bold")
fig.text(0.515, 0.98, "B", ha="center", fontsize=14, fontweight="bold")
fig.text(0.01, 0.48, "C", ha="center", fontsize=14, fontweight="bold")
fig.text(0.515, 0.48, "D", ha="center", fontsize=14, fontweight="bold")


plt.savefig("./results/images/Figure 5 threshold line.pdf")
plt.close()
