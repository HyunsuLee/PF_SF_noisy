import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from predecessor.visualize import return_plot_data_for_noise
from predecessor.utils import make_df_from_exp

save_file_name_1D = "./results/processed_pickles/noise1D_2023-12-22-11-09.pickle"
save_file_name_2D = "./results/processed_pickles/noise2D_2024-01-31-11-41.pickle"

with open(save_file_name_1D, "rb") as f:
    exp_dic_1D = pickle.load(f)

with open(save_file_name_2D, "rb") as f:
    exp_dic_2D = pickle.load(f)

algorithms = ["Q", "SF", "Q($\\lambda$)", "PF"]

noise_level = list(exp_dic_1D["Q"].keys())

lambda_list = list(exp_dic_1D["PF"][0.05].keys())

# TODO for differentiating between 1D and 2D, noise relationship. mean episode length.
df_step_1D = make_df_from_exp(
    exp_dic_1D, algorithms, noise_level, lambda_list, "mean_epi_step"
)

df_step_2D = make_df_from_exp(
    exp_dic_2D, algorithms, noise_level, lambda_list, "mean_epi_step"
)

fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), sharex=True, sharey=True, tight_layout=True
)


sns.pointplot(
    x="noise",
    y="mean_epi_step",
    hue="algorithm",
    data=df_step_1D,
    errorbar="se",
    ax=axs[0],
    legend=False,
)

# plt.savefig("./results/images/noise_relationship_1D.png")


sns.pointplot(
    x="noise",
    y="mean_epi_step",
    hue="algorithm",
    data=df_step_2D,
    errorbar="se",
    ax=axs[1],
    legend=True,
)
axs[0].set_ylabel("Average episode length", fontsize=12)
axs[0].set_xlabel("$\\sigma$", fontsize=12, loc="right")
axs[1].set_xlabel("")
axs[0].set_title("1D", fontsize=14, fontweight="bold")
axs[1].set_title("2D", fontsize=14, fontweight="bold")

plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
plt.savefig("./results/images/noise_relationship.pdf")
plt.close()
