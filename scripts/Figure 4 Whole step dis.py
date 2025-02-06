import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

count_df = pd.read_csv("./results/csvs/noise1D_2023-12-22-11-09_count_epi_length.csv")

axs = sns.catplot(
    data=count_df,
    x="trials_step",
    y="count",
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
    kind="box",
    col="noise",
)

legend = axs.legend
legend.set_title("")

axs.axes.flatten()[0].set_title("$\sigma = 0.05$")
axs.axes.flatten()[1].set_title("$\sigma = 0.25$")
axs.axes.flatten()[2].set_title("$\sigma = 0.5$")
# axs.set_titles("Episode length distribution", fontsize=12)
plt.xlim(-1.5, 83.5)
plt.xticks(np.arange(0, 82, 10))

axs.axes.flatten()[0].set_xlabel("")
axs.axes.flatten()[1].set_xlabel("Episode length", fontsize=12)
axs.axes.flatten()[2].set_xlabel("")


axs.set_ylabels("Number of episode in a trial", fontsize=12)
plt.ticklabel_format(style="sci", useMathText=True, axis="y", scilimits=(0, 0))

plt.savefig("./results/images/Figure 4 Whole step dis.pdf")
