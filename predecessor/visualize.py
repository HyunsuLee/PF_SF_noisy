# visualize.py

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")
plt.style.context("default")

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"


def plot_data(
    exp_dic, data_type, std_data, lambda_list, algorithms, noise_level, **kwargs
):
    kwargs.setdefault("figsize", (10, 5))
    kwargs.setdefault("xlabel", "Episodes")
    kwargs.setdefault("ylabel", "Average episode length")
    kwargs.setdefault("xlim", (0, 3000))
    fig, axs = plt.subplots(1, len(noise_level), figsize=kwargs["figsize"], sharey=True)
    for i, noise in enumerate(noise_level):
        plt.subplot(1, len(noise_level), i + 1)
        plot_data_for_noise(
            exp_dic, data_type, std_data, lambda_list, algorithms, noise, **kwargs
        )
        if i == 0:
            plt.ylabel(kwargs["ylabel"])
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_data_for_noise(
    exp_dic, data_type, std_data, lambda_list, algorithms, noise_level, **kwargs
):
    kwargs.setdefault("xlabel", "Episodes")
    kwargs.setdefault("ylabel", "Average episode length")
    kwargs.setdefault("xlim", (0, 3000))
    for algorithm in algorithms:
        if algorithm == "Q" or algorithm == "SF":
            data = exp_dic[algorithm][noise_level][0][data_type]
            sem = exp_dic[algorithm][noise_level][0][std_data] / np.sqrt(100)
            len = data.shape[0]
            upper = data + sem
            lower = data - sem
            plt.plot(data, label=algorithm)
            plt.fill_between(range(len), upper, lower, alpha=0.2)
        else:
            for lambda_ in lambda_list:
                data = exp_dic[algorithm][noise_level][lambda_][data_type]
                sem = exp_dic[algorithm][noise_level][lambda_][std_data] / np.sqrt(100)
                len = data.shape[0]
                upper = data + sem
                lower = data - sem
                if algorithm == "Q($\\lambda$)":
                    plt.plot(data, label="Q($\lambda$ = " + str(lambda_) + ")")
                else:
                    plt.plot(
                        data,
                        label=algorithm + "(" + "$\lambda$ = " + str(lambda_) + ")",
                    )
                plt.fill_between(range(len), upper, lower, alpha=0.2)

    plt.xlabel(kwargs["xlabel"])
    # plt.ylabel(kwargs["ylabel"])
    plt.xlim(kwargs["xlim"])
    plt.title("$\sigma$ = " + str(noise_level))


def return_plot_data_for_noise(
    axs, exp_dic, data_type, std_data, lambda_list, algorithms, noise_level, **kwargs
):
    kwargs.setdefault("xlabel_vis", True)
    kwargs.setdefault("xlabel", "Episodes")
    kwargs.setdefault("ylabel", "Average episode length")
    kwargs.setdefault("xlim", (0, 3000))
    kwargs.setdefault("title", True)
    kwargs.setdefault("alpha", 1.0)
    # axs = plt.subplot(111)
    for algorithm in algorithms:
        if algorithm == "Q" or algorithm == "SF":
            data = exp_dic[algorithm][noise_level][0][data_type]
            sem = exp_dic[algorithm][noise_level][0][std_data] / np.sqrt(100)
            len = data.shape[0]
            upper = data + sem
            lower = data - sem
            axs.plot(data, label=algorithm, alpha=kwargs["alpha"])
            axs.fill_between(range(len), upper, lower, alpha=0.2)
        else:
            for lambda_ in lambda_list:
                data = exp_dic[algorithm][noise_level][lambda_][data_type]
                sem = exp_dic[algorithm][noise_level][lambda_][std_data] / np.sqrt(100)
                len = data.shape[0]
                upper = data + sem
                lower = data - sem
                if algorithm == "Q($\\lambda$)":
                    axs.plot(
                        data,
                        label="Q($\lambda$ = " + str(lambda_) + ")",
                        alpha=kwargs["alpha"],
                    )
                else:
                    axs.plot(
                        data,
                        label=algorithm + "(" + "$\lambda$ = " + str(lambda_) + ")",
                        alpha=kwargs["alpha"],
                    )
                axs.fill_between(range(len), upper, lower, alpha=0.2)
    if kwargs["xlabel_vis"]:
        axs.set_xlabel(kwargs["xlabel"])
    # plt.ylabel(kwargs["ylabel"])
    axs.set_xlim(kwargs["xlim"])
    if kwargs["title"]:
        axs.set_title("$\sigma$ = " + str(noise_level))

    return axs


def plot_threshold_for_noise(
    axs, exp_dic, lambda_list, algorithms, noise_level, **kwargs
):
    color_map = plt.get_cmap("tab10")
    colors = color_map(np.arange(0, 8, 1))

    kwargs.setdefault("xlabel", "Episodes")
    kwargs.setdefault("ylabel", "\\theta")
    kwargs.setdefault("xlim", (0, 3000))
    kwargs.setdefault("thresholds", [20, 40, 60])
    for i, algorithm in enumerate(algorithms):
        if algorithm == "Q" or algorithm == "SF":
            x, y, sems = [], [], []
            for threshold in kwargs["thresholds"]:
                thre = exp_dic[algorithm][noise_level][0]["threshold_" + str(threshold)]
                mean = np.mean(thre)
                sem = np.std(thre) / np.sqrt(100)
                x.append(threshold)
                y.append(mean)
                sems.append(sem)
            x, y, sems = np.array(x), np.array(y), np.array(sems)
            axs.plot(y, x, label=algorithm, color=colors[i])
            axs.errorbar(y, x, xerr=sem, fmt="o", color=colors[i])
        elif algorithm == "Q($\\lambda$)" or algorithm == "PF":
            for j, lambda_ in enumerate(lambda_list):
                x, y, sems = [], [], []
                for threshold in kwargs["thresholds"]:
                    thre = exp_dic[algorithm][noise_level][lambda_][
                        "threshold_" + str(threshold)
                    ]
                    mean = np.mean(thre)
                    sem = np.std(thre) / np.sqrt(100)
                    x.append(threshold)
                    y.append(mean)
                    sems.append(sem)
                x, y, sems = np.array(x), np.array(y), np.array(sems)
                if algorithm == "Q($\\lambda$)":
                    axs.plot(
                        y,
                        x,
                        label=f"Q($\lambda$ = {lambda_})",
                        color=colors[i + j],
                    )
                    axs.errorbar(
                        y,
                        x,
                        xerr=sem,
                        fmt="o",
                        color=colors[i + j],
                    )
                else:
                    axs.plot(
                        y,
                        x,
                        label=f"{algorithm} ($\lambda$ = {lambda_})",
                        color=colors[i + 2 + j],
                    )
                    axs.errorbar(
                        y,
                        x,
                        xerr=sem,
                        fmt="o",
                        color=colors[i + 2 + j],
                    )

    axs.set_xlabel(kwargs["xlabel"])
    # plt.ylabel(kwargs["ylabel"])
    axs.set_xlim(kwargs["xlim"])
    axs.set_title("$\sigma$ = " + str(noise_level))
    return axs
