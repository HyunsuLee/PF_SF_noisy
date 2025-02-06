# utils.py

import numpy as np
import pandas as pd
from collections import Counter


def mean_across_trials(trials_list):
    return np.mean(np.array(trials_list), axis=0)


def std_across_trials(trials_list):
    return np.std(np.array(trials_list), axis=0)


def get_last_reward(trials_list):
    return np.array(trials_list)[:, -1]


def mean_across_episode(trials_list):
    # return np.mean(np.array(trials_list)[:, -100:], axis=1)
    return np.mean(np.array(trials_list), axis=1)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def step_to_below_threshold(step, threshold):
    for i in range(len(step)):
        if step[i] < threshold:
            return i
    return len(step)


def moving_average_across_trials(trials_step, window):
    ma = []
    for i in range(len(trials_step)):
        ma.append(moving_average(trials_step[i], window))
    return ma


def get_dStep_dEpi(trials_step, window_size: int = 10, dEpi: int = 1):
    if window_size == 0:
        ma_trials = np.array(trials_step)
    else:
        ma_trials = np.array(moving_average_across_trials(trials_step, window_size))
    dStep_dEpi = ma_trials[:, dEpi:] - ma_trials[:, :-dEpi]
    return dStep_dEpi


def step_to_below_threshold_across_trials(trials_step, threshold, window):
    step_list = []
    for i in range(len(trials_step)):
        step_list.append(
            step_to_below_threshold(moving_average(trials_step[i], window), threshold)
        )
    return step_list


def my_argmax(Qarray):
    max_index = np.where(Qarray == Qarray.max())[0]
    if len(max_index) == 1:
        return np.argmax(Qarray)
    else:
        return np.random.randint(len(Qarray))


def make_df_from_exp(exp_dic, algorithms, noise_level, lambda_list, data_type, **kargs):
    df = pd.DataFrame()

    for algorithm in algorithms:
        for noise in noise_level:
            if algorithm in ["Q", "SF"]:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                data_type: exp_dic[algorithm][noise][0][data_type],
                                "algorithm": algorithm,
                                "noise": noise,
                            }
                        ),
                    ]
                )
            elif algorithm == "Q($\\lambda$)":
                for lambda_ in lambda_list:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    data_type: exp_dic[algorithm][noise][lambda_][
                                        data_type
                                    ],
                                    "algorithm": f"Q($\lambda$ = {lambda_})",
                                    "noise": noise,
                                }
                            ),
                        ]
                    )
            elif algorithm == "PF":
                for lambda_ in lambda_list:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    data_type: exp_dic[algorithm][noise][lambda_][
                                        data_type
                                    ],
                                    "algorithm": f"PF($\lambda$ = {lambda_})",
                                    "noise": noise,
                                }
                            ),
                        ]
                    )
    return df


def make_algorithm_list(lambda_list):
    algorithm_list = []
    for algorithm in ["Q", "SF"]:
        algorithm_list.append(algorithm)
    for algorithm in ["Q($\\lambda$)", "PF"]:
        for lambda_ in lambda_list:
            if algorithm == "PF":
                algorithm_list.append(f"{algorithm}($\\lambda$ = {lambda_})")
            else:
                algorithm_list.append(f"Q($\\lambda$ = {lambda_})")
    return algorithm_list


def summary_table(exp_dic, algorithms, noise_level, lambda_list, **kargs):
    kargs.setdefault("data_type", "last_reward")

    df_last_reward = make_df_from_exp(
        exp_dic,
        algorithms,
        noise_level,
        lambda_list,
        data_type=kargs["data_type"],
    )
    df_summary = df_last_reward.groupby(["noise", "algorithm"]).describe()
    df_summary.columns = df_summary.columns.droplevel(0)
    df_summary["SEM"] = df_summary["std"] / np.sqrt(df_summary["count"])
    if kargs["data_type"] == "last_reward":
        df_summary["mean/sem"] = df_summary["mean"] / df_summary["SEM"]
        df_summary["mean/sem"] = df_summary["mean/sem"].round(2)

    # df_summary["std"] = df_summary["std"].round(2)
    df_summary["SEM"] = df_summary["SEM"].round(2)
    df_summary["mean"] = df_summary["mean"].round(2)
    # df_summary["IQR"] = df_summary["75%"] - df_summary["25%"]

    if kargs["data_type"] == "last_reward":
        df_summary = df_summary[["mean", "SEM", "25%", "50%", "75%", "mean/sem"]]
    elif kargs["data_type"] == "mean_epi_step" or kargs["data_type"] == "last_ma_step":
        df_summary = df_summary[["mean", "SEM", "25%", "50%", "75%"]]
    algorithm_list = make_algorithm_list(lambda_list)
    df_summary = df_summary.reindex(algorithm_list, level=1)
    df_summary.rename(
        columns={
            "mean": "Mean",
            # "std": "Std",
            "SEM": "SEM",
            "25%": "25%",
            "50%": "50%",
            "75%": "75%",
            # "IQR": "IQR",
            "mean/sem": "Mean/SEM",
        },
        inplace=True,
    )
    return df_summary


def whole_steps_from_exp(
    exp_dic, algorithms, noise_level, lambda_list, data_type, episode_limit=3000
):
    df = pd.DataFrame()

    for algorithm in algorithms:
        for noise in noise_level:
            if algorithm in ["Q", "SF"]:
                data_list = exp_dic[algorithm][noise][0][data_type]
                for data in data_list:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    data_type: data[:episode_limit],
                                    "algorithm": algorithm,
                                    "noise": noise,
                                }
                            ),
                        ]
                    )
            elif algorithm == "Q($\\lambda$)":
                for lambda_ in lambda_list:
                    data_list = exp_dic[algorithm][noise][lambda_][data_type]
                    for data in data_list:
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        data_type: data[:episode_limit],
                                        "algorithm": f"Q($\lambda$ = {lambda_})",
                                        "noise": noise,
                                    }
                                ),
                            ]
                        )
            elif algorithm == "PF":
                for lambda_ in lambda_list:
                    data_list = exp_dic[algorithm][noise][lambda_][data_type]
                    for data in data_list:
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        data_type: data[:episode_limit],
                                        "algorithm": f"PF($\lambda$ = {lambda_})",
                                        "noise": noise,
                                    }
                                ),
                            ]
                        )
    return df


def count_steps_of_trial(
    exp_dic, algorithms, noise_level, lambda_list, data_type, episode_limit=3000
):
    df = pd.DataFrame()

    for algorithm in algorithms:
        for noise in noise_level:
            if algorithm in ["Q", "SF"]:
                data_list = exp_dic[algorithm][noise][0][data_type]
                for i, data in enumerate(data_list):
                    data_count = Counter(data[:episode_limit])
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    data_type: list(data_count.keys()),
                                    "count": list(data_count.values()),
                                    "trial": i + 1,
                                    "algorithm": algorithm,
                                    "noise": noise,
                                }
                            ),
                        ]
                    )
            elif algorithm == "Q($\\lambda$)":
                for lambda_ in lambda_list:
                    data_list = exp_dic[algorithm][noise][lambda_][data_type]
                    for i, data in enumerate(data_list):
                        data_count = Counter(data[:episode_limit])
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        data_type: list(data_count.keys()),
                                        "count": list(data_count.values()),
                                        "trial": i + 1,
                                        "algorithm": f"Q($\lambda$ = {lambda_})",
                                        "noise": noise,
                                    }
                                ),
                            ]
                        )
            elif algorithm == "PF":
                for lambda_ in lambda_list:
                    data_list = exp_dic[algorithm][noise][lambda_][data_type]
                    for i, data in enumerate(data_list):
                        data_count = Counter(data[:episode_limit])
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        data_type: list(data_count.keys()),
                                        "count": list(data_count.values()),
                                        "trial": i + 1,
                                        "algorithm": f"PF($\lambda$ = {lambda_})",
                                        "noise": noise,
                                    }
                                ),
                            ]
                        )
    return df


if __name__ == "__main__":
    test = np.array([1, 2, 3, 4, 5])
    print(moving_average(test, 2))
