#! --utf-8--
# loop over noise
# Path: scripts/LoopOverNoise.py

# load library
import numpy as np
import datetime
from tqdm import tqdm
import pickle

# load neuronav library
from neuronav.envs.grid_templates import GridTemplate
from neuronav.envs.grid_env import GridEnv, GridSize, GridObservation

# load custom modules
from predecessor.env import GridWorld1D, GridEnvNoise, compute_true_q_value
from predecessor.agents import QLearningAgent, QLearningLambdaAgent, SFAgent, PFAgent
from predecessor.training import training_loop_2D
from predecessor.utils import mean_across_trials, std_across_trials, get_last_reward


def run_trial(agent, env, agent_type, total_trials, episode, max_step_length):
    training = training_loop_2D(
        env,
        agent,
        agent_type=agent_type,
        episodes=episode,
        max_step_length=max_step_length,
    )
    trials_step, trials_reward, trials_q = [], [], []

    for trial in tqdm(range(total_trials), desc="trial"):
        training.reset()
        step, reward, q_tables = training.run()
        trials_step.append(step)
        trials_reward.append(reward)
        trials_q.append(q_tables)
    return trials_step, trials_reward, trials_q


def save_raw_results(
    exp, agent_type, noise, trials_step, trials_reward, trials_q, lambda_=None
):
    if lambda_ is None:
        exp[agent_type][noise][0] = {
            "noise_level": noise,
            "trials_step": trials_step,
            "trials_reward": trials_reward,
            "trials_q": trials_q,
        }
    else:
        exp[agent_type][noise][lambda_] = {
            "noise_level": noise,
            "trials_step": trials_step,
            "trials_reward": trials_reward,
            "trials_q": trials_q,
            "lambda": lambda_,
        }


def run_experiment(
    env, noise_list, lambda_list, total_trials, episode, max_step_length, env_type="1D"
):
    _ = env.reset()
    exp = {
        "Q": {},
        "SF": {},
        "Q(lambda)": {},
        "PF": {},
        "params": {
            "env size": env.size,
            "env_type": env_type,
            "total trials": total_trials,
            "episode": episode,
            "max step length": max_step_length,
        },
    }  # initialize
    for noise in tqdm(noise_list, desc="noise level"):
        # initialize dictionary
        for agent_type in ["Q", "SF", "Q(lambda)", "PF"]:
            exp[agent_type][noise] = {}
            if agent_type in ["Q", "SF"]:
                exp[agent_type][noise][0] = {}
            elif agent_type in ["Q(lambda)", "PF"]:
                for lambda_ in lambda_list:
                    exp[agent_type][noise][lambda_] = {}
        # set env noise
        env.noise = noise
        _ = env.reset()
        # run experiment
        for agent_type, Agent in [("Q", QLearningAgent), ("SF", SFAgent)]:
            agent = Agent(env.size, env.action_space.n)
            print("Running " + agent_type + " learning.")
            trials_step, trials_reward, trials_q = run_trial(
                agent, env, agent_type, total_trials, episode, max_step_length
            )
            # save_results(exp, agent_type, noise, trials_step, trials_reward)
            save_raw_results(
                exp, agent_type, noise, trials_step, trials_reward, trials_q
            )
        # save results
        with open(save_dir + "noise" + env_type + "_" + now + ".pickle", "wb") as f:
            pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)
        print("Saved results." + str(noise) + " noise. Q, SF learning.\n")
        for lambda_ in lambda_list:
            for agent_type, Agent in [
                ("Q(lambda)", QLearningLambdaAgent),
                ("PF", PFAgent),
            ]:
                print("Running " + agent_type + " learning.")
                agent = Agent(env.size, env.action_space.n, lambda_=lambda_)
                trials_step, trials_reward, trials_q = run_trial(
                    agent, env, agent_type, total_trials, episode, max_step_length
                )
                # save_results(exp, agent_type, noise, trials_step, trials_reward, lambda_)
                save_raw_results(
                    exp,
                    agent_type,
                    noise,
                    trials_step,
                    trials_reward,
                    trials_q,
                    lambda_,
                )
            # save results
            with open(save_dir + "noise" + env_type + "_" + now + ".pickle", "wb") as f:
                pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)

        print("Saved results." + str(noise) + " noise. Q(lambda), PF learning.\n")
    print("Finished.")


if __name__ == "__main__":
    # file manage
    save_dir = "./results/pickles/"
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # parameter for environment

    noise_list = [0.05, 0.25, 0.5]
    # parameter for agent
    lambda_list = [0.7, 0.8, 0.9]

    # parameter for training
    total_trials = 100
    episode = 3000
    max_step_length = 200

    # initialize environment
    env = GridEnvNoise(
        # template=GridTemplate.four_rooms,
        size=GridSize.micro,
        obs_type=GridObservation.onehot,
    )
    # corridor_size = 20
    # env = GridWorld1D(size=corridor_size)
    print("Total trials: " + str(total_trials))
    run_experiment(
        env,
        noise_list,
        lambda_list,
        total_trials,
        episode,
        max_step_length,
        env_type="2D",
    )
