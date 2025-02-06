import matplotlib.pyplot as plt

from neuronav.envs.grid_env import GridSize, GridObservation
from predecessor.env import GridEnvNoise

env = GridEnvNoise(
    # template=GridTemplate.four_rooms,
    size=GridSize.micro,
    obs_type=GridObservation.onehot,
)

env.reset()
env.render()

plt.savefig("./results/images/Figure 2B.pdf")
