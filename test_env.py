from environment.grid_world import GridWorld
import random

obstacles = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
env = GridWorld(size=10, start=(0, 0), goal=(9, 9), obstacles=obstacles)

state = env.reset()
env.render()

for i in range(10):
    action = random.randint(0, 3)
    state, reward, done, _ = env.step(action)
    print(f"Step {i}: action={action}, state={state}, reward={reward}, done={done}")
    env.render()
    if done:
        break
    