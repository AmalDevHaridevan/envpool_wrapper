import envpool_wrap as envpool
print(envpool.list_all_envs())
import gym 
from example_env import MyEnv
import tqdm 
import time 
import numpy as np

def run_gym(num_envs, total_step, async_):
    task_id="MyEnv"
    env = gym.vector.make(task_id, num_envs, async_, envid=1)
    env.reset()
    action = env.action_space.sample()
    done = False
    t = time.time()
    for _ in tqdm.trange(total_step):
        if num_envs == 1:
            if done:
                done = False
                env.reset()
            else:
                done = env.step(action)[2]
        else:
            env.step(action)
    fps = total_step * num_envs / (time.time() - t)
    print(f"FPS = { fps:.2f}")
    return fps

def run_envpool(num_envs, total_step, async_):
    task_id="MyEnv_wrap"
    kwargs = {}
    env = envpool.make_gym(task_id, num_envs=num_envs)
    env.async_reset()
    action = np.array([env.action_space.sample() for _ in range(num_envs)])
    t = time.time()
    for _ in tqdm.trange(total_step):
        info = env.recv()[-1]
        env.step(action, info["env_id"])
    duration = time.time() - t
    fps = total_step * num_envs / duration 
    print(f"Duration = {duration:.2f}s")
    print(f"EnvPool FPS = {fps:.2f}")
    return fps
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", help="number of gym envs", type=int, required=True)
    parser.add_argument("--async_env", help="async mode or not", action="store_true", required=True)
    parser.add_argument("--max_iters", help="maximum number of steps to test", type=int, required=True)
    args = parser.parse_args()
    gym_fps = run_gym(args.n_envs, args.max_iters, args.async_env)
    envpool_fps = run_envpool(args.n_envs, args.max_iters, args.async_env)
    print("~"*50)
    print("Gym Vec env fps: ", gym_fps)
    print("Envpool wrapped fps: ", envpool_fps)
    print("~"*50)