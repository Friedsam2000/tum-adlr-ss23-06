from GridEnvironment import CustomEnv as GridEnvironment

env = GridEnvironment([50,50])

done = False
obs = env.reset()
while True:
    random_action = env.action_space.sample()
    print("action",random_action)
    obs, reward, done, info = env.step(random_action)
    print('reward',reward)
    print('observation',obs)
    env.render()
