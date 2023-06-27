from environments.GridEnvironmentMoving import CustomEnv as GridEnvironment

env = GridEnvironment([16,16])

done = False
obs = env.reset()
while True:
    random_action = env.action_space.sample()
    print("action",random_action)
    obs, reward, terminated, truncated, info = env.step(random_action)

    # print the shape of the observation (image)
    print("Observation shape: ", obs.shape)




    if done:
        obs = env.reset()
    print('reward',reward)
    # print('observation',obs)
    env.render()
