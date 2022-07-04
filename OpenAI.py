from gym import envs
print(envs.registry.all())


import gym
env = gym.make('Acrobot-v1')
dir(env.action_space)
env.action_space.n
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()




import gym
env = gym.make('MsPacman-v0')
env.action_space
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("Obs.",observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("action",action)
        print("reward",reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


import pandas as pd


record = {

    'Name': ['Ankit', 'Amit', 'Aishwarya', 'Priyanka', 'Priya', 'Shaurya'],
    'Age': [21, 19, 20, 18, 17, 21],
    'Stream': ['Math', 'Commerce', 'Science', 'Math', 'Math', 'Science'],
    'Percentage': [88, 92, 95, 70, 65, 78]}

Dataframe = pd.DataFrame(record, columns = ['Name', 'Age', 'Stream', 'Percentage'])

Dataframe[Dataframe["Name"]=="Amit"].iloc[:,1].unique()[0]