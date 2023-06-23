import gym

# Create the Cartpole environment
env = gym.make('CartPole-v1')

# Reset the environment
observation = env.reset()

done = False
while not done:
    # Render the environment
    env.render()

    # Take a random action
    action = env.action_space.sample()

    # Perform the action
    observation, reward, done, info = env.step(action)[0], env.step(action)[1], env.step(action)[2], env.step(action)[3]

# Close the environment
env.close()

