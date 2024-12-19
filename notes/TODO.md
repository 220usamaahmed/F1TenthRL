# TODO

- [x] Create SB3 compatable wrapper for F110Env
- [x] Create Agent interface
- [x] Create dummy agent
- [x] Create agent/wrapper for PPO model
- [x] Create custom feature extractor for PPO model
- [x] Add action recording to env and create playback agent to debug runs
- [x] Add save and load model feature for RL based models
- [x] Pass LIDAR params to env
- [x] Control frame rate during visualization
- [x] Use normalized action and observations space values and scale according to env params
- [x] Fix camera movement in visualization
- [x] Add visulization for action and observation values
- [x] Explore optuna for hyper paramter tuning
- [x] Save rewards, observations and info in recording. There is probably an issue with NaN in there
- [x] Save hyperparamter and env details with model saves
- [x] Save optuna study
- [x] Change env to end after one lap
- [x] Save intermediate models
- [x] Add additional agents as obsticles 
- [x] Add additional running agents
- [ ] Change env map on reset
- [ ] Use random seeding during training
- [ ] Use optima to find loss function component weights
- [ ] Add reward based on lap time
- [ ] Explore imitation learning

# Reading List
- [ ] https://www.youtube.com/watch?v=aTDkYFZFWug
- [ ] https://xbpeng.github.io/projects/DeepMimic/index.html
- [ ] State representation learning, Decoupling Feature Extration from Policy Learning, Augmented Autoencoders
- [ ] gSDE (Generalized State-Dependent Exploration)
- [ ] Reward normalization?
- [ ] https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
- [ ] https://github.com/f1tenth/f1tenth_racetracks?tab=readme-ov-file
