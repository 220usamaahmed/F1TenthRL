# Hyperparameter tuning

## PPO

What are our tunable paramters?

According to ChatGPT these should be part of the study.
See if we really need to optimize for all of these and if other PPO paramters may also need to be optimized

1. learning_rate
2. n_steps
3. batch_size
4. n_epochs
5. gamma
6. gae_lambda
7. clip_range
8. ent_coef
9. vf_coef
10. use_sde
11. target_kl
12. feature_extractor features_dim
13. net_arch pi, vf

- What reward function to use when training/evaluating?
- What env to use for evaluation?
