## Explanation this package ##

This package is a Recurrent Behavior Cloning.

And it is compatible with [Imitation library](https://github.com/HumanCompatibleAI/imitation)

It is okay to use a expert dataset which is from human , whether it has recurrent state or not (like lstm_state or gru_state).

## Training ##

```python3 train_gru_bc.py```

## result in BipedalWalker-v3 ##

BC loss (ent_weight = 1e-3 , l2_weight = 0.0)

![image](https://github.com/CAI23sbP/RecurrentBC/assets/108871750/0eeba33d-1968-4971-80ca-8a73caade7cc)



## Capability ##

Pytorch == 1.12.1

Stable-baselines3 == 2.0.0

Sb3-contrib == 2.0.o

Imitation == 1.0.0

## Sister GRU packages ##

[RecurrentRLHF](https://github.com/CAI23sbP/RecurrentRLHF) (Preference based RL with Recurrent reward model)

[GRU_AC](https://github.com/CAI23sbP/GRU_AC) (Actor-critic or Proximal Policy Optimizer with GRU)


## references ##

BipedalWalker policy's hyper-paramter [[git repo](https://github.com/andri27-ts/Reinforcement-Learning/blob/master/Week5/PPO.py)]

GRU BC reference [[git repo](https://github.com/Ram81/pirlnav)] [[paper](https://arxiv.org/pdf/2301.07302)]
