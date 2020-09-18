# Report: Navigation project

### Implementation
- code structure


### Learning Algorithm
Uses DQN

The training was stopped when the average reward for the last 100 episodes was 
greater than the `reward_threshold` of 17.0

#### Neural Net architecture
A neural net with 3 linear layers was used, as well as an output layer. 
The linear layers have sizes 32, 48, 32. T 

#### Hyperparameters


### Results
![reward by episode](img/reward.png "Reward")

![performance](img/banana_navigation.gif "Agent performance")