# Navigation Project - Udacity Deep Reinforcement Learning
Implementation of "Navigation" project from Udacity Deep Reinforcement Learning course

### Getting Started

#### Install requirements
```
conda activate drlnd
pip install -r requirements.txt
```

#### Running
Run the main script to train the agent or load saved weights
```
python src/main.py
```
or 
```
python src/main.py --train
```
Passing the `--train` flag runs in training mode, which will train new weights. Running without this flag will load 
saved weights from a file and run the environment using the trained agent.

