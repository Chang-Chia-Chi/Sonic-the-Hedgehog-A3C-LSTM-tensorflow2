import os
from agent import Agent

weights_path = os.getcwd()+'/weights/training/sonic_weight'
agent = Agent(512, pretrained=False, weights_path=weights_path)
agent.fit()