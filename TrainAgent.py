from AgentDriver import QLearningAgent
from SelfPlayTrainer import SelfPlayTrainer
from PlayGame import save_agent

agent=QLearningAgent(alpha=0.1,gamma=0.9,epsilon=0.2)
trainer=SelfPlayTrainer(agent, random_chance=0.2)
trainer.run_training(episodes=600000) # 600k
save_agent(agent,'agent.json')