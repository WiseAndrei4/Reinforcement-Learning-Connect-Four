from FeatureQLearningAgentDriver import QLearningAgent
from SelfPlayTrainer import SelfPlayTrainer

agent=QLearningAgent()
trainer=SelfPlayTrainer(agent, random_chance=0.2)
trainer.run_training(episodes=200000) # 200k
agent.save_weights("weights.json")

print("Done")
