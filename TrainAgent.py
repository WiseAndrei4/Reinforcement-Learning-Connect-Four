from FeatureQLearningAgentDriver import FeatureQLearningAgent
from SelfPlayTrainer import SelfPlayTrainer

agent = FeatureQLearningAgent()
trainer = SelfPlayTrainer(agent, random_chance=0.2)
trainer.run_training(episodes=2000000) # 2m
agent.save_weights("weights.json")

print("Done")
