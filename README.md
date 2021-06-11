# multiagent in a social dilemma with a common Pool of resource on Mars
 Topic: Colonising Mars: using reinforcement learning on a multi-agent system where the power supply to the agent’s battery is limited.

Today the world focuses on the research of agents (autonomous systems capable of learning 
their environment and performing an action on it to achieve a specific goal) for various 
purposes. Even in the field of space, research and, technology involve the employment of 
agent rovers (Francis et al., 2019). The last few decades, the space scientists are targeting 
extending human life beyond Earth-like to Mars. Analysing the surrounding state and 
constructing living structures will require multiple agents (multi-agents-a community of 
agents) to speed the process. 
Moreover, the multi-agents create the need for an increased amount of resources to serve the requirements of each agent.
However, resources like fuel are scarce and expensive (Manson, 2018) to serve many agents, 
usually leading to a dilemma for resource usage. Thus, the multi-agents must plan, predict, 
and act in a manner to sustain resources for future use.
Even before the making of multiagents, a simulation model-based study shall aid the creation itself.
The research targets a simulation multi-player game-based model to study agent action, 
predict other agent actions and optimise resources. The research aims employing a learning 
algorithm (help an agent maximise its reward in its environment with a successive trial and 
error method).
Further, the research targets using Q-learning (helps an agent achieve the maximum reward 
over a set of successive steps in a model-free environment) and SARSA to help the agent 
determine a cooperative or competitive approach. The multi-agent is 2-10 in number and can 
eventually grow in the future. The learning algorithms will determine beneficial approaches 
to sustain resources concerning the growing multi-agents.

Methodology: 
We have computed the Q-learning and SARSA algorithms for 2-10 rovers over a 48 hour duration with 220 watts per hour nuclear energy for the first 28 hours and 196 watts per hour for the remaing time.
The learning parameter of Q-learning was varied to levels 0.5-lesser learning rate and 0.9-higher learning rate alpha.

Results:
The Learning in Q-learning showed to be better.
The learning percentage of cooperation was higher for smaller number of multi-agents and lower for higher number of agents but with time and greater number of iterations the agents were able to learn cooperation to a greater extent.

![image](https://user-images.githubusercontent.com/55480687/121642089-649f3e00-cad3-11eb-8366-8c669626e2f5.png)
A screenshot showing the working of 5 agent rovers with a centralized nuclear reactor. Green indicates Sleep Mode and blue indicates turn Working Mode of the hetrogeneous rover.

![image](https://user-images.githubusercontent.com/55480687/121642142-7680e100-cad3-11eb-99ce-d8e9551095e7.png)
Q-learning Vs SARSA

 Execution:
Run the gui.py to see the simulation model.
