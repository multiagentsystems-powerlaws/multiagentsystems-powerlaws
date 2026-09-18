import random
from loggers.schemas import AgentEvent



class BaseAgent:

    def __init__(self, agent_id: str, topology, event_bus, run_id: str):
        self.agent_id = agent_id
        self.topology = topology
        self.event_bus = event_bus
        self.run_id = run_id

    def step(self, step_id: int):
        neighbors = self.topology.neighbors(self.agent_id)
        if not neighbors:
            return
        target = random.choice(neighbors)
        self.event_bus.publish(AgentEvent(
            run_id=self.run_id,
            step=step_id,
            agent_id=self.agent_id,
            event_type="consult",
            target_agent=target,
            topology=self.topology.name(),
            num_agents=len(self.topology.agents),
        ))