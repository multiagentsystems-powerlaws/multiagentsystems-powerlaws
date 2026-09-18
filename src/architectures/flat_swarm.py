from agents.llm_agent import LLMAgent


class FlatSwarmArchitecture:

    def build_agents(self, num_agents: int, topology, event_bus, run_id: str,
                     task_description: str, model: str = "gpt-4o-mini"):
        return [
            LLMAgent(
                agent_id=f"A{i}",
                topology=topology,
                event_bus=event_bus,
                run_id=run_id,
                task_description=task_description,
                model=model,
            )
            for i in range(num_agents)
        ]