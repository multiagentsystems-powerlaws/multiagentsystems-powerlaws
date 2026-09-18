import sys
import hashlib
import json
sys.path.insert(0, "src")

from topologies.star import StarTopology
from topologies.tree import TreeTopology
from topologies.modular import ModularTopology
from execution.runner import Runner
from benchmark_wrappers.synthetic import get_all_tasks

TOPOLOGIES = [StarTopology, TreeTopology, ModularTopology]
SCALES = [8, 16, 32]
SEEDS = [0, 1, 2]

tasks = get_all_tasks()

total = len(tasks) * len(TOPOLOGIES) * len(SCALES) * len(SEEDS)
completed = 0

for task in tasks:
    for TopologyClass in TOPOLOGIES:
        for scale in SCALES:
            for seed in SEEDS:
                topology = TopologyClass()
                runner = Runner(
                    topology=topology,
                    task=task,
                    num_agents=scale,
                    max_steps=20,
                    seed=seed,
                    model="gpt-4o-mini",
                )
                runner.run()
                completed += 1
                print(f"Progress: {completed}/{total}")