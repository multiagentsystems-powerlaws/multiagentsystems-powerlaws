import sys, os
sys.path.insert(0, 'src')

from topologies.star import StarTopology
from loggers.event_bus import EventBus
from langchain_openai import ChatOpenAI
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, including OPENAI_API_KEY

run_dir = Path('data/debug/star/n3/s0/test')
run_dir.mkdir(parents=True, exist_ok=True)
bus = EventBus(run_dir)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
topo = StarTopology(
    num_agents=3,
    llm=llm,
    bus=bus,
    run_id='debug_star',
    benchmark='MARBLE',
    task_id='test',
    task_family='test',
    difficulty='easy',
    seed=0,
    architecture='hub_and_spoke',
    snapshot_every=5,
)
try:
    result = topo.run('What is 2+2?')
    print('SUCCESS:', result[:200] if result else '(empty)')
    import subprocess
    out = subprocess.check_output(
        ['wc', '-l', 'data/debug/star/n3/s0/test/events.jsonl'],
        stderr=subprocess.DEVNULL
    ).decode().split()[0]
    print(f'events logged: {out}')
except Exception as e:
    import traceback
    traceback.print_exc()