from websocietysimulator import Simulator
from websocietysimulator.llm import LLMBase, MyLLM
#from example.myagent2_base import MyRecommendationAgent
from example.RecAgent_baseline import MyRecommendationAgent
import os
import torch

simulator = Simulator(data_dir="./processed_data", device="auto", cache=False)

task_set = "yelp"  # "goodreads"d or "yelp"

simulator.set_task_and_groundtruth(
    task_dir=f"./example/track2/{task_set}/tasks",
    groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
)

simulator.set_agent(MyRecommendationAgent)

# Set LLM client
simulator.set_llm(MyLLM(port="12345"))

outputs_path = f"./example/track2/{task_set}/agent_outputs_baseline.pt"

if os.path.exists(outputs_path):
    print(f"Loading agent outputs from {outputs_path}")
    agent_outputs = torch.load(outputs_path, map_location="cpu")
else:
    print("No saved outputs found. Running simulation...")
    agent_outputs = simulator.run_simulation(
        number_of_tasks=None,
        enable_threading=True,
        max_workers=10,
    )
    torch.save(agent_outputs, outputs_path)
    print(f"Saved agent outputs to {outputs_path}")

simulator.simulation_outputs = agent_outputs

bad_indices = []

for idx, out in enumerate(agent_outputs):
    if out is None:
        print(f"[DEBUG] simulation_outputs[{idx}] is None")
        bad_indices.append(idx)
        continue

    if not isinstance(out, dict):
        print(f"[DEBUG] simulation_outputs[{idx}] is not a dict: {type(out)} -> {out}")
        bad_indices.append(idx)
        continue

    pred = out.get("output", None)

    if pred is None:
        print(f"[DEBUG] simulation_outputs[{idx}]['output'] is None")
        print("        Task info:", out.get("task"))
        bad_indices.append(idx)
    elif not isinstance(pred, list):
        print(f"[DEBUG] simulation_outputs[{idx}]['output'] is not a list")
        print(f"        Type: {type(pred)}, Value: {pred}")
        print("        Task info:", out.get("task"))
        bad_indices.append(idx)

print(f"[DEBUG] Total problematic outputs: {len(bad_indices)}")

evaluation_results = simulator.evaluate()
print(evaluation_results)
