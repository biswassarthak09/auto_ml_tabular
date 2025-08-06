import json
import os

input_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(input_dir, "all_optimiztion_results_mahmoud.json")
filenames = [
    'boston_optimization_results',
    'exam_dataset_optimization_results',
    "fri_c1_1000_5_optimization_results",
]
all_results = []

for base_name in filenames:
    filename = base_name + ".json"
    file_path = os.path.join(input_dir, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    else:
        print(f"File not found: {filename}")

with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Saved merged results to {output_file}")