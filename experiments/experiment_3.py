import json
import os
import csv

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
OUTPUT_DIR = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\output"
CSV_EXPORT_PATH = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\experiment_3_results.csv"

# Mapping Experiment 3 Scenarios (Excluding Scenario 4)
EXPERIMENT_3_SCENARIOS = {
    1: "Static Bystander",
    2: "Chaotic Bystander",
    3: "Rhythmic False-Positive"
}

POSITIONS = {
    1: "0 Degrees (Camera Facing)",
    2: "45 Degrees",
    3: "90 Degrees (Side View)"
}

TOTAL_SECONDS = 60

def load_json_data(filepath):
    """Loads a JSON file and returns a dictionary of {second: rate}."""
    with open(filepath, 'r') as file:
        data = json.load(file)
        return {item['second']: item.get('rate') for item in data['data']}

def evaluate_confusion_matrix(output_data):
    """
    Since manual verification confirmed no false-positives (T_wrong = 0),
    any valid rate is T_correct, and any null is T_null.
    """
    t_correct = 0
    t_wrong = 0  # Confirmed via manual video review
    t_null = 0

    for sec in range(TOTAL_SECONDS):
        rate = output_data.get(sec)

        if rate is not None:
            t_correct += 1
        else:
            t_null += 1

    # Calculate Target Acquisition Rate against entire emergency duration
    acc_id = (t_correct / TOTAL_SECONDS) * 100

    return t_correct, t_wrong, t_null, acc_id

def main():
    results = []

    for scenario_id, scenario_name in EXPERIMENT_3_SCENARIOS.items():
        for position_id, position_name in POSITIONS.items():
            
            filename = f"output_3_{scenario_id}_{position_id}.json"
            out_filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(out_filepath):
                out_data = load_json_data(out_filepath)
                
                t_correct, t_wrong, t_null, acc_id = evaluate_confusion_matrix(out_data)

                results.append({
                    "Experiment": 3,
                    "Scenario_ID": scenario_id,
                    "Scenario_Name": scenario_name,
                    "Position_ID": position_id,
                    "Camera_Angle": position_name,
                    "File_Name": filename,
                    "N_Total": TOTAL_SECONDS,
                    "T_Correct": t_correct,
                    "T_Wrong": t_wrong,
                    "T_Null": t_null,
                    "Target_Acquisition_Rate_(%)": round(acc_id, 2)
                })
            else:
                print(f"Missing File: Could not find {filename}")

    # Export to CSV
    if results:
        keys = results[0].keys()
        with open(CSV_EXPORT_PATH, 'w', newline='') as csv_file:
            dict_writer = csv.DictWriter(csv_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nSuccess! Experiment 3 evaluation metrics saved to: {CSV_EXPORT_PATH}")
    else:
        print("\nNo results to save. Ensure files exist and paths are correct.")

if __name__ == "__main__":
    main()