import json
import os
import csv

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Adjust these paths to your specific local directories
GT_DIR = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR"
OUTPUT_DIR = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\output" # Folder with your output_2_y_z.json files
CSV_EXPORT_PATH = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\experiment_2_results.csv"

# Mapping Experiment 2 Scenarios (y) to their respective Ground Truth files.
EXPERIMENT_2_SCENARIOS = {
    1: {"name": "Fatigue Curve", "gt_file": "cpr_data_120_80_.json"},
    2: {"name": "Adrenaline Spike", "gt_file": "cpr_data_90_135.json"},
    3: {"name": "Rescuer Swap", "gt_file": "cpr_data_100_130_snap.json"},
    4: {"name": "Boundary Extreme (60)", "gt_file": "cpr_data_60.json"},
    5: {"name": "Boundary Extreme (160)", "gt_file": "cpr_data_160.json"}
}

# Mapping Positions (z) to descriptive angles for the CSV
POSITIONS = {
    1: "0 Degrees (Camera Facing)",
    2: "45 Degrees",
    3: "90 Degrees (Side View)"
}

TOTAL_SECONDS = 60  # N_total defined in your methodology

def load_json_data(filepath):
    """Loads a JSON file and returns a dictionary of {second: rate}."""
    with open(filepath, 'r') as file:
        data = json.load(file)
        return {item['second']: item.get('rate') for item in data['data']}

def evaluate_trial(gt_data, output_data):
    """Calculates U_rate and MAE for a single video output."""
    n_valid = 0
    absolute_errors = []

    for sec in range(TOTAL_SECONDS):
        gt_rate = gt_data.get(sec)
        out_rate = output_data.get(sec)

        # Check if the system produced a valid numerical output
        if out_rate is not None:
            n_valid += 1
            if gt_rate is not None:
                absolute_errors.append(abs(gt_rate - out_rate))

    # Calculate Uptime (U_rate)
    u_rate = (n_valid / TOTAL_SECONDS) * 100

    # Calculate Mean Absolute Error (MAE) exclusively over valid seconds
    mae = sum(absolute_errors) / n_valid if n_valid > 0 else None

    return n_valid, u_rate, mae

def main():
    results = []

    # Outer Loop: Cycle through Scenarios (y)
    for scenario_id, config in EXPERIMENT_2_SCENARIOS.items():
        scenario_name = config["name"]
        gt_filepath = os.path.join(GT_DIR, config["gt_file"])
        
        if not os.path.exists(gt_filepath):
            print(f"Warning: Ground truth file missing -> {gt_filepath}")
            continue
            
        gt_data = load_json_data(gt_filepath)

        # Inner Loop: Cycle through Positions (z)
        for position_id, position_name in POSITIONS.items():
            
            # Construct the exact expected filename: output_2_y_z.json
            filename = f"output_2_{scenario_id}_{position_id}.json"
            out_filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(out_filepath):
                out_data = load_json_data(out_filepath)
                
                # Run the mathematical evaluation metrics
                n_valid, u_rate, mae = evaluate_trial(gt_data, out_data)

                results.append({
                    "Experiment": 2,
                    "Scenario_ID": scenario_id,
                    "Scenario_Name": scenario_name,
                    "Position_ID": position_id,
                    "Camera_Angle": position_name,
                    "File_Name": filename,
                    "N_Total": TOTAL_SECONDS,
                    "N_Valid": n_valid,
                    "Uptime_Rate_(%)": round(u_rate, 2),
                    "MAE": round(mae, 4) if mae is not None else "N/A (0 Uptime)"
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
        print(f"\nSuccess! Experiment 2 evaluation metrics saved to: {CSV_EXPORT_PATH}")
    else:
        print("\nNo results to save. Check your OUTPUT_DIR path and ensure files exist.")

if __name__ == "__main__":
    main()