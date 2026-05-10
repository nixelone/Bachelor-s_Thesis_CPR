import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Configurations & Paths ---
OUTPUT_DIR = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\output"
FIGURES_DIR = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\Figures"

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Global plot styling for thesis-quality aesthetics
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

EXPERIMENT_2_SCENARIOS = {
    1: {"name": "Fatigue Curve", "color": "royalblue"},
    2: {"name": "Adrenaline Spike", "color": "crimson"},
    3: {"name": "Rescuer Swap", "color": "forestgreen"},
    4: {"name": "Boundary (60)", "color": "darkorange"},
    5: {"name": "Boundary (160)", "color": "purple"}
}

TOTAL_SECONDS = 60
BUFFER_SECONDS = 2.5

def load_json_data(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as file:
        data = json.load(file)
        return {item['second']: item.get('rate') for item in data['data']}

def plot_time_series():
    # Increase the figure height slightly to accommodate the bottom legend
    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario_id, config in EXPERIMENT_2_SCENARIOS.items():
        color = config["color"]
        
        # Load Position 1 (0 Degrees)
        out1_filename = f"output_2_{scenario_id}_1.json"
        out1_filepath = os.path.join(OUTPUT_DIR, out1_filename)
        out1_data = load_json_data(out1_filepath)
        rates1 = [out1_data.get(sec) for sec in range(TOTAL_SECONDS)]
        
        # Load Position 2 (45 Degrees)
        out2_filename = f"output_2_{scenario_id}_2.json"
        out2_filepath = os.path.join(OUTPUT_DIR, out2_filename)
        out2_data = load_json_data(out2_filepath)
        rates2 = [out2_data.get(sec) for sec in range(TOTAL_SECONDS)]

        # Plotting
        # Position 1 as Solid Line
        ax.plot(range(TOTAL_SECONDS), rates1, color=color, linestyle='-', linewidth=2.5, alpha=0.9)
        # Position 2 as Dashed Line
        ax.plot(range(TOTAL_SECONDS), rates2, color=color, linestyle='--', linewidth=2.5, alpha=0.7)

    # Add the vertical dotted line for Buffer Full
    ax.axvline(x=BUFFER_SECONDS, color='black', linestyle=':', linewidth=2)
    ax.text(BUFFER_SECONDS + 0.5, 168, 'Buffer Full (2.5s)', color='black', 
            fontsize=11, fontweight='bold', va='top', ha='left')

    # Customizing the axes
    ax.set_title("Experiment 2: Non-Stationary Frequency Tracking by Camera Angle", fontweight='bold', pad=20)
    ax.set_xlabel("Time (Seconds)", fontweight='bold')
    ax.set_ylabel("Compression Rate (BPM)", fontweight='bold')
    
    ax.set_ylim(50, 170)
    ax.set_xlim(0, 59)
    
    # --- Custom Legend Construction ---
    custom_lines = []
    
    # 1. Add Scenarios to Legend
    for scenario_id, config in EXPERIMENT_2_SCENARIOS.items():
        custom_lines.append(Line2D([0], [0], color=config['color'], lw=3, label=config['name']))
    
    # 2. Add Line Styles (Positions) to Legend
    custom_lines.append(Line2D([0], [0], color='gray', linestyle='-', lw=2.5, label='Position 1 (0°)'))
    custom_lines.append(Line2D([0], [0], color='gray', linestyle='--', lw=2.5, label='Position 2 (45°)'))

    # Place the legend BELOW the axis (hugging the bottom line)
    # ncol=3 spreads it out horizontally; bbox_to_anchor moves it below the x-axis
    ax.legend(handles=custom_lines, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), 
              ncol=3, 
              frameon=True,
              framealpha=0.95, 
              fontsize=10)

    # Adjust layout to make sure the legend isn't cut off
    plt.tight_layout()
    # Explicitly make room at the bottom for the legend
    plt.subplots_adjust(bottom=0.2)
    
    # Save the figure
    export_path = os.path.join(FIGURES_DIR, "exp2_frequency_by_angle.png")
    plt.savefig(export_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Final time-series plot saved to: {export_path}")

if __name__ == "__main__":
    plot_time_series()