import json

def generate_bpm_json(delay_sec, start_bpm, end_bpm, pre_ramp_sec, ramp_sec, post_ramp_sec, video_file="v2.mp4", filename="cpr_data.json"):
    # Total duration is the initial null delay plus the 3 audio phases
    total_active_seconds = pre_ramp_sec + ramp_sec + post_ramp_sec
    total_seconds = delay_sec + total_active_seconds
    
    data_array = []
    
    for s in range(total_seconds):
        if s < delay_sec:
            # During the delay phase, output null (Python's None becomes null in JSON)
            current_bpm = None
        else:
            # Calculate the time since the actual metronome started
            active_t = s - delay_sec
            
            if active_t < pre_ramp_sec:
                current_bpm = start_bpm
            elif active_t <= pre_ramp_sec + ramp_sec and ramp_sec > 0:
                # Linear interpolation to find the exact BPM during a gradual ramp
                progress = (active_t - pre_ramp_sec) / ramp_sec
                current_bpm = start_bpm + (end_bpm - start_bpm) * progress
            else:
                # We are in the final post-ramp phase (or past a snap transition)
                current_bpm = end_bpm
                
            # Round to nearest integer for clean JSON data, as requested
            current_bpm = int(round(current_bpm))
            
        data_array.append({
            "second": s,
            "rate": current_bpm
        })
        
    # Build the final dictionary to match your required structure
    json_output = {
        "description": "CPR compression rate per second",
        "videoFile": video_file,
        "unit": "compressions/min",
        "data": data_array
    }
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(json_output, f, indent=2)
        
    print(f"Successfully generated {filename} with {total_seconds} seconds of data.")

# --- SET YOUR PARAMETERS HERE ---

# Example based on your request:
# 3 seconds of null (delay)
# 27 seconds at 100 BPM
# 0 seconds ramp (Instant Snap)
# 30 seconds at 120 BPM
# Total = 60 seconds

generate_bpm_json(
    delay_sec=0, 
    start_bpm=119, 
    end_bpm=79, 
    pre_ramp_sec=0, 
    ramp_sec=60, 
    post_ramp_sec=0,
    video_file="v2.mp4"
)