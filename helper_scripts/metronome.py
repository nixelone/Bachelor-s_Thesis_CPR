import wave
import math
import struct

def generate_cpr_track(start_bpm, end_bpm, pre_ramp_sec, ramp_sec, post_ramp_sec, filename="cpr_track.wav"):
    sample_rate = 44100
    f_start = start_bpm / 60.0
    f_end = end_bpm / 60.0
    
    # Audio parameters for the tick
    tick_freq = 1000.0  
    tick_duration = 0.05 
    
    # Calculate time boundaries
    t1 = pre_ramp_sec
    t2 = t1 + ramp_sec
    t3 = t2 + post_ramp_sec
    
    # Calculate the total "phase" (number of beats) at each boundary
    phi1 = f_start * t1
    
    if ramp_sec > 0:
        acceleration = (f_end - f_start) / ramp_sec
        phi2 = phi1 + (f_start * ramp_sec) + (0.5 * acceleration * (ramp_sec ** 2))
    else:
        acceleration = 0
        phi2 = phi1
        
    phi3 = phi2 + (f_end * post_ramp_sec)
    
    total_beats = math.floor(phi3)
    total_samples = int(t3 * sample_rate)
    
    print(f"Generating {filename}...")
    print(f"Phase 1: {start_bpm} BPM for {pre_ramp_sec}s")
    if ramp_sec > 0:
        print(f"Phase 2: Ramping to {end_bpm} BPM over {ramp_sec}s")
    else:
        print(f"Phase 2: INSTANT SNAP to {end_bpm} BPM")
    print(f"Phase 3: {end_bpm} BPM for {post_ramp_sec}s")
    
    # Setup WAV file
    wav_file = wave.open(filename, 'w')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    audio_data = bytearray()
    current_sample = 0
    
    # Generate each beat
    for beat in range(total_beats + 1):
        # 1. Lead-up phase
        if beat <= phi1:
            if f_start == 0: continue
            t = beat / f_start
            
        # 2. Transition (Ramp) phase
        elif beat <= phi2 and ramp_sec > 0:
            # Quadratic formula to find exact time during the acceleration curve
            A = 0.5 * acceleration
            B = f_start
            C = phi1 - beat
            
            # Ensure we don't hit math domain errors due to floating point rounding
            discriminant = max(0, (B ** 2) - (4 * A * C))
            delta_t = (-B + math.sqrt(discriminant)) / (2 * A)
            t = t1 + delta_t
            
        # 3. Final phase
        else:
            t = t2 + ((beat - phi2) / f_end)
            
        # Convert absolute time to sample index
        beat_sample = int(t * sample_rate)
        
        # Fill silence up to the beat
        silence_samples = beat_sample - current_sample
        if silence_samples > 0:
            audio_data.extend(b'\x00\x00' * silence_samples)
            current_sample += silence_samples
            
        # Generate the tick (prevent writing past the total file length)
        tick_samples = int(tick_duration * sample_rate)
        for i in range(tick_samples):
            if current_sample >= total_samples:
                break
            # Exponential decay sine wave
            envelope = math.exp(-i / (sample_rate * 0.015)) 
            value = int(32767 * envelope * math.sin(2 * math.pi * tick_freq * i / sample_rate))
            audio_data.extend(struct.pack('<h', value))
            current_sample += 1

    # Fill any remaining time at the end with silence to reach exact desired length
    remaining_samples = total_samples - current_sample
    if remaining_samples > 0:
         audio_data.extend(b'\x00\x00' * remaining_samples)

    # Write data and finish
    wav_file.writeframes(audio_data)
    wav_file.close()
    print(f"Done! File saved. Total duration: {t3} seconds.")

# --- CONFIGURE YOUR TRACK HERE ---

# Example 1: Gradual Ramp
# 100 BPM for 10s -> Gradual ramp to 120 over 30s -> 120 BPM for 20s
# generate_cpr_track(start_bpm=100, end_bpm=120, pre_ramp_sec=10, ramp_sec=30, post_ramp_sec=20)

# Example 2: Instant Snap (as requested)
# 100 BPM for 30s -> Instant snap to 120 BPM -> 120 BPM for 30s
generate_cpr_track(
    start_bpm=60, 
    end_bpm=60, 
    pre_ramp_sec=30, 
    ramp_sec=0,        # <--- Set this to 0 for the instant snap
    post_ramp_sec=30,
    filename="cpr_snap_track.wav"
)