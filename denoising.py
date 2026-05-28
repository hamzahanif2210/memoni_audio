import soundfile as sf
import dpdfnet

input_file = "/project/ctb-stelzer/hamza95/00006.mp3"
output_file = "/project/ctb-stelzer/hamza95/00006_enhanced.wav"

# Load audio
audio, sr = sf.read(input_file)

# Enhance audio using dpdfnet8_48khz_hr
enhanced = dpdfnet.enhance(
    audio,
    sample_rate=sr,
    model="dpdfnet8_48khz_hr",
    attn_limit_db=12
)

# Save enhanced audio
sf.write(output_file, enhanced, sr)

print(f"Enhanced audio saved to: {output_file}")