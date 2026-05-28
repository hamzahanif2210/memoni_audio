from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key="sk_z3o1p43i_JAgt2xzJiOjUW6u8VPKe4UXz",
)

# Transcribe mode (default)
response = client.speech_to_text.transcribe(
    file=open("/scratch/hamza95/memoni_audio/processed/test/cleaned/00000_seg0090_cleaned.mp3", "rb"),
    model="saaras:v3",
    mode="transcribe"  # or "translate", "verbatim", "translit", "codemix"
)

print(response)

response2 = client.speech_to_text.transcribe(
    file=open("/scratch/hamza95/memoni_audio/processed/test/cleaned/00000_seg0090_cleaned.mp3", "rb"),
    model="saaras:v3",
    mode="translit"  # or "translate", "verbatim", "translit", "codemix"
)

print(response2)