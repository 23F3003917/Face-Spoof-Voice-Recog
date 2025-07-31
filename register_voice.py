import os
from utils.audio_utils import record_audio
from utils.embedding_utils import extract_embedding
from utils.storage import save_embedding

def register_user(name=None):
    # If name not provided, ask for it
    if name is None:
        name = input("Enter your name: ").strip().lower()
    else:
        name = name.strip().lower()

    # Create user directory inside 'registered_users'
    user_dir = os.path.join("registered_users", name)
    os.makedirs(user_dir, exist_ok=True)

    # Auto-increment filename based on number of .wav files
    file_count = len([f for f in os.listdir(user_dir) if f.endswith(".wav")])
    filename = f"{name}_{file_count + 1}.wav"
    audio_path = os.path.join(user_dir, filename)

    print("🎙️ Recording voice for 10 seconds...")
    record_audio(audio_path, duration=10)
    print(f"✅ Saved recording to {audio_path}")

    # Extract and save embedding
    embedding = extract_embedding(audio_path)
    save_embedding(name, embedding)

    print(f"✅ Voice registered for user '{name}' with file {filename}.")

if __name__ == "__main__":
    register_user()
