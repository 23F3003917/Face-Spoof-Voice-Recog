import os
from utils.audio_utils import record_audio
from utils.embedding_utils import extract_embedding, cosine_similarity
from utils.storage import load_embeddings

def recognize_user(expected_user):
    os.makedirs("recognition_inputs", exist_ok=True)

    # Auto-increment filename
    count = len([f for f in os.listdir("recognition_inputs") if f.endswith(".wav")])
    filename = f"input_{count + 1}.wav"
    audio_path = os.path.join("recognition_inputs", filename)

    print("🎙️ Recording voice for 10 seconds...")
    record_audio(audio_path, duration=10)
    print(f"✅ Saved input to {audio_path}")

    input_embedding = extract_embedding(audio_path)
    stored_embeddings = load_embeddings()

    if expected_user not in stored_embeddings:
        print(f"❌ No stored voice embedding found for user: {expected_user}")
        return 0.0  # or raise an exception

    expected_embedding = stored_embeddings[expected_user]
    similarity = cosine_similarity(input_embedding, expected_embedding)
    print(f"🔍 Similarity with {expected_user}: {similarity:.4f}")

    return similarity

if __name__ == "__main__":
    user = input("Enter expected user name: ").strip().lower()
    recognize_user(user)
