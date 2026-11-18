import torch
import soundfile as sf
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ============== НАСТРОЙКИ ==============
INPUT_FILE = "czech_audio_sentences.txt"
OUTPUT_DIR = "audio_zeros"


# =======================================

def main():
    print("🔧 Проверка: Нулевой вектор (Zero Embeddings)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загрузка модели
    processor = SpeechT5Processor.from_pretrained("fav-kky/SpeechT5-base-cs-tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("fav-kky/SpeechT5-base-cs-tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    # === ВЕКТОР НУЛЕЙ ===
    # Это "ничего" на языке математики. 512 нулей.
    speaker_embeddings = torch.zeros(1, 512).to(device)

    # Берем первое предложение для теста
    text = "Ahoj, slyšíš mě teď?"

    print("⏳ Генерирую тестовый файл...")
    inputs = processor(text=text, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        sf.write("test_zeros.wav", speech.cpu().numpy(), 16000)
        print("✅ Файл 'test_zeros.wav' создан.")
        print("🎧 Послушайте его сейчас.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()