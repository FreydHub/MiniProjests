import os
import shutil
import sys
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Путь к папке установки
ESPEAK_HOME = r"C:\Program Files\eSpeak NG"
ESPEAK_DLL = os.path.join(ESPEAK_HOME, "libespeak-ng.dll")
ESPEAK_DATA = os.path.join(ESPEAK_HOME, "espeak-ng-data")

if os.path.exists(ESPEAK_DLL):
    print(f"Библиотека найдена: {ESPEAK_DLL}")

    # 1. Указываем библиотеку
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = ESPEAK_DLL

    # 2. Указываем путь к словарям (ЛЕЧИТ ОШИБКУ "failed to load voice")
    if os.path.exists(ESPEAK_DATA):
        os.environ["ESPEAK_DATA_PATH"] = ESPEAK_DATA
        print(f"✅ Словари найдены: {ESPEAK_DATA}")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Папка espeak-ng-data не найдена!")

    # 3. Добавляем в системный путь
    os.environ["PATH"] += ";" + ESPEAK_HOME
else:
    print("eSpeak не найден. Будет использоваться медленный режим.")



def patched_symlink(src, dst, target_is_directory=False):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    if target_is_directory or os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


os.symlink = patched_symlink

script_dir = os.path.dirname(os.path.abspath(__file__))
toucan_path = os.path.join(script_dir, "IMS-Toucan")
sys.path.append(toucan_path)

local_model_path = os.path.join(script_dir, "Models", "ToucanTTS_Meta", "best.pt")
if not os.path.exists(local_model_path):
    print("Нет файла модели (best.pt).")
    exit()

try:
    from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
except ImportError:
    print("Не найдена папка IMS-Toucan.")
    exit()

INPUT_FILE = "czech_audio_sentences.txt"
OUTPUT_DIR = "audio_toucan"


def main():
    print("\nЗапуск ToucanTTS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель
    tts = ToucanTTSInterface(device=device, tts_model_path="Meta")

    print("🇨🇿 Язык: Czech (ces)")
    tts.set_language("ces")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Нет файла {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Генерация {len(texts)} файлов...")
    print("Если eSpeak заработал - будет быстро. Если нет - медленно, но надежно.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0

    for i, text in enumerate(tqdm(texts)):
        output_path = os.path.join(OUTPUT_DIR, f"{i + 1}.wav")
        if os.path.exists(output_path):
            continue

        try:
            # Генерация
            output = tts(text)

            # === УНИВЕРСАЛЬНАЯ ОБРАБОТКА (ЧТОБЫ НЕ ПАДАЛО) ===

            # 1. Извлекаем аудио из кортежа
            if isinstance(output, tuple):
                audio = output[0]
            else:
                audio = output

            # 2. Проверяем формат (Tensor или Numpy)
            if hasattr(audio, "cpu"):
                # Если сработал eSpeak -> это Tensor
                audio_np = audio.cpu().numpy()
            else:
                # Если сработал Transphone -> это Numpy
                audio_np = audio

            # 3. Сохраняем
            sf.write(output_path, audio_np, 24000)
            success_count += 1

        except Exception as e:
            if i < 3: print(f"⚠️ Ошибка ({i}): {e}")
            pass

        if i % 200 == 0:
            torch.cuda.empty_cache()

    print(f"\n🎉 ГОТОВО! Создано: {success_count} файлов.")


if __name__ == "__main__":
    main()