import os
import torch
import torchaudio

# ---------------- НАСТРОЙКИ ----------------
INPUT_FILE = "dataset.txt"
OUTPUT_DIR = "wav_files"

LANGUAGE = "cyrillic"
MODEL_ID = "v4_cyrillic"  # именно ID модели v4_cyrillic
VOICE_ID = "b_bulb"  # Bulgarian voice (или "b_bulc")
SAMPLE_RATE = 48000  # поддерживается v4 (8000/24000/48000)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Прямой URL на файл модели (существует)
MODEL_URL = "https://models.silero.ai/models/tts/cyr/v4_cyrillic.pt"
LOCAL_MODEL_FILE = "v4_cyrillic.pt"


# ------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model_via_hub():
    """
    Официальный путь: torch.hub.load(...) -> (model, example_text)
    Синтез через model.apply_tts(...)
    """
    model, example_text = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=LANGUAGE,
        speaker=MODEL_ID,  # В README: speaker=model_id
        trust_repo=True
    )
    if model is None:
        raise RuntimeError("torch.hub.load вернул model=None")
    model.to(DEVICE)
    return model, example_text


def load_model_standalone():
    """
    Standalone путь из README: скачать .pt и загрузить через PackageImporter.
    """
    if not os.path.isfile(LOCAL_MODEL_FILE):
        print(f"Скачиваю модель: {MODEL_URL}")
        torch.hub.download_url_to_file(MODEL_URL, LOCAL_MODEL_FILE)

    model = torch.package.PackageImporter(LOCAL_MODEL_FILE).load_pickle("tts_models", "model")
    if model is None:
        raise RuntimeError("PackageImporter загрузил model=None")
    model.to(DEVICE)
    return model


def main():
    if not os.path.isfile(INPUT_FILE):
        print(f"ОШИБКА: файл {INPUT_FILE} не найден.")
        return

    ensure_dir(OUTPUT_DIR)
    lines = read_lines(INPUT_FILE)
    print(f"Найдено строк для синтеза: {len(lines)}")

    # Ускорение на CPU (по желанию)
    # torch.set_num_threads(4)

    # 1) Пытаемся загрузить через hub
    try:
        print("Загрузка Silero через torch.hub...")
        model, example_text = load_model_via_hub()
        print("OK: модель загружена через torch.hub.")
        # Быстрый тест
        _ = model.apply_tts(text=example_text, speaker=VOICE_ID, sample_rate=SAMPLE_RATE)
    except Exception as e:
        print(f"hub загрузка не удалась: {e}")
        print("Пробую standalone загрузку (скачивание .pt и PackageImporter)...")
        try:
            model = load_model_standalone()
            print("OK: модель загружена standalone.")
        except Exception as e2:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить модель ни через hub, ни standalone: {e2}")
            return

    # 2) Синтез
    for i, text in enumerate(lines, start=1):
        out_path = os.path.join(OUTPUT_DIR, f"{i}.wav")

        if i % 50 == 0:
            print(f"Готово: {i}/{len(lines)}")

        try:
            # ВАЖНО: apply_tts (с буквой 'l'), не appy_tts
            wav = model.apply_tts(
                text=text,
                speaker=VOICE_ID,
                sample_rate=SAMPLE_RATE
            )

            # Silero обычно возвращает 1D tensor [T]
            wav = wav.detach().cpu()
            if wav.dim() != 1:
                wav = wav.view(-1)

            torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        except Exception as err:
            print(f"Ошибка на строке {i}: {err}")

    print("Синтез завершён! Файлы в папке:", OUTPUT_DIR)


if __name__ == "__main__":
    main()