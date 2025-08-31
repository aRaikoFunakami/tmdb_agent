import os
import base64
import requests

VOICE_HINT_PATH = os.path.join(os.path.dirname(__file__), "_tool_wait_hint_sage.wav")
VOICE_HINT_TEXT = "少々お待ちください"
VOICE_HINT_LANG = "ja"
VOICE_HINT_VOICE = "sage"


def ensure_tool_wait_hint_voice():
    """
    ツール実行時のウェイト音声ファイルがなければOpenAI TTSで生成し、base64で返す。
    既にあればファイルからbase64で返す。
    """
    if not os.path.exists(VOICE_HINT_PATH):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4o-mini-tts",
            "input": VOICE_HINT_TEXT,
            "voice": VOICE_HINT_VOICE,
            "response_format": "wav",
            "speed": 1.2,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        with open(VOICE_HINT_PATH, "wb") as f:
            f.write(response.content)
    # base64エンコードして返す
    with open(VOICE_HINT_PATH, "rb") as f:
        wav_bytes = f.read()
    return base64.b64encode(wav_bytes).decode("ascii")
