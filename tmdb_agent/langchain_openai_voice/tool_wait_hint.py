import os
import base64
import requests


# 言語ごとのテキスト・voice・ファイル名設定
VOICE_HINT_CONFIG = {
    "ja": {
        "text": " 調査をしています。少々お待ちください",
        "voice": "sage",
        "filename": "_tool_wait_hint_sage.wav",
    },
    "en": {
        "text": " I’m looking this up for you, just a moment!",
        "voice": "sage",
        "filename": "_tool_wait_hint_sage_en.wav",
    },
}



def ensure_tool_wait_hint_voice(language: str = "ja"):
    """
    指定言語のツール実行ウェイト音声ファイルがなければOpenAI TTSで生成し、base64で返す。
    既にあればファイルからbase64で返す。
    language: "ja" (デフォルト) または "en" など
    """
    config = VOICE_HINT_CONFIG.get(language, VOICE_HINT_CONFIG["ja"])
    path = os.path.join(os.path.dirname(__file__), config["filename"])
    if not os.path.exists(path):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4o-mini-tts",
            "input": config["text"],
            "voice": config["voice"],
            "response_format": "wav",
            "speed": 1.0,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
    # base64エンコードして返す
    with open(path, "rb") as f:
        wav_bytes = f.read()
    return base64.b64encode(wav_bytes).decode("ascii")
