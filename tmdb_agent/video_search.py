import json
import logging
from typing import Any, Type
from pydantic import BaseModel, Field
import asyncio

from langchain.tools import BaseTool
from sudachipy import tokenizer, dictionary
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# 形態素解析して SearcH API に適した形式に変換するための関数
TOKENIZER = dictionary.Dictionary().create()
MODE = tokenizer.Tokenizer.SplitMode.B

def tokenize_text(text):
    return [m.surface() for m in TOKENIZER.tokenize(text, MODE)]

class VideoSearchInput(BaseModel):
    service: str = Field(
        description='Name of video website for video search. Use "videocenter" for movies and TV shows, "youtube" for general video content (tutorials, music, etc.).'
    )
    input: str = Field(description="Search string for searching videos.")


class VideoSearch(BaseTool):
    name: str = "search_videos"
    description: str = (
        "Function to search videos on a specified service. Use 'videocenter' for movies and TV shows, 'youtube' for general video content."
        "The VideoSearch tool not only searches, but also actually plays the content specified by the user. "
    )
    args_schema: Type[BaseModel] = VideoSearchInput
    return_direct: bool = True

    def _generate_response(self, service: str, input: str) -> dict:
        """Generate a standardized response for video search."""
        lang_code = "en" 
        # 言語コードを決定
        try:
            lang_code = detect(input)
            # サポートされている言語のみ対応、それ以外は英語
        except LangDetectException:
            # 言語検出に失敗した場合は英語をデフォルトとする
            lang_code = "en"

        print(f"Detected language code: {lang_code}")
        # 日本語の場合は形態素解析を行う
        if( lang_code == "ja" ):
            input = " ".join(tokenize_text(input))

        return {
            "type": "tools.search_videos",
            "description": """
                This JSON describes an action where the client application should open a web browser and search for the specified query.
                The action is specified in the "intent" field, and additional instructions are provided in the "instructions" field.
                """,
            "return_direct": True,
            "intent": {
                "webbrowser": {
                    "search_videos": {
                        "service": service,
                        "input": input,
                    },
                },
            },
        }

    def _handle_error(self, error: Exception) -> dict:
        """Handle errors and return a consistent error response."""
        error_message = f"Error in searching videos: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun(self, service: str, input: str):
        """Asynchronous video search."""
        try:
            service = service.lower()
            
            # Validate service
            supported_services = ["videocenter", "youtube"]
            if service not in supported_services:
                raise ValueError(f"Unsupported service: {service}. Supported services: {', '.join(supported_services)}")
            
            logging.info(f"Service = {service}, Input = {input}")

            response = self._generate_response(service, input)
            logging.info(f"Response: {response}")
            return response
        except Exception as e:
            return self._handle_error(e)

    def _run(self, service: str, input: str):
        """Synchronous wrapper around async logic."""
        try:
            return json.dumps(
                asyncio.run(self._arun(service, input)), indent=4, ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(self._handle_error(e), indent=4, ensure_ascii=False)


# Ensure proper module usage
if __name__ == "__main__":
    # Example usage with both services
    tool = VideoSearch()
    
    # Test videocenter (movies/TV)
    print("=== Testing videocenter service (movies/TV) ===")
    result1 = tool._run("videocenter", "Star Wars")
    print(result1)
    
    # Test youtube (general videos)
    print("\n=== Testing youtube service (general videos) ===")
    result2 = tool._run("youtube", "cooking tutorials")
    print(result2)
