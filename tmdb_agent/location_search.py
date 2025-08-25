import json
import logging
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
import asyncio

from langchain.tools import BaseTool
from langchain_tavily import TavilySearch
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


class LocationSearchInput(BaseModel):
    location: str = Field(
        description="Location, POI, or address to search for movie/TV show content. Can be a place name, landmark, address, or geographic area where movies/TV shows are set, filmed, or popular."
    )
    content_type: str = Field(
        default="multi",
        description="Type of content to search for. Options: 'movies' (films set in or about the location), 'tv_shows' (TV series set in or about the location), or 'multi' (all entertainment content)."
    )
    language: str = Field(
        default="auto",
        description="Language preference for search results. Use 'auto' for automatic detection, or specify language codes like 'ja', 'en', 'es', etc."
    )


class LocationSearch(BaseTool):
    name: str = "search_location_content"
    description: str = (
        "Function to search for movie and TV show content based on location, POI, or address. "
        "Finds films and TV series set in specific locations, filming locations, local entertainment venues, and location-related entertainment content. "
        "Returns detailed information about movies, TV shows, and entertainment venues associated with the specified location."
    )
    args_schema: Type[BaseModel] = LocationSearchInput
    return_direct: bool = True

    _tavily_search: TavilySearch = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tavily_search = TavilySearch(max_results=10, topic="general", include_images=False, search_depth="advanced")


    def _detect_language(self, text: str) -> str:
        """Detect language from input text."""
        try:
            lang = detect(text)
            if lang in ["ja", "en"]:
                return lang
            return  "ja"
        except LangDetectException:
            return "en"
    
    def _build_search_query(self, location: str, content_type: str, language: str) -> str:
        """Build an optimized search query for location-based movie/TV content."""
        
        # 言語に応じた映画・TV番組関連キーワードマッピング
        content_keywords = {
            "movies": {
                "ja": f"{location}にゆかりのある人気で最近の映画",
                "en": f"famous and recent movies films cinema related to {location}",
            },
            "tv_shows": {
                "ja": f"{location}にゆかりのある人気で最近のTV番組やドラマ",
                "en": f"famous and recent TV shows television series drama shows related to {location}",
            },
            "multi": {
                "ja": f"{location}にゆかりのある人気で最近の映画 TV番組 ドラマ アニメ",
                "en": f"famous and recent movies TV shows content media productions related to {location}",
            }
        }
        
        # 自動言語検出
        if language == "auto":
            language = self._detect_language(location)
        
        # キーワードを取得（デフォルトは英語）
        keywords = content_keywords.get(content_type, content_keywords["multi"])
        query = keywords.get(language)
        
        return query


    def _handle_error(self, error: Exception) -> dict:
        """Handle errors and return a consistent error response."""
        error_message = f"Error in location content search: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun(self, location: str, content_type: str = "multi", language: str = "auto"):
        """Asynchronous movie/TV location content search."""
        try:
            # 自動言語検出
            if language == "auto":
                language = self._detect_language(location)
            
            # サポートされているコンテンツタイプの検証
            supported_types = ["movies", "tv_shows", "multi"]
            if content_type not in supported_types:
                logging.warning(f"Unsupported content type: {content_type}. Using 'multi' instead.")
                content_type = "multi"
            
            logging.info(f"Location = {location}, Content Type = {content_type}, Language = {language}")
            
            # 検索クエリを構築
            search_query = self._build_search_query(location, content_type, language)
            logging.info(f"Search Query = {search_query}")
            
            # Tavilyで検索実行
            search_results = await self._tavily_search.ainvoke({"query": search_query})

            # レスポンス生成
            # resultsが辞書の場合、results部分を取得
            if isinstance(search_results, dict):
                response = search_results.get("results", [])
            else:
                response = search_results
            
            logging.info(f"Response: {response}")
            
            return response
            
        except Exception as e:
            return self._handle_error(e)

    def _run(self, location: str, content_type: str = "multi", language: str = "auto"):
        """Synchronous wrapper around async logic."""
        try:
            return json.dumps(
                asyncio.run(self._arun(location, content_type, language)), 
                indent=4, 
                ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(self._handle_error(e), indent=4, ensure_ascii=False)


# Ensure proper module usage
if __name__ == "__main__":
    # Example usage for movie/TV content search
    tool = LocationSearch()
    
    # Test different types of movie/TV searches
    test_cases = [
        ("温泉", "movies", "ja"),
    ]
    
    for location, content_type, language in test_cases:
        print(f"\n=== Testing {content_type} search for {location} (language: {language}) ===")
        try:
            result = tool._run(location, content_type, language)
            #print(result[:500] + "..." if len(result) > 500 else result)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
