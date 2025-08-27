try:
    from .vectordb_cache import VectorDBCache, param_hash
except ImportError:
    from vectordb_cache import VectorDBCache, param_hash
import json
import logging
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
import asyncio

from langchain.tools import BaseTool
from langchain_tavily import TavilySearch
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# for PoC
local_vectordb_cache = VectorDBCache()

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


class MediaItem(BaseModel):
    # Title of a well-known film/TV/series/anime
    title: str = Field(description="Official title of the work")
    # 1-2 sentence plain description (no spoilers)
    description: str = Field(description="Short description of the work")

class TopMedia(BaseModel):
    # Exactly 3 items, ordered by global notoriety (most famous first)
    items: List[MediaItem] = Field(min_length=3, max_length=3, description="Top 3 by fame")


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
    _extract_llm: ChatOpenAI = PrivateAttr()
    _vectordb_cache: VectorDBCache = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tavily_search = TavilySearch(max_results=10, topic="general", include_images=False, search_depth="advanced")
        self._extract_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._vectordb_cache = local_vectordb_cache


    def _detect_language(self, text: str) -> str:
        """Detect language from input text."""
        try:
            lang = detect(text)
            if lang in ["ja", "en"]:
                return lang
            return  "ja"
        except LangDetectException:
            return "en"

    def _generate_response(self, videos: list) -> dict:
        """Generate a standardized response for location-based content search."""

        return {
            "type": "tools.location_search",
            "description": """
                This JSON contains items intended to be displayed in the client application
                using components such as GroupButton. Each item includes the necessary
                fields and instructions for proper rendering and interaction.
            """,
            "return_direct": True,
            "selection": {
                "videos": [{
                    "title": video.get("title"),
                    "description": video.get("description")
                } for video in videos]
            },
        }
    
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


    def _extract_top_media(self, raw_results: list, language: str) -> dict:
        """Use LLM to extract Top-3 famous works (film/TV/drama/anime) as strict JSON."""

        # Bind Pydantic schema to enforce strict JSON
        parser_llm = self._extract_llm.with_structured_output(TopMedia)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise extractor of famous audiovisual works. "
             "From the provided web search corpus, identify only works that are FILMS, TV series, live-action dramas, or ANIME. "
             "Exclude people, characters, episodes, songs, books (unless widely known as a film/TV/anime adaptation), news articles, or venues. "
             "Return strictly the JSON that conforms to the provided schema. "
             "Order by global fame/popularity (most famous first). "
             "Keep descriptions concise (1-2 sentences), spoiler-free, and factual."),
            ("system", "Write descriptions in Japanese." if language == "ja" else "Write descriptions in English."),
            ("human",
             "Tavily's output to analyze:\n\n{input}\n\n"
             "Extract exactly the Top 3 most globally famous works (films/TV/dramas/anime) present in the corpus "
             "and return ONLY the strict JSON for the schema.")
        ])

        result = (prompt | parser_llm).invoke({"input": raw_results})
        # Convert to plain dict for JSON serialization by _run()
        return result.model_dump(mode="json")

    def _handle_error(self, error: Exception) -> dict:
        """Handle errors and return a consistent error response."""
        error_message = f"Error in location content search: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun(self, location: str, content_type: str = "multi", language: str = "auto"):
        """Asynchronous movie/TV location content search with vector cache."""
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


            # --- キャッシュメタ情報生成 ---
            meta = {}
            logging.info(f"Cache meta: {meta}")

            # --- キャッシュ検索 ---
            cached, hit, score = self._vectordb_cache.search_with_score(search_query, meta)
            if hit:
                logging.info(f"VectorDBCache HIT (score={score:.4f})")
                return cached
            else:
                logging.info(f"VectorDBCache MISS (score={score:.4f})")

            # Tavilyで検索実行
            search_results = await self._tavily_search.ainvoke({"query": search_query})

            # Format: pick Tavily "results" list; handle list fallback
            if isinstance(search_results, dict):
                raw_results = search_results.get("results", [])
            else:
                raw_results = search_results

            # Use LLM to extract Top-3 famous works (title + description) as strict JSON
            try:
                videos = self._extract_top_media(raw_results, language)
            except Exception as extract_err:
                logging.exception(f"LLM extraction failed: {extract_err}")
                # Fallback to empty structure with correct schema (will be validated by caller if needed)
                videos = {"items": []}

            logging.info(f"videos: {videos}")

            response = self._generate_response(videos.get("items", []))
            logging.info(f"Response: {response}")

            # --- キャッシュ保存 ---
            self._vectordb_cache.add(search_query, meta, response)

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
    logging.basicConfig(level=logging.INFO)
    tool = LocationSearch()
    print("\n--- LocationSearch PoC test ---")
    test_cases = [
        ("草津温泉", "movies", "ja"),
        ("渋谷", "multi", "ja"),
        ("京都", "tv_shows", "auto"),
        ("Tokyo Tower", "movies", "en"),
    ]
    for location, content_type, language in test_cases:
        print(f"\n=== {content_type} | {location} | {language} ===")
        try:
            result = tool._run(location, content_type, language)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
