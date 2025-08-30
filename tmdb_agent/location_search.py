from sqlitedict import SqliteDict
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

# シンプルなキー完全一致キャッシュ
class SimpleSqliteCache:
    def __init__(self, db_path="location_cache.sqlite"):
        self.db_path = db_path
        self.db = SqliteDict(self.db_path, autocommit=True)

    def get(self, key):
        return self.db.get(key, None)

    def set(self, key, value):
        self.db[key] = value

    def close(self):
        self.db.close()

local_sqlite_cache = SimpleSqliteCache()

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
    title: str = Field(description="Official title of the work.")
    # 1-2 sentence plain description (no spoilers)
    description: str = Field(description="Short description of the work")
    reason: str = Field(description="Reason why this title was selected as one of the Top 3 globally famous works")

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
    _sqlite_cache: SimpleSqliteCache = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tavily_search = TavilySearch(
            max_results=10, 
            topic="general",
            include_images=False, search_depth="advanced",
            #include_raw_content=True  # 重要
        )
        self._extract_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._sqlite_cache = local_sqlite_cache


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
            "description": (
                "This JSON represents the top 3 candidate works (movies, TV shows, anime) related to the specified location.\n"
                "Each element in the 'videos' array contains:\n"
                "- title: The official title of the work\n"
                "- description: A concise 1-2 sentence summary (no spoilers)\n"
                "- reason: The reason why this work was selected as a top 3 candidate (e.g., global fame, awards, clear connection to the location)\n"
                "This information allows users to discover globally notable works associated with the location."
            ),
            "return_direct": True,
            "selection": {
                "videos": [{
                    "title": video.get("title"),
                    "description": video.get("description"),
                    "reason": video.get("reason")
                } for video in videos]
            },
        }
    
    def _build_search_query(self, location: str, content_type: str, language: str) -> str:
        """Build an optimized search query for location-based movie/TV content."""
        
        # 言語に応じた映画・TV番組関連キーワードマッピング
        # _build_search_query の keywords を強化（例）
        content_keywords = {
            "movies": {
                "ja": f'"{location}" 映画 (舞台 OR ロケ地 OR 撮影地 OR セット) -観光 -旅行ガイド',
                "en": f'"{location}" (film OR movie) (set in OR filmed in OR location OR setting) -tourism -travel guide',
            },
            "tv_shows": {
                "ja": f'"{location}" (TV OR ドラマ) (舞台 OR ロケ地 OR 撮影地)',
                "en": f'"{location}" ("tv series" OR drama) (set in OR filmed in)',
            },
            "multi": {
                "ja": f'"{location}" (映画 OR TV OR ドラマ OR アニメ) (舞台 OR ロケ地 OR 撮影地)',
                "en": f'"{location}" (film OR "tv series" OR anime) (set in OR filmed in)',
            },
        }
        
        # 自動言語検出
        if language == "auto":
            language = self._detect_language(location)
        
        # キーワードを取得（デフォルトは英語）
        keywords = content_keywords.get(content_type, content_keywords["movies"]) # PoC では movies のみサポートする
        query = keywords.get(language)
        
        return query


    def _extract_top_media(self, raw_results: list, language: str, location: str) -> dict:
        """Use LLM to extract Top-3 famous works (film/TV/drama/anime) as strict JSON."""
        parser_llm = self._extract_llm.with_structured_output(TopMedia)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an extractor of audiovisual works (films, TV series, dramas, anime) that are explicitly connected to the LOCATION in the provided corpus. "
                "Works must have clear evidence in the corpus (e.g., set in, filmed in, story takes place in). "
                "Do NOT rely on prior knowledge. Skip any title without explicit evidence. "
            ),
            ("system",
                "Write all descriptions in Japanese." if language == "ja" else "Write all descriptions in English."),
            ("human",
                "LOCATION: {location}\n\nCorpus:\n{input}\n\n"
                "Extract up to the Top 3 works (movies) that have explicit evidence of connection to the location. "
                "If fewer than 3 works have evidence, return fewer. "
                "Return ONLY the strict JSON that conforms to the schema."
                "For the 'title' field, return ONLY the official work title (e.g., 'Oshin', 'Spirited Away'). "
                "Do NOT include article headlines, locations, site names, or descriptive text. "
                "Good examples: 'Oshin', 'Star Wars', 'Your Name', 'Thermae Romae'. "
                "Bad examples: 'TV drama Oshin filming location at Ginzan Onsen snowy scenery', 'Star Wars official site', 'Your Name set in Shinkai’s hometown'. "
                "Return strict JSON following the schema. Order primarily by location relevance, then by global fame."
                "You MUST find official movie titles only."
                "You MUST NOT find tv show titles, youtube content titles, and other titles."
                "You MUST NOT include same titles."
                "If you cannot find official movie titles, you MUST NOT return any unofficial titles or placeholders."
            )
        ])
        result = (prompt | parser_llm).invoke({"input": raw_results, "location": location})
        return result.model_dump(mode="json")

    def _handle_error(self, error: Exception) -> dict:
        """Handle errors and return a consistent error response."""
        error_message = f"Error in location content search: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun(self, location: str, content_type: str = "multi", language: str = "auto"):
        """Asynchronous movie/TV location content search with sqlite cache."""
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

            # --- キャッシュキー生成（完全一致） ---
            cache_key = f"{search_query}|{content_type}|{language}"
            logging.info(f"Cache key: {cache_key}")

            # --- キャッシュ検索 ---
            cached = self._sqlite_cache.get(cache_key)
            if cached is not None:
                logging.info(f"SqliteCache HIT")
                return cached
            else:
                logging.info(f"SqliteCache MISS")

            # Tavilyで検索実行
            logging.info("Invoking TavilySearch...")
            search_results = await self._tavily_search.ainvoke({"query": search_query})

            # Format: pick Tavily "results" list; handle list fallback
            if isinstance(search_results, dict):
                raw_results = search_results.get("results", [])
            else:
                raw_results = search_results

            # Use LLM to extract Top-3 famous works (title + description) as strict JSON
            logging.info("Invoking LLM for extraction...")
            try:
                # LLMに渡す検索結果を最大5件、かつ各要素を1000文字以内にtruncate
                if isinstance(raw_results, list):
                    limited_results = [
                        (r[:1000] if isinstance(r, str) else r) for r in raw_results[:5]
                    ]
                else:
                    limited_results = raw_results
                videos = self._extract_top_media(limited_results, language, location)
            except Exception as extract_err:
                logging.exception(f"LLM extraction failed: {extract_err}")
                # Fallback to empty structure with correct schema (will be validated by caller if needed)
                videos = {"items": []}

            logging.info(f"videos: {videos}")

            logging.info("Generating response...")
            response = self._generate_response(videos.get("items", []))
            logging.info(f"Response: {response}")

            # --- キャッシュ保存 ---
            self._sqlite_cache.set(cache_key, response)
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
