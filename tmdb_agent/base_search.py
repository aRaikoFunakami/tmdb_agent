from abc import ABC, abstractmethod
from sqlitedict import SqliteDict
import json
import logging
from typing import Any, Dict, List
from pydantic import PrivateAttr
import asyncio
import random

from langchain.tools import BaseTool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI


USE_PARALLEL_EXTRACTION = True  # True で並列版のコンテンツ検索を使う


class SimpleSqliteCache:
    """シンプルなキー完全一致キャッシュ"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = SqliteDict(self.db_path, autocommit=True)

    def get(self, key: str):
        return self.db.get(key, None)

    def set(self, key: str, value: Any):
        self.db[key] = value

    def close(self):
        self.db.close()


class BaseSearchTool(BaseTool, ABC):
    """検索ツールの共通基底クラス"""
    
    return_direct: bool = True
    
    _tavily_search: TavilySearch = PrivateAttr()
    _extract_llm: ChatOpenAI = PrivateAttr()
    _sqlite_cache: SimpleSqliteCache = PrivateAttr()

    def __init__(self, language=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "language", language)
        self._tavily_search = TavilySearch(
            max_results=15, 
            topic="general",
            include_images=False, 
            search_depth="advanced",
        )
        self._extract_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._sqlite_cache = SimpleSqliteCache(self._get_cache_file_name())
        print(f"{self.__class__.__name__} initialized with language: {language}")

    @abstractmethod
    def _get_cache_file_name(self) -> str:
        """キャッシュファイル名を返す"""
        pass

    @abstractmethod
    def _build_search_query(self, input_data: Any) -> str:
        """検索クエリを構築"""
        pass

    @abstractmethod
    def _extract_content(self, raw_results: List[Any], input_data: Any) -> Dict[str, Any]:
        """コンテンツ抽出（非並列版）"""
        pass

    @abstractmethod
    def _extract_content_parallel(self, raw_results: List[Any], input_data: Any) -> Dict[str, Any]:
        """コンテンツ抽出（並列版）"""
        pass

    @abstractmethod
    def _get_response_type(self) -> str:
        """レスポンスタイプを返す"""
        pass

    @abstractmethod
    def _get_cache_key(self, input_data: Any) -> str:
        """キャッシュキーを生成"""
        pass

    def _check_tmdb_title(self, title: str, original_description: str, original_reason: str) -> dict | None:
        """
        タイトルをTMDB multi searchで確認し、
        - 完全一致があればそのまま返す
        - 類似タイトルがあればTMDB情報でdescription/reasonも上書き
        - 見つからなければNone
        """
        import requests
        import os
        TMDB_API_KEY = os.getenv("TMDB_API_KEY")
        url = "https://api.themoviedb.org/3/search/multi"
        params = {"api_key": TMDB_API_KEY, "query": title, "language": self.language if self.language in ["ja", "en"] else "en"}
        try:
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
            data = res.json()
            results = data.get("results", [])
            logging.info(f"TMDB search for title: {title}, found {len(results)} results")
        except Exception:
            results = []
        if not results:
            logging.info(f"TMDB no match for title: {title}")
            return None
        # 完全一致優先
        for r in results:
            if (r.get("title") == title) or (r.get("name") == title) or (r.get("original_title") == title) or (r.get("original_name") == title):
                return {
                    "title": title,
                    "description": original_description,
                    "reason": original_reason
                }
        # 類似タイトルがあればそちらを採用し、description/reasonもtmdb情報で上書き
        r = results[0]
        tmdb_title = r.get("title") or r.get("name")
        overview = r.get("overview", "")
        return {
            "title": tmdb_title,
            "description": overview if overview else original_description,
            "reason": f"Original title: {title}, Reason: {original_reason if original_reason else 'Not specified'}"
        }

    def _filter_videos_by_tmdb(self, videos: list) -> list:
        """
        動画リストに対してTMDB存在チェックを行い、タイトルの正規化で重複を除外して返す。
        scoreも元videoから引き継ぐ。
        """
        checked_videos = []
        seen_titles = set()
        for video in videos:
            title = video.get("title")
            if not title:
                continue
            checked = self._check_tmdb_title(title, video.get("description"), video.get("reason"))
            if checked:
                norm_title = checked["title"].strip().lower() if checked.get("title") else None
                if norm_title and norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    # scoreを引き継ぐ
                    checked["score"] = video.get("score", 1.0)
                    checked_videos.append(checked)
        return checked_videos

    def _generate_response(self, checked_videos: list, max_result: int = 5) -> dict:
        """共通のレスポンス生成ロジック"""
        # scoreで降順ソート
        sorted_videos = sorted(checked_videos, key=lambda x: x.get("score", 0), reverse=True)
        top2 = sorted_videos[:2] if len(sorted_videos) >= 2 else sorted_videos
        rest = [s for s in sorted_videos[2:]] if len(sorted_videos) > 2 else []
        n_random = max_result - len(top2)
        if n_random > 0 and rest:
            sampled_rest = random.sample(rest, min(n_random, len(rest)))
        else:
            sampled_rest = []
        sampled = top2 + sampled_rest
        
        return {
            "type": self._get_response_type(),
            "description": (
                f"This JSON represents the top {len(sampled)} candidate works (movies, TV shows, anime) related to the specified query.\n"
                "Each element in the 'videos' array contains:\n"
                "- title: The official title of the work\n"
                "- description: A concise 1-2 sentence summary (no spoilers)\n"
                f"- reason: The reason why this work was selected\n"
                f"- score: Relevance score (0 < score <= 1)"
            ),
            "return_direct": True,
            "selection": {
                "videos": sampled
            },
        }

    def _handle_error(self, error: Exception) -> dict:
        """Handle errors and return a consistent error response."""
        error_message = f"Error in {self.__class__.__name__}: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun_common(self, input_data: Any):
        """共通の非同期実行ロジック"""
        try:
            logging.info(f"Input = {input_data}, Language = {self.language}")

            # 検索クエリを構築
            search_query = self._build_search_query(input_data)
            logging.info(f"Search Query = {search_query}")

            # キャッシュキー生成
            cache_key = self._get_cache_key(input_data)
            logging.info(f"Cache key: {cache_key}")

            # キャッシュ検索
            cached = self._sqlite_cache.get(cache_key)
            if cached is not None:
                logging.info(f"SqliteCache HIT: {cache_key}")
                response = self._generate_response(cached)
                logging.info(f"Response: {response}")
                return response
            else:
                logging.info(f"SqliteCache MISS: {cache_key}")
                
                # Tavilyで検索実行
                logging.info("Invoking TavilySearch...")
                search_results = await self._tavily_search.ainvoke({"query": search_query})

                # Format: pick Tavily "results" list; handle list fallback
                if isinstance(search_results, dict):
                    raw_results = search_results.get("results", [])
                else:
                    raw_results = search_results

                # Use LLM to extract content as strict JSON
                logging.info("Invoking LLM for extraction...")
                try:
                    if isinstance(raw_results, list):
                        limited_results = [
                            (r[:750] if isinstance(r, str) else r) for r in raw_results[:15]
                        ]
                    else:
                        limited_results = raw_results
                    
                    if USE_PARALLEL_EXTRACTION:
                        videos = await self._extract_content_parallel(limited_results, input_data)
                    else:
                        videos = self._extract_content(limited_results, input_data)
                except Exception as extract_err:
                    logging.exception(f"LLM extraction failed: {extract_err}")
                    videos = {"items": []}

                logging.info(f"videos: {videos}")

                # TMDB存在チェック済みリストをキャッシュ
                checked_videos = self._filter_videos_by_tmdb(videos.get("items", []))
                self._sqlite_cache.set(cache_key, checked_videos)

                # checked_videos からランダムサンプリングしてレスポンス生成
                response = self._generate_response(checked_videos)
                logging.info(f"Response: {response}")
                return response

        except Exception as e:
            return self._handle_error(e)

    def _run(self, **kwargs):
        """Synchronous wrapper around async logic."""
        try:
            return json.dumps(
                asyncio.run(self._arun_common(kwargs)), 
                indent=4, 
                ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(self._handle_error(e), indent=4, ensure_ascii=False)


# 基底クラスの簡単なテスト
if __name__ == "__main__":
    
    # テスト用の具象クラス
    class TestSearchTool(BaseSearchTool):
        name: str = "test_search"
        description: str = "Test search tool"
        
        def _get_cache_file_name(self) -> str:
            return "test_cache.sqlite"
        
        def _build_search_query(self, input_data) -> str:
            return f"test query: {input_data}"
        
        def _extract_content(self, raw_results, input_data):
            return {"items": []}
        
        async def _extract_content_parallel(self, raw_results, input_data):
            return {"items": []}
        
        def _get_response_type(self) -> str:
            return "tools.test_search"
        
        def _get_cache_key(self, input_data) -> str:
            return f"test_{input_data}"
    
    print("Testing BaseSearchTool...")
    
    # TMDBチェック機能のテスト
    tool = TestSearchTool(language="ja")
    
    # _check_tmdb_titleのテスト（実際のAPIを呼ばずに）
    result = tool._check_tmdb_title("Test Movie", "Test description", "Test reason")
    print(f"TMDB check result: {result}")
    
    # _filter_videos_by_tmdbのテスト
    test_videos = [
        {"title": "Test Movie", "description": "Test desc", "reason": "Test reason", "score": 0.8}
    ]
    filtered = tool._filter_videos_by_tmdb(test_videos)
    print(f"Filtered videos: {filtered}")
    
    # _generate_responseのテスト
    response = tool._generate_response([
        {"title": "Movie 1", "description": "Desc 1", "reason": "Reason 1", "score": 0.9},
        {"title": "Movie 2", "description": "Desc 2", "reason": "Reason 2", "score": 0.8}
    ])
    print(f"Generated response: {response}")
    
    print("BaseSearchTool tests completed!")
