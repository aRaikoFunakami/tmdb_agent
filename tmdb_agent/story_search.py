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
import random

USE_PARALLEL_EXTRACTION = True  # True で並列版のコンテンツ検索を使う

# シンプルなキー完全一致キャッシュ
class SimpleSqliteCache:
    def __init__(self, db_path="story_cache.sqlite"):
        self.db_path = db_path
        self.db = SqliteDict(self.db_path, autocommit=True)

    def get(self, key):
        return self.db.get(key, None)

    def set(self, key, value):
        self.db[key] = value

    def close(self):
        self.db.close()

local_sqlite_cache = SimpleSqliteCache()

class StorySearchInput(BaseModel):
    query: str = Field(description="物語やアニメの内容に関する自然言語の質問。例: 'エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？'")

class StoryItem(BaseModel):
    title: str = Field(description="作品の公式タイトル")
    description: str = Field(description="1-2文の簡潔な説明（ネタバレなし）")
    reason: str = Field(description="このタイトルが選ばれた理由")
    score: float = Field(description="このタイトルの適合率 (0 < score <= 1): 0は適合しない、1は完全に適合する")

class TopStories(BaseModel):
    items: List[StoryItem] = Field(min_length=0, max_length=10, description="Top 10 by relevance")

class StorySearch(BaseTool):
    name: str = "search_story_content"
    description: str = (
        "物語やアニメの内容に関する自然言語の質問から、関連するアニメ・物語作品をWeb検索し、タイトル・説明・理由を返す機能。"
    )
    args_schema: Type[BaseModel] = StorySearchInput
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
            include_images=False, search_depth="advanced",
        )
        self._extract_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._sqlite_cache = local_sqlite_cache


        
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
        except Exception as e:
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
            "type": "tools.story_search",
            "description": (
                f"This JSON represents the top {len(sampled)} candidate works (movies, TV shows, anime, stories) related to the specified narrative query.\n"
                "Each element in the 'stories' array contains:\n"
                "- title: Official title\n"
                "- description: 1-2 sentence summary (no spoilers)\n"
                f"- reason: Why this work was selected\n"
                f"- score: Relevance (0 < score <= 1)"
            ),
            "return_direct": True,
            "selection": {
                "videos": sampled
            },
        }

    def _build_search_query(self, query: str) -> str:
        if self.language == "ja":
            return f"{query} 映画 OR TV番組 OR ドラマ OR アニメ OR 物語 OR 作品"
        else:
            return f"{query} movie OR tv show OR drama OR anime OR story OR series"

    def _extract_top_stories(self, raw_results: list, query: str) -> dict:
        parser_llm = self._extract_llm.with_structured_output(TopStories)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an extractor of anime and story works that are explicitly related to the QUERY in the provided corpus. "
                "Works must have clear evidence in the corpus (e.g., plot, character, setting). "
                "Do NOT rely on prior knowledge. Skip any title without explicit evidence. "
            ),
            ("system",
                "Write all descriptions in Japanese." if self.language == "ja" else "Write all descriptions in English."),
            ("human",
                "QUERY: {query}\n\nCorpus:\n{input}\n\n"
                "Extract up to the Top 10 works (anime, story) that have explicit evidence of connection to the query. "
                "If fewer than 10 works have evidence, return fewer. "
                "Return ONLY the strict JSON that conforms to the schema."
                "For the 'title' field, return ONLY the official work title. "
                "Do NOT include article headlines, locations, site names, or descriptive text. "
                "Return strict JSON following the schema. Order primarily by query relevance, then by fame."
                "You MUST find official anime or story titles only."
                "You MUST NOT include same titles."
                "If you cannot find official titles, you MUST NOT return any unofficial titles or placeholders."
                "For the 'score' field, use the following rules:\n"
                "- If the QUERY's main keywords (nouns/verbs) are explicitly and directly mentioned in the reason or description, set score=1.0\n"
                "- If only part of the QUERY is matched, set score between 0.7 and 0.9\n"
                "- If only the general theme is matched (e.g., just 'time travel'), set score between 0.5 and 0.7\n"
                "- If there is little or no relation, set score between 0.0 and 0.4\n"
                "Examples:\n"
                "QUERY: '車でタイムスリップする話'\n"
                " - 'バック・トゥ・ザ・フューチャー' (reason: '車でタイムスリップする話に明確に関連しているため。') → score: 1.0\n"
                " - '時をかける少女' (reason: '女子高生がタイムスリップする物語。') → score: 0.7\n"
                " - 'バブルへGO!!' (reason: 'タイムスリップの要素が明確に描かれているため。') → score: 0.6\n"
            )
        ])
        result = (prompt | parser_llm).invoke({"input": raw_results, "query": query})
        # scoreが無い場合は1.0をデフォルトで補完
        data = result.model_dump(mode="json")
        for item in data.get("items", []):
            if "score" not in item or not isinstance(item["score"], (float, int)):
                item["score"] = 1.0
            # scoreが範囲外なら補正
            if item["score"] > 1.0:
                item["score"] = 1.0
            if item["score"] < 0.0:
                item["score"] = 0.0
        return data

    async def _extract_top_stories_parallel(self, raw_results: list, query: str) -> dict:
        import asyncio
        parser_llm = self._extract_llm.with_structured_output(TopStories)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an extractor of movie, tv show, drama, anime and story works that are explicitly related to the QUERY in the provided corpus. "
                "Works must have clear evidence in the corpus (e.g., plot, character, setting). "
                "Do NOT rely on prior knowledge. Skip any title without explicit evidence. "
            ),
            ("system",
                "Write all descriptions in Japanese." if self.language == "ja" else "Write all descriptions in English."),
            ("human",
                "QUERY: {query}\n\nCorpus:\n{input}\n\n"
                "Extract up to the Top 3 works (movie, tv show, drama, anime, story) that have explicit evidence of connection to the query. "
                "If fewer than 3 works have evidence, return fewer. "
                "Return ONLY the strict JSON that conforms to the schema."
                "For the 'title' field, return ONLY the official work title. "
                "Do NOT include article headlines, locations, site names, or descriptive text. "
                "Return strict JSON following the schema. Order primarily by query relevance, then by fame."
                "You MUST find official movie, tv show, drama, anime, storytitles only."
                "You MUST NOT include same titles."
                "If you cannot find official titles, you MUST NOT return any unofficial titles or placeholders."
            )
        ])

        async def extract_one(article):
            try:
                res = await asyncio.to_thread(
                    lambda: (prompt | parser_llm).invoke({"input": article, "query": query})
                )
                return res.model_dump(mode="json")
            except Exception:
                return {"items": []}

        tasks = [extract_one(article) for article in raw_results]
        results = await asyncio.gather(*tasks)

        seen_titles = set()
        merged_items = []
        for r in results:
            for item in r.get("items", []):
                title = item.get("title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    merged_items.append(item)
        return {"items": merged_items}

    def _handle_error(self, error: Exception) -> dict:
        error_message = f"Error in story content search: {str(error)}"
        logging.error(error_message)
        return {"error": error_message}

    async def _arun(self, query: str):
        try:
            logging.info(f"Query = {query}, Language = {self.language}")

            cache_key = f"{query}"
            logging.info(f"Cache key: {cache_key}")
            
            search_query = self._build_search_query(query)
            logging.info(f"Search Query = {search_query}") 
            
            cached = self._sqlite_cache.get(cache_key)
            if cached is not None:
                logging.info(f"SqliteCache HIT: {cache_key}")
                response = self._generate_response(cached)
                logging.info(f"Response: {response}")
                return response
            else:
                logging.info(f"SqliteCache MISS: {cache_key}")

                logging.info("Invoking TavilySearch...")
                search_results = await self._tavily_search.ainvoke({"query": search_query})
                logging.info("Invoking TavilySearch finished")

                if isinstance(search_results, dict):
                    raw_results = search_results.get("results", [])
                else:
                    raw_results = search_results
                try:
                    if isinstance(raw_results, list):
                        limited_results = [
                            (r[:750] if isinstance(r, str) else r) for r in raw_results[:15]
                        ]
                    else:
                        limited_results = raw_results
                    if USE_PARALLEL_EXTRACTION:
                        videos = await self._extract_top_stories_parallel(limited_results, query)
                    else:
                        videos = self._extract_top_stories(limited_results, query)
                except Exception as extract_err:
                    logging.exception(f"LLM extraction failed: {extract_err}")
                    videos = {"items": []}

                logging.info(f"videos: {videos}")

                # TMDB存在チェック済みリストをキャッシュ
                checked_videos = self._filter_videos_by_tmdb(videos.get("items", []))
                self._sqlite_cache.set(cache_key, checked_videos)

                response = self._generate_response(checked_videos)
                logging.info(f"Response: {response}")
                return response
        except Exception as e:
            return self._handle_error(e)

    def _run(self, query: str):
        try:
            return json.dumps(
                asyncio.run(self._arun(query)), 
                indent=4, 
                ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(self._handle_error(e), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n--- StorySearch PoC test ---")
    test_cases = [
        ("エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？", "ja"),
        ("車でタイムスリップする話", "ja"),
        ("ジュダイが戦う", "en"),
    ]
    for query, language in test_cases:
        print(f"\n=== {query} | {language} ===")
        try:
            tool = StorySearch(language=language)
            result = tool._run(query)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
