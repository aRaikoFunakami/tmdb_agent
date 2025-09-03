from typing import Type
from pydantic import BaseModel, Field
import asyncio
import logging

from langchain.prompts import ChatPromptTemplate
from typing import List

from .base_search import BaseSearchTool


class LocationSearchInput(BaseModel):
    location: str = Field(
        description="Location, POI, or address to search for movie/TV show content. Can be a place name, landmark, address, or geographic area where movies/TV shows are set, filmed, or popular."
    )
    content_type: str = Field(
        default="multi",
        description="Type of content to search for. Options: 'movies' (films set in or about the location), 'tv_shows' (TV series set in or about the location), or 'multi' (all entertainment content)."
    )


class MediaItem(BaseModel):
    # Title of a well-known film/TV/series/anime
    title: str = Field(description="Official title of the work.")
    # 1-2 sentence plain description (no spoilers)
    description: str = Field(description="Short description of the work")
    reason: str = Field(description="Reason why this title was selected as one of the top 10 globally famous works")
    score: float = Field(description="Relevance score (0 < score <= 1): 0 is not relevant, 1 is fully relevant")


class TopMedia(BaseModel):
    # Exactly 10 items, ordered by global notoriety (most famous first)
    items: List[MediaItem] = Field(min_length=0, max_length=10, description="Top 10 by relevance")


class LocationSearch(BaseSearchTool):
    name: str = "search_location_content"
    description: str = (
        "Function to search for movie and TV show content based on location, POI, or address. "
        "Finds films and TV series set in specific locations, filming locations, local entertainment venues, and location-related entertainment content. "
        "Returns detailed information about movies, TV shows, and entertainment venues associated with the specified location."
    )
    args_schema: Type[BaseModel] = LocationSearchInput

    def _get_cache_file_name(self) -> str:
        return "location_cache.sqlite"

    def _build_search_query(self, input_data) -> str:
        """Build an optimized search query for location-based movie/TV content."""
        location = input_data.get("location", "")
        content_type = input_data.get("content_type", "multi")
        
        # サポートされているコンテンツタイプの検証
        supported_types = ["movies", "tv_shows", "multi"]
        if content_type not in supported_types:
            logging.warning(f"Unsupported content type: {content_type}. Using 'multi' instead.")
            content_type = "multi"
        
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
        lang = self.language if self.language in ["ja", "en"] else "en"
        keywords = content_keywords.get(content_type, content_keywords["movies"])
        query = keywords.get(lang)
        return query

    def _get_cache_key(self, input_data) -> str:
        """キャッシュキーを生成"""
        return self._build_search_query(input_data)

    def _get_response_type(self) -> str:
        return "tools.location_search"

    def _extract_content(self, raw_results: list, input_data) -> dict:
        """Use LLM to extract Top-10 famous works (film/TV/drama/anime) as strict JSON, with score."""
        location = input_data.get("location", "")
        
        parser_llm = self._extract_llm.with_structured_output(TopMedia)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an extractor of audiovisual works (films, TV series, dramas, anime) that are explicitly connected to the LOCATION in the provided corpus. "
                "Works must have clear evidence in the corpus (e.g., set in, filmed in, story takes place in). "
                "Do NOT rely on prior knowledge. Skip any title without explicit evidence. "
            ),
            ("system",
                "Write all descriptions in Japanese." if self.language == "ja" else "Write all descriptions in English."),
            ("human",
                "LOCATION: {location}\n\nCorpus:\n{input}\n\n"
                "Extract up to the Top 10 works (movies, tv shows, anime) that have explicit evidence of connection to the location. "
                "If fewer than 10 works have evidence, return fewer. "
                "Return ONLY the strict JSON that conforms to the schema."
                "For the 'title' field, return ONLY the official work title. "
                "Do NOT include article headlines, locations, site names, or descriptive text. "
                "Return strict JSON following the schema. Order primarily by location relevance, then by global fame."
                "You MUST find official titles only."
                "You MUST NOT include same titles."
                "If you cannot find official titles, you MUST NOT return any unofficial titles or placeholders."
                "For the 'score' field, use the following rules:\n"
                "- If the LOCATION is explicitly and directly mentioned in the reason or description, set score=1.0\n"
                "- If only part of the LOCATION is matched, set score between 0.7 and 0.9\n"
                "- If only the general area/theme is matched, set score between 0.5 and 0.7\n"
                "- If there is little or no relation, set score between 0.0 and 0.4\n"
                "Examples:\n"
                "LOCATION: '渋谷'\n"
                " - '渋谷怪談' (reason: '渋谷が舞台のホラー映画。') → score: 1.0\n"
                " - '君の名は。' (reason: '東京が舞台の一部。') → score: 0.7\n"
                " - 'ロスト・イン・トランスレーション' (reason: '東京の様々な場所が登場。') → score: 0.6\n"
                "LOCATION: 'Shibuya'\n"
                " - 'Shibuya Kaidan' (reason: 'A horror movie set in Shibuya.') → score: 1.0\n"
                " - 'Your Name.' (reason: 'Partly set in Tokyo.') → score: 0.7\n"
                " - 'Lost in Translation' (reason: 'Features various locations in Tokyo.') → score: 0.6\n"
            )
        ])
        result = (prompt | parser_llm).invoke({"input": raw_results, "location": location})
        # scoreが無い場合は1.0をデフォルトで補完
        data = result.model_dump(mode="json")
        for item in data.get("items", []):
            if "score" not in item or not isinstance(item["score"], (float, int)):
                item["score"] = 1.0
            if item["score"] > 1.0:
                item["score"] = 1.0
            if item["score"] < 0.0:
                item["score"] = 0.0
        return data

    async def _extract_content_parallel(self, raw_results: list, input_data) -> dict:
        """
        各記事ごとにtop3作品を並列で抽出し、重複タイトルを除外して結合する。
        """
        location = input_data.get("location", "")
        
        parser_llm = self._extract_llm.with_structured_output(TopMedia)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an extractor of audiovisual works (films, TV series, dramas, anime) that are explicitly connected to the LOCATION in the provided corpus. "
                "Works must have clear evidence in the corpus (e.g., set in, filmed in, story takes place in). "
                "Do NOT rely on prior knowledge. Skip any title without explicit evidence. "
            ),
            ("system",
                "Write all descriptions in Japanese." if self.language == "ja" else "Write all descriptions in English."),
            ("human",
                "LOCATION: {location}\n\nCorpus:\n{input}\n\n"
                "Extract up to the Top 3 works (movies, tv shows, anime) that have explicit evidence of connection to the location. "
                "If fewer than 3 works have evidence, return fewer. "
                "Return ONLY the strict JSON that conforms to the schema."
                "For the 'title' field, return ONLY the official work title. "
                "Do NOT include article headlines, locations, site names, or descriptive text. "
                "Return strict JSON following the schema. Order primarily by location relevance, then by global fame."
                "You MUST find official titles only."
                "You MUST NOT include same titles."
                "If you cannot find official titles, you MUST NOT return any unofficial titles or placeholders."
                "For the 'score' field, use the following rules:\n"
                "- If the LOCATION is explicitly and directly mentioned in the reason or description, set score=1.0\n"
                "- If only part of the LOCATION is matched, set score between 0.7 and 0.9\n"
                "- If only the general area/theme is matched, set score between 0.5 and 0.7\n"
                "- If there is little or no relation, set score between 0.0 and 0.4\n"
                "Examples:\n"
                "LOCATION: '渋谷'\n"
                " - '渋谷怪談' (reason: '渋谷が舞台のホラー映画。') → score: 1.0\n"
                " - '君の名は。' (reason: '東京が舞台の一部。') → score: 0.7\n"
                " - 'ロスト・イン・トランスレーション' (reason: '東京の様々な場所が登場。') → score: 0.6\n"
                "LOCATION: 'Shibuya'\n"
                " - 'Shibuya Kaidan' (reason: 'A horror movie set in Shibuya.') → score: 1.0\n"
                " - 'Your Name.' (reason: 'Partly set in Tokyo.') → score: 0.7\n"
                " - 'Lost in Translation' (reason: 'Features various locations in Tokyo.') → score: 0.6\n"
            )
        ])

        async def extract_one(article):
            try:
                res = await asyncio.to_thread(
                    lambda: (prompt | parser_llm).invoke({"input": article, "location": location})
                )
                # scoreが無い場合は1.0をデフォルトで補完
                data = res.model_dump(mode="json")
                for item in data.get("items", []):
                    if "score" not in item or not isinstance(item["score"], (float, int)):
                        item["score"] = 1.0
                    if item["score"] > 1.0:
                        item["score"] = 1.0
                    if item["score"] < 0.0:
                        item["score"] = 0.0
                return data
            except Exception:
                return {"items": []}

        # 並列で各記事ごとに抽出
        tasks = [extract_one(article) for article in raw_results]
        results = await asyncio.gather(*tasks)

        # すべてのitemsを集約し、タイトル重複を除外
        seen_titles = set()
        merged_items = []
        for r in results:
            for item in r.get("items", []):
                title = item.get("title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    merged_items.append(item)
        return {"items": merged_items}

    async def _arun(self, location: str, content_type: str = "multi"):
        """Asynchronous movie/TV location content search with sqlite cache."""
        input_data = {"location": location, "content_type": content_type}
        return await self._arun_common(input_data)


# LocationSearch単体テスト
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- LocationSearch PoC test ---")
    test_cases = [
        ("パリ", "movies", "ja"),
        ("Tokyo Tower", "movies", "en"),
    ]
    for location, content_type, language in test_cases:
        print(f"\n=== {content_type} | {location} | {language} ===")
        try:
            tool = LocationSearch(language=language)
            result = tool._run(location=location, content_type=content_type)
            print("Result type:", type(result))
            print("Result preview:", result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            print(f"Error: {e}")
