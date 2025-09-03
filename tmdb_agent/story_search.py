from typing import Type
from pydantic import BaseModel, Field
import asyncio
import logging

from langchain.prompts import ChatPromptTemplate
from typing import List

from .base_search import BaseSearchTool


class StorySearchInput(BaseModel):
    query: str = Field(description="物語やアニメの内容に関する自然言語の質問。例: 'エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？'")


class StoryItem(BaseModel):
    title: str = Field(description="作品の公式タイトル")
    description: str = Field(description="1-2文の簡潔な説明（ネタバレなし）")
    reason: str = Field(description="このタイトルが選ばれた理由")
    score: float = Field(description="このタイトルの適合率 (0 < score <= 1): 0は適合しない、1は完全に適合する")


class TopStories(BaseModel):
    items: List[StoryItem] = Field(min_length=0, max_length=10, description="Top 10 by relevance")


class StorySearch(BaseSearchTool):
    name: str = "search_story_content"
    description: str = (
        "物語やアニメの内容に関する自然言語の質問から、関連するアニメ・物語作品をWeb検索し、タイトル・説明・理由を返す機能。"
    )
    args_schema: Type[BaseModel] = StorySearchInput

    def _get_cache_file_name(self) -> str:
        return "story_cache.sqlite"

    def _build_search_query(self, input_data) -> str:
        """Build search query for story content."""
        query = input_data.get("query", "")
        if self.language == "ja":
            return f"{query} 映画 OR TV番組 OR ドラマ OR アニメ OR 物語 OR 作品"
        else:
            return f"{query} movie OR tv show OR drama OR anime OR story OR series"

    def _get_cache_key(self, input_data) -> str:
        """キャッシュキーを生成"""
        return input_data.get("query", "")

    def _get_response_type(self) -> str:
        return "tools.story_search"

    async def _extract_content_parallel(self, raw_results: list, input_data) -> dict:
        """
        各記事ごとにtop3作品を並列で抽出し、重複タイトルを除外して結合する。
        """
        query = input_data.get("query", "")
        
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

    async def _arun(self, query: str):
        """Asynchronous story content search with sqlite cache."""
        input_data = {"query": query}
        return await self._arun_common(input_data)


# StorySearch単体テスト
if __name__ == "__main__":
    import logging
    
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
            result = tool._run(query=query)
            print("Result type:", type(result))
            print("Result preview:", result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            print(f"Error: {e}")
