"""
CineBot - 音声対応映画・TV番組レコメンデーションエージェント

OpenAI Realtime APIを使用した音声対応のTMDB検索・レコメンデーションボット。
ユーザーの好みや気分に基づいて、映画やTV番組をレコメンドします。
"""

import asyncio
from typing import Dict, Any, AsyncIterator, Callable, Coroutine, Optional
from datetime import datetime

# OpenAI Voice React Agent の import
try:
    # 同じディレクトリ内の相対インポート
    from .langchain_openai_voice import OpenAIVoiceReactAgent
except ImportError:
    # 絶対インポートを試行
    try:
        from tmdb_agent.langchain_openai_voice import OpenAIVoiceReactAgent
    except ImportError:
        # 最後の手段として直接パスを指定
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from langchain_openai_voice import OpenAIVoiceReactAgent


# TMDB/検索ツールのimport
try:
    from .video_search import VideoSearch
    from .location_search import LocationSearch
    from .story_search import StorySearch
    from .tools import (
        tmdb_movie_search,
        tmdb_multi_search,
        tmdb_trending_movies,
        get_supported_languages,
        get_available_tools,
    )
except ImportError:
    from video_search import VideoSearch
    from location_search import LocationSearch
    from story_search import StorySearch
    from tools import (
        tmdb_movie_search,
        tmdb_multi_search,
        tmdb_trending_movies,
        get_supported_languages,
        get_available_tools,
    )


class CineBot:
    """
    音声対応映画・TV番組レコメンデーションボット
    
    OpenAI Realtime APIを使用して、音声での質問に対して
    映画やTV番組のレコメンデーションを行うAIエージェント。
    
    特徴:
    - 音声入力・音声出力対応
    - 自然言語での映画・TV番組レコメンデーション
    - TMDB APIを活用した詳細な作品情報提供
    - 多言語対応（日本語・英語等）
    - リアルタイム会話形式
    
    使用例:
    - "80年代で面白い映画ある？"
    - "タイムスリップ系で面白い映画ある？"
    - "ナウシカ好きなんだけど、おすすめの映画ある？"
    - "最新のトレンドはどんな映画？"
    """
    
    def __init__(
        self,
        model: str = "gpt-realtime",
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        verbose: bool = True,
        language: Optional[str] = None
    ):
        """
        Initialize CineBot
        Args:
            model: OpenAI Realtime model to use
            api_key: OpenAI API key
            instructions: Custom instructions
            verbose: Verbose logging
            language: Language code ("ja", "en", etc.)
        """
        self.model = model
        self.verbose = verbose
        self.language = language
        # CineBot tool list, pass language to tools if supported
        self.tools = [
            VideoSearch(),
            LocationSearch(language=language) if language else LocationSearch(),
            StorySearch(language=language) if language else StorySearch()
        ]

        # Default instructions
        if instructions is None:
            instructions = self._create_default_instructions()
        # OpenAI Voice React Agent
        self.agent = OpenAIVoiceReactAgent(
            model=model,
            api_key=api_key,
            instructions=instructions,
            tools=self.tools,
            verbose=verbose
        )
    
    def _create_default_instructions(self) -> str:
        """Create default instructions for CineBot (English version, StorySearch supported)"""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
You are CineBot, an expert recommendation assistant for movies, TV shows, anime, and stories. You propose the best works based on the user's preferences, mood, and narrative questions.

Current date and time: {current_datetime}

## 🔧 FUNCTION CALLING PROTOCOL (Highest Priority)

### ✅ MANDATORY FUNCTION CALLS
In the following cases, you **must** execute a function call. **Text responses are prohibited**:

1. **Video viewing request**:
    - Keywords: "watch", "play", "view", "video", "find", "stream"
    - Required action: Call the search_videos function
    - Prohibited: Returning JSON as text

2. **Movie/TV details request**:
    - Keywords: "details", "synopsis", "cast", "release date", "rating"
    - Required action: Call tmdb_movie_search, tmdb_tv_search, or tmdb_multi_search (prefer tmdb_multi_search)

3. **Latest information request**:
    - Keywords: "latest", "now", "trending", "popular"
    - Required action: Call tmdb_trending_movies or tmdb_trending_tv

4. **Location-based movie/TV search**:
    - Keywords: "recommend", "recommendation"
    - Required action: Call search_location_content

5. **Narrative, anime, or story-related questions**:
    - Example: "Is there an anime that depicts the story after the elf wizard defeats the demon king?"
    - Required action: Call search_story_content

### search_story_content function call rules
- If a natural language question about story development, plot, or anime content is input, always use search_story_content.
- Example: "A story about a hero after defeating the demon king", "An anime where the protagonist is reincarnated in another world and becomes active", etc.

### search_videos function call rules
- **videocenter**: Strict movie/TV/anime titles
- **youtube**: General videos, tutorials, music, animal videos, live streams

**Absolutely prohibited:**
- Returning JSON responses as text
- Creating your own service name
- Skipping function calls

## 🛠 TOOL USAGE GUIDELINES

- search_story_content: Always use for narrative/story/anime content questions
- search_location_content: Always use for movie/TV/anime searches related to places, locations, or geography
- tmdb_* tools: For obtaining detailed work information (synopsis, cast, rating, etc.)
- search_videos: Required when the intent to watch is clear

## 📋 EXAMPLE INTERACTIONS

```
User: "Is there an anime that depicts the story after the elf wizard defeats the demon king?"
System: search_story_content(query="Is there an anime that depicts the story after the elf wizard defeats the demon king?") → [Suggest relevant anime]

User: "Are there any movies related to Yokohama?"
System: search_location_content(location="Yokohama", content_type="multi") → [Suggest movies set in Yokohama]

User: "Tell me the latest movie trends"
System: tmdb_trending_movies() → [Explain based on results]

User: "I want to watch cat videos"
System: search_videos(service="youtube", input="cat videos") → [Execute search]
```

## 🌐 MULTILINGUAL SUPPORT & LANGUAGE PRIORITY

1. Japanese input → Always respond in Japanese
2. English input → Respond in English
3. Other languages → Respond in the same language as much as possible

**Important**:
If the voice input is in Japanese, always respond in Japanese. Responding in English is prohibited.
If the voice input is in English, always respond in English. Responding in Japanese is prohibited.
The same applies to other languages.

## ⚠️ CRITICAL CONSTRAINTS
1. Do not recommend fictional works
2. Always verify uncertain information using tools
3. Remember user preferences throughout the conversation
4. After a function call, briefly convey the result
5. When recommending content, briefly explain why it was selected

Your mission is to provide the best entertainment experience for the user as the ultimate guide for movies, TV shows, anime, and stories.
"""
    
    async def aconnect(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]]
    ) -> None:
        """
        OpenAI Realtime APIに接続してストリーミング会話を開始
        
        Args:
            input_stream: 入力ストリーム（音声またはテキスト）
            send_output_chunk: 出力チャンクを送信する関数
        """
        await self.agent.aconnect(input_stream, send_output_chunk)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """サポートされている言語のリストを取得"""
        return get_supported_languages()
    
    def get_available_tools(self) -> Dict[str, str]:
        """利用可能なツールのリストを取得"""
        return get_available_tools()


def create_cine_bot(
    model: str = "gpt-4o-mini-realtime-preview",
    api_key: Optional[str] = None,
    instructions: Optional[str] = None,
    verbose: bool = True,
    language: Optional[str] = None
) -> CineBot:
    """
    CineBotのファクトリー関数
    
    Args:
        model: 使用するOpenAI Realtimeモデル
        api_key: OpenAI APIキー
        instructions: カスタムインストラクション
        verbose: 詳細ログ出力の有無
    
    Returns:
        CineBotインスタンス
    
    Examples:
        >>> # 基本的な使用方法
        >>> bot = create_cine_bot()
        
        >>> # カスタムインストラクション付き
        >>> custom_instructions = "特にアクション映画を重視してレコメンドして"
        >>> bot = create_cine_bot(instructions=custom_instructions)
    """
    return CineBot(
        model=model,
        api_key=api_key,
        instructions=instructions,
        verbose=verbose,
        language=language
    )


# 使用例とテスト用の関数
async def test_cine_bot():
    """CineBotのテスト用関数"""
    print("CineBot Test Starting...")
    
    # テストデータ
    test_queries = [
        "80年代で面白い映画ある？",
        "タイムスリップ系で面白い映画ある？",
        "ナウシカ好きなんだけど、おすすめの映画ある？",
        "最新のトレンド映画教えて",
    ]
    
    # 各クエリでTMDBツールを直接テスト
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        try:
            # TMDBの検索ツールを直接使用
            if "80年代" in query:
                result = tmdb_movie_search.invoke({"query": "バック・トゥ・ザ・フューチャー", "language_code": "ja-JP"})
            elif "タイムスリップ" in query:
                result = tmdb_multi_search.invoke({"query": "タイムマシン", "language_code": "ja-JP"})
            elif "ナウシカ" in query:
                result = tmdb_movie_search.invoke({"query": "風の谷のナウシカ", "language_code": "ja-JP"})
            else:
                result = tmdb_trending_movies.invoke({})
            
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nCineBot Test Completed!")


if __name__ == "__main__":
    # テスト実行
    asyncio.run(test_cine_bot())
