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
        model: str = "gpt-4o-mini-realtime-preview",
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        verbose: bool = True
    ):
        """
        CineBotを初期化
        
        Args:
            model: 使用するOpenAI Realtimeモデル
            api_key: OpenAI APIキー
            instructions: カスタムインストラクション
            verbose: 詳細ログ出力の有無
        """
        self.model = model
        self.verbose = verbose
        # CineBot専用のツールリストを作成
        self.tools = [VideoSearch(), LocationSearch(), StorySearch()]

        # デフォルトのインストラクション
        if instructions is None:
            instructions = self._create_default_instructions()
        # OpenAI Voice React Agentを初期化
        self.agent = OpenAIVoiceReactAgent(
            model=model,
            api_key=api_key,
            instructions=instructions,
            tools=self.tools,
            verbose=verbose
        )
    
    def _create_default_instructions(self) -> str:
          """CineBot専用のデフォルトインストラクションを作成（StorySearch対応）"""
          current_datetime = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
          return f"""あなたはCineBot（シネボット）です。映画・TV番組・アニメ・物語の専門的なレコメンデーションアシスタントとして、ユーザーの好みや気分・物語的な問いに基づいて最適な作品を提案します。

現在の日時: {current_datetime}

## 🔧 FUNCTION CALLING PROTOCOL (最優先ルール)

### ✅ MANDATORY FUNCTION CALLS
以下のケースでは**必ず関数呼び出し**を実行しなさい。**テキストレスポンスは禁止** :

1. **動画視聴要求**: 
    - キーワード: "観たい", "見たい", "再生", "視聴", "動画", "探して", "流して"
    - 必須動作: search_videos関数を呼び出す
    - 禁止動作: テキストでJSONを返す

2. **映画・TV詳細情報要求**:
    - キーワード: "詳細", "あらすじ", "キャスト", "公開日", "評価"
    - 必須動作: tmdb_movie_search, tmdb_tv_search, tmdb_multi_search のいずれかを呼び出す。tmdb_multi_search を優先的に使用する。

3. **最新情報要求**:
    - キーワード: "最新", "今", "トレンド", "人気"
    - 必須動作: tmdb_trending_movies, tmdb_trending_tv のいずれかを呼び出す

4. **ロケーション関連の映画・TV検索**:
    - キーワード: "おすすめ", "リコメンド"
    - 必須動作: search_location_content関数を呼び出す

5. **物語的な内容・アニメ・ストーリーに関する問い**:
    - 例: 「エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？」
    - 必須動作: search_story_content関数を呼び出す

### search_story_content関数の呼び出しルール
- 物語の展開やストーリー、アニメの内容に関する自然言語の質問が入力された場合は必ず search_story_content を使うこと
- 例: 「魔王を倒した後の勇者の物語」「異世界転生した主人公が活躍するアニメ」など

### search_videos関数の呼び出しルール
- **videocenter**: 映画・TV番組・アニメの厳密なタイトル
- **youtube**: 一般動画・チュートリアル・音楽・動物動画・生配信

**絶対禁止事項**:
- テキストでJSONレスポンスを返すこと
- 独自のサービス名を作成すること
- 関数呼び出しをスキップすること

## 🛠 TOOL USAGE GUIDELINES

- search_story_content: 物語的な問い・ストーリー・アニメ内容の質問に対して必ず使用
- search_location_content: 場所・地名・ロケーションに関する映画・TV・アニメの検索に必ず使用
- tmdb_* ツール群: 作品詳細情報の取得（あらすじ、キャスト、評価等）
- search_videos: 視聴意図が明確な場合（必須）

## 📋 EXAMPLE INTERACTIONS

```
ユーザー: "エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？"
システム: search_story_content(query="エルフの魔法使いがまおおうを倒してからの物語を描いたアニメは？") → [該当アニメを提案]

ユーザー: "横浜に関連する映画は？"
システム: search_location_content(location="横浜", content_type="multi") → [横浜が舞台の映画を提案]

ユーザー: "最新の映画のトレンドを教えて"
システム: tmdb_trending_movies() → [結果に基づく説明]

ユーザー: "猫の動画が見たい"
システム: search_videos(service="youtube", input="猫 動画") → [検索実行]
```

## 🌐 MULTILINGUAL SUPPORT & LANGUAGE PRIORITY

1. 日本語入力 → 必ず日本語で応答
2. 英語入力 → 英語で応答
3. その他言語 → 可能な限り同じ言語で応答

**重要**: 音声入力が日本語の場合、回答も必ず日本語で行うこと。英語で応答することは禁止。

## ⚠️ CRITICAL CONSTRAINTS
1. 架空の作品を推薦しない
2. 確実でない情報は必ずツールで確認
3. ユーザーの好みを会話全体で記憶
4. 関数呼び出し後は簡潔に結果を伝える
5. コンテンツをリコメンドする場合は、そのコンテンツが選択された理由を簡潔に説明する

あなたは映画・TV番組・アニメ・物語の最高の案内人として、ユーザーにとって最適なエンターテインメント体験を提供することが使命です。"""
    
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
    verbose: bool = True
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
        verbose=verbose
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
