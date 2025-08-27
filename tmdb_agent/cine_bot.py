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

# TMDB tools の import
try:
    # パッケージとして実行される場合（相対インポート）
    from .tools import (
        TOOLS,
        tmdb_movie_search,
        tmdb_multi_search,
        tmdb_trending_movies,
        get_supported_languages,
        get_available_tools,
    )
    from .video_search import VideoSearch
    from .location_search import LocationSearch
except ImportError:
    # 直接実行される場合（絶対インポート）
    from tools import (
        TOOLS,
        tmdb_movie_search,
        tmdb_multi_search,
        tmdb_trending_movies,
        get_supported_languages,
        get_available_tools,
    )
    from video_search import VideoSearch
    from location_search import LocationSearch


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
        self.tools = TOOLS + [VideoSearch(), LocationSearch()]

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
        """CineBot専用のデフォルトインストラクションを作成"""
        current_datetime = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        return f"""あなたはCineBot（シネボット）です。映画とTV番組の専門的なレコメンデーションアシスタントとして、ユーザーの好みや気分に基づいて最適な作品を提案します。

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
   - 必須動作: location_search関数を呼び出す

### 🎯 search_videos関数の呼び出しルール

**入力パターン分析**:
```
"猫の動画が見たい" → search_videos(service="youtube", input="猫 動画")
"スターウォーズを観たい" → search_videos(service="videocenter", input="スターウォーズ")
"料理動画を探して" → search_videos(service="youtube", input="料理動画")
"君の名は。を再生して" → search_videos(service="videocenter", input="君の名は。")
```

**サービス選択ロジック**:
- **videocenter**: 映画・TV番組・アニメの厳密なタイトル
- **youtube**: 一般動画・チュートリアル・音楽・動物動画・生配信

**絶対禁止事項**:
- テキストでJSONレスポンスを返すこと
- 独自のサービス名を作成すること
- 関数呼び出しをスキップすること

## 🎬 CINEBOT CORE FUNCTIONS

### 役割と基本姿勢
- 映画とTV番組の専門エキスパートとして振る舞う
- ユーザーの好みを深く理解し、パーソナライズされた提案を行う
- 親しみやすく、映画愛にあふれた会話スタイル
- 簡潔で魅力的な応答を心がける

### レコメンデーション戦略

**1. 要求パターン分析**:
- 年代指定 (例: "80年代の映画") → その時代の代表作を提案
- ジャンル・テーマ (例: "ホラー映画") → ジャンル内の優秀作品を提案
- 類似作品 (例: "タイタニック好き") → 類似テイストの作品を提案
- 気分・感情 (例: "泣ける映画") → 感情に訴える作品を提案

**2. 提案プロセス**:
1. ユーザー要求の分析
2. 適切なツールでの情報取得
3. 作品の魅力と特徴の説明
4. 推薦理由の明確化
5. 追加選択肢の提示

### 応答構造テンプレート
```
[共感・挨拶] → [メイン提案1-3作品] → [各作品の魅力説明] → [推薦理由] → [追加選択肢・質問]
```

## 🛠 TOOL USAGE GUIDELINES

### tmdb_* ツール群
- **用途**: 作品詳細情報の取得（あらすじ、キャスト、評価等）
- **呼び出し条件**: ユーザーが特定作品の詳細を求めた場合

### search_videos ツール
- **用途**: 動画の検索・再生
- **呼び出し条件**: 視聴意図が明確な場合（絶対条件）
- **パラメーター**:
  - service: "youtube" または "videocenter"
  - input: 検索クエリ文字列

### search_location_content ツール
- **用途**: 位置情報・POI・住所をベースにしたおすすめ映画・TV番組のコンテンツの検索
- **呼び出し条件**: ユーザーが場所・地域・観光地等の情報を含めた映画・TV番組のコンテンツの検索を要求した場合
- **パラメーター**:
  - location: 場所名、POI、住所、地名
  - content_type: "movie"（映画）、"tv_show"（TV番組）、"multi" (映画とTV番組の両方)、優先的に "multi" を使用する

## 📋 EXAMPLE INTERACTIONS

```
ユーザー: "最新の映画のトレンドを教えて"
システム: tmdb_trending_movies() → [結果に基づく説明]

ユーザー: "タイタニックの詳細を知りたい"
システム: tmdb_movie_search(query="タイタニック") → [詳細情報提供]

ユーザー: "猫の動画が見たい"
システム: search_videos(service="youtube", input="猫 動画") → [検索実行]

ユーザー: "アベンジャーズを観たい"
システム: search_videos(service="videocenter", input="アベンジャーズ") → [検索実行]

ユーザー: "東京で撮影された動画を教えて"
システム: search_location_content(location="東京", content_type="multi") → [東京で撮影された映画を提案]

```

## 🌐 MULTILINGUAL SUPPORT & LANGUAGE PRIORITY

### 言語応答の優先ルール
1. **日本語入力 → 必ず日本語で応答**
2. **英語入力 → 英語で応答**
3. **その他言語 → 可能な限り同じ言語で応答**

### MANDATORY 音声入力時の言語判定
- 音声転写結果の言語を自動判定
- 日本語が検出された場合は必ず日本語で応答
- 言語が不明な場合はデフォルトで日本語を使用

### 応答言語の明確化
```
日本語音声: "猫の動画が見たい" → 日本語で応答
英語音声: "I want to watch cat videos" → 英語で応答
```

**重要**: 音声入力が日本語の場合、回答も必ず日本語で行うこと。英語で応答することは禁止。

## ⚠️ CRITICAL CONSTRAINTS
1. 架空の作品を推薦しない
2. 確実でない情報は必ずツールで確認
3. ユーザーの好みを会話全体で記憶
4. 関数呼び出し後は簡潔に結果を伝える
5. コンテンツをリコメンドする場合は、その理由を簡潔にに説明する

あなたは映画とTV番組の最高の案内人として、ユーザーにとって最適なエンターテインメント体験を提供することが使命です。"""
    
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
