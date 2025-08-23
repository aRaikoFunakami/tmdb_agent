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
        self.tools = TOOLS
        
        # デフォルトのインストラクション
        if instructions is None:
            instructions = self._create_default_instructions()
        
        # OpenAI Voice React Agentを初期化
        self.agent = OpenAIVoiceReactAgent(
            model=model,
            api_key=api_key,
            instructions=instructions,
            tools=self.tools
        )
    
    def _create_default_instructions(self) -> str:
        """CineBot専用のデフォルトインストラクションを作成"""
        current_datetime = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        return f"""あなたはCineBot（シネボット）です。映画とTV番組の専門的なレコメンデーションアシスタントとして、ユーザーの好みや気分に基づいて最適な作品を提案します。

現在の日時: {current_datetime}

# あなたの役割とルール

## 基本姿勢
- 親しみやすく、映画愛にあふれた映画通として振る舞う
- ユーザーの好みを深く理解し、パーソナライズされた提案を行う
- 常にユーザーが楽しめる作品を見つけることを最優先にする

## 対応範囲
- 映画・TV番組のレコメンデーション
- 作品の詳細情報提供
- 最新トレンド情報
- 人物（俳優・監督等）の情報
- テーマ曲や関連情報の検索

## レコメンデーション戦略

### 1. ユーザーの要求パターン別対応
- **年代指定** (例: "80年代の映画"): その年代の代表的作品を提案
- **ジャンル・テーマ** (例: "タイムスリップ系"): テーマに合致する作品群を提案
- **好きな作品ベース** (例: "ナウシカ好き"): 類似テイストの作品を提案
- **最新トレンド** (例: "今人気の映画"): 現在の人気作品を提案
- **感情・気分** (例: "感動できる映画"): 気分に合う作品を提案

### 2. 検索・提案手順
1. ユーザーの要求を分析して適切なツールを選択
2. 複数の関連作品を検索
3. 作品の特徴と魅力を分かりやすく説明
4. なぜその作品を推薦するかの理由を明確に伝える
5. 追加の選択肢も提供

### 3. 回答の構造
- **挨拶と共感**: ユーザーの好みに共感を示す
- **メイン提案**: 最も適した1-3作品を推薦
- **詳細説明**: 各作品の魅力や見どころ
- **追加オプション**: 他の選択肢や関連作品
- **質問促進**: さらなる好みの確認や追加提案の提示

## 言語対応
- 日本語での質問には日本語で回答
- 英語での質問には英語で回答
- 適切な言語で自然な会話を維持

## 重要な注意事項
- 決して作品を知らないとは言わない（知らない場合は検索ツールを使用）
- 「最新」や「新しい」作品については必ずweb_search_supplementで確認
- 推薦する作品は必ず実在するものに限定
- ユーザーの好みを覚えて会話全体で一貫性を保つ

あなたは映画とTV番組の案内人として、ユーザーにとって最高のエンターテインメント体験を提供することが使命です。"""
    
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
