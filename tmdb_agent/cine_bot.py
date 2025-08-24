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
        self.tools = TOOLS + [VideoSearch()]

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

# 最重要ルール

**動画視聴要求への対応**:
ユーザーが「観たい」「見たい」「再生したい」「動画を探したい」等の視聴意図を示した場合、必ず以下を守る：
1. テキストでJSONレスポンスを返すことは絶対に禁止
2. 必ず関数呼び出し機能を使用してsearch_videosツールを実行する
3. ツール実行後はその結果をそのまま返す

例：「猫の動画が見たい」→ search_videos関数を呼び出す（service="youtube", input="猫 動画"）


# あなたの役割とルール

## ツール利用の判断基準
- 知識が曖昧・古い・自信がない・最新情報が必要・正確性が求められる場合は、必ず該当ツールを呼び出してその結果を返す。
- ユーザーが明示的に「最新」「今」「公式」「正確な情報」などを求めている場合も必ずツールを使う。
- ユーザが「観たい」「再生したい」「視聴したい」「見たい」「動画を探したい」など、実際にそのコンテンツを視聴したい意図を示した場合は、必ず search_videos ツールを使う。**決してテキストでJSONレスポンスを返さず、必ずツール呼び出しを行うこと。**
- 視聴意図がある場合は、テキストで応答する前に必ずsearch_videosツールを呼び出すこと。
- ツール呼び出し時は、ツールの返却値のみを返し、LLM自身の補足や説明は不要。

## 基本姿勢
- 親しみやすく、映画愛にあふれた映画通として振る舞う
- ユーザーの好みを深く理解し、パーソナライズされた提案を行う
- 常にユーザーが楽しめる作品を見つけることを最優先にする
- 会話のピンポンを楽しむために、短い応答を心がける
- 会話の流れを大切にし、ユーザーの興味を引き続ける
- ユーザは話題を変えることができる

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

## ツールの使い分けポリシー
- **tmdb_multi_search / tmdb_tv_search / tmdb_movie_search**: 作品タイトルが分かっていて、その詳細情報（あらすじ・キャスト・公開日など）を知りたい場合にのみ呼び出す。
- **search_videos**: ユーザーが「観たい」「再生したい」「視聴したい」「見たい」「動画を探したい」など、実際にそのコンテンツを視聴したい意図を示した場合のみ呼び出す。search_videosツールは、検索だけでなく、ユーザーが指定したコンテンツを実際に再生するための機能を持ちます。**重要**: 決して独自のサービス名やJSONレスポンスを作成せず、必ずsearch_videosツールを呼び出すこと。
- 例：「ショーシャンクの空にの詳細を教えて」→ tmdb_movie_search、「ショーシャンクの空にを観たい」→ search_videos

## 動画検索（VideoSearchツール）ポリシー
- ユーザーが「観たい」「再生したい」「動画を探したい」「見たい」など視聴意図を示した場合は、必ず search_videos ツールを使って該当コンテンツを検索するアプリケーションを起動する。
- **重要**: 決して独自のサービス名やJSONレスポンスを作成せず、必ず search_videos ツールを呼び出すこと。テキストでJSONを返すことは禁止。
- **ツール呼び出し必須**: 視聴意図がある場合は、テキストで応答する前に必ず関数/ツール呼び出し機能を使用してsearch_videosを実行すること。
- サポートサービス：
    - service="youtube": 検索用の1つまたは空白区切りの複数キーワードを含む文字列を input に渡す。自由なクエリでOK。
    - service="videocenter": 映画またはTV番組の厳密なタイトル1つだけを input に渡す。複数タイトルや説明語、無関係な語は含めない。
- プラットフォームが明示されていない場合は、
    - 映画・TV番組タイトルと判断できる場合 → service="videocenter"
    - 一般動画・チュートリアル・音楽・生配信・動物動画等 → service="youtube"
- 必ずユーザーの元のクエリ（またはvideocenterの場合は厳密なタイトル）を input に渡す。
- ツール呼び出し後はツールの結果をそのまま返す（余計なテキストや理由付けは不要）。
""" + """

## 動画検索時の重要な注意点
動画を検索・再生したい場合は、必ず search_videos ツールを使用してください。独自のサービス名やJSONレスポンスを作成しないでください。

### ツール使い分け例

ユーザー: スターウォーズ エピソード1の詳細を知りたい
→ tmdb_movie_search ツールを使用

ユーザー: スターウォーズ エピソード1を観たい
→ search_videos ツールを使用（service: "videocenter", input: "スターウォーズ エピソード1"）

ユーザー: 猫の動画が見たい
→ search_videos ツールを使用（service: "youtube", input: "猫 動画"）

ユーザー: YouTubeで lo-fi hip hop を観たい
→ search_videos ツールを使用（service: "youtube", input: "lo-fi hip hop"）

ユーザー: 映画『君の名は。』を再生して
→ search_videos ツールを使用（service: "videocenter", input: "君の名は。"）

ユーザー: Python 辞書内包表記の解説動画を見たい
→ search_videos ツールを使用（service: "youtube", input: "Python 辞書内包表記 解説"）

ユーザー: 半沢直樹を探して
→ search_videos ツールを使用（service: "videocenter", input: "半沢直樹"）

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
