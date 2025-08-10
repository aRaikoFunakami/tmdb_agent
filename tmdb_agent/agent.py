"""
TMDB検索エージェント統合実装

TMDB APIを使った多言語対応のコンテンツ検索エージェント。
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from typing import Dict, Any

# 相対インポートと絶対インポートの両方に対応
try:
    # パッケージとして実行される場合（相対インポート）
    from .tools import (
        TOOLS,
        tmdb_movie_search,
        tmdb_person_search,
        tmdb_tv_search,
        tmdb_multi_search,
        tmdb_movie_credits_search,
        tmdb_tv_credits_search,
        tmdb_credits_search_by_id,
        tmdb_popular_people,
        tmdb_trending_all,
        tmdb_trending_movies,
        tmdb_trending_tv,
        tmdb_trending_people,
        tmdb_get_trending_all,
        tmdb_get_trending_movies,
        tmdb_get_trending_tv,
        tmdb_get_trending_people,
        web_search_supplement,
        theme_song_search,
        get_supported_languages,
        get_available_tools,
        detect_language_and_get_tmdb_code,
        get_language_code,
        TOOLS_TEXT,
        TOOL_NAMES,
    )
except ImportError:
    # 直接実行される場合（絶対インポート）
    from tools import (
        TOOLS,
        tmdb_movie_search,
        tmdb_person_search,
        tmdb_tv_search,
        tmdb_multi_search,
        tmdb_movie_credits_search,
        tmdb_tv_credits_search,
        tmdb_credits_search_by_id,
        tmdb_popular_people,
        tmdb_trending_all,
        tmdb_trending_movies,
        tmdb_trending_tv,
        tmdb_trending_people,
        tmdb_get_trending_all,
        tmdb_get_trending_movies,
        tmdb_get_trending_tv,
        tmdb_get_trending_people,
        web_search_supplement,
        theme_song_search,
        get_supported_languages,
        get_available_tools,
        detect_language_and_get_tmdb_code,
        get_language_code,
        TOOLS_TEXT,
        TOOL_NAMES,
    )


class TMDBSearchAgent:
    """
    統合TMDB検索エージェント

    映画、TV番組、人物の検索に対応し、自動言語検出機能付き。
    LangGraph、CLI、API等の多様なコンテキストで利用可能な汎用検索エージェント。
    
    統合による改善:
    - TMDBWorkerAgentとの冗長性を排除
    - LLMの二重呼び出しを解消してコスト削減
    - 単一クラスで完結する責務明確な設計
    
    技術スタック:
    - LangChain v0.3+ (最新APIに対応)
    - ReAct エージェントパターン
    - uvパッケージマネージャーで管理
    - 型ヒント必須（Python 3.11+）
    - マルチLLM対応（OpenAI, Gemini, Claude, Ollama等）

    LangGraphでの使用:
        >>> # LangGraphワークフローでの使用
        >>> agent = TMDBSearchAgent(llm)
        >>> result = agent.agent_executor.invoke({"input": query})

    Examples:
        >>> # 基本的な使用方法
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        >>> agent = TMDBSearchAgent(llm)
        >>> result = agent.search("進撃の巨人について教えて")
        >>> print(result)

        >>> # カスタム設定
        >>> agent = TMDBSearchAgent(
        ...     llm=llm,
        ...     verbose=False
        ... )
        >>> result = agent.search("昔見た映画で車がタイムマシンになるやつ")

        >>> # 詳細な結果を取得
        >>> detailed = agent.search_detailed("スターウォーズについて教えて")
        >>> print(detailed['output'])
        >>> print(f"処理ステップ数: {len(detailed['intermediate_steps'])}")
    """

    def __init__(self, llm: BaseLanguageModel, verbose: bool = True):
        """
        TMDBSearchAgentを初期化

        Args:
            llm: 使用するLLMオブジェクト
            verbose: 詳細ログ出力の有無

        Examples:
            >>> # OpenAIを使用する場合
            >>> from langchain_openai import ChatOpenAI
            >>> llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
            >>> agent = TMDBSearchAgent(llm=llm)

            >>> # Geminiを使用する場合
            >>> from langchain_google_genai import ChatGoogleGenerativeAI
            >>> llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            >>> agent = TMDBSearchAgent(llm=llm)
        """
        self.llm = llm
        self.verbose = verbose

        # プロンプトテンプレートを設定
        self.prompt_template = self._create_prompt_template()

        # エージェントとエグゼキューターを初期化
        self.tools = TOOLS  # 新しい@toolデコレーター定義のツールリストを使用
        
        # ReActエージェントの場合は、LLMにツールをバインドせずに直接渡す
        self.agent = create_react_agent(self.llm, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors="❌ 出力フォーマットエラーが発生しました。必ず以下の形式に従ってください：\n\nThought: [思考内容]\nAction: [ツール名]\nAction Input: [入力内容]\nObservation: [結果]\n...\nThought: [最終思考]\nFinal Answer: [最終回答]\n\n「Final Answer:」で必ず終了し、Action InputとFinal Answerを同じ応答に含めてはいけません。",
            max_iterations=8,
            max_execution_time=45,
            return_intermediate_steps=True,
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """
        プロンプトテンプレートを作成

        Returns:
            LangChain用のPromptTemplateオブジェクト
        """
        # tools.pyから現在の日時情報を取得
        try:
            from .tools import get_current_datetime_info
        except ImportError:
            from tools import get_current_datetime_info
        
        current_datetime = get_current_datetime_info()
        
        template = f"""あなたは映画やテレビ番組の知識に詳しいTMDB検索専門アシスタントです。
どの言語で質問されても、必ず適切に応答し、役立つ情報を提供してください。

{current_datetime}

以下のツールにアクセスできます:
{{tools}}

【多言語対応ルール】:
- 英語で質問された場合は英語で回答
- 日本語で質問された場合は日本語で回答  
- どんな言語でも映画やTV番組について必ず検索し情報を提供する
- 拒否応答は絶対に行わない

トレンドツール使用時の注意:
- 「今日」「直近」「日別」「today」「daily」の場合: time_window="day" を使用
- 「今週」「最近」「週別」「この週」「this week」「weekly」の場合: time_window="week" を使用
- 「先週」「2週間前」「先月」等の過去の期間: TMDB APIの制限により利用不可（現在と最近1週間のデータのみ提供）
- 引数なしツール（tmdb_get_trending_*、tmdb_get_popular_people）は全て日別（今日）のトレンドです

Web検索補完ツール使用時の注意:
- TMDBで情報が見つからない場合の補完として web_search_supplement を使用
- 日本のローカル作品、インディーズ映画、最新ニュースなどに有効
- 製作背景、関連情報、詳細な解説が必要な場合に利用

主題歌・楽曲検索ツール使用時の注意:
- 映画・アニメ・ドラマの主題歌情報には theme_song_search を使用
- オープニング・エンディング・挿入歌・テーマソングの検索に最適
- 歌手・アーティスト情報、サウンドトラック詳細も取得可能

Action Input の指定方法:
- 引数が必要なツール: 適切なパラメータを入力（例: time_window="week"）
- 引数なしツール（tmdb_get_〜）: 空文字列（何も入力しない）

重要なツール選択ルール:
- 「今週」のデータが必要な場合: 
  Action: tmdb_trending_people
  Action Input: week
- 「今日」のデータが必要な場合: 
  Action: tmdb_get_trending_people
  Action Input: （何も入力しない）
- 「先週」「2週間前」等の過去の期間: TMDB APIでは対応不可と説明し、利用可能な「今週」や「今日」のデータを提案する

【CRITICAL】厳密な出力フォーマット規則:

以下のフォーマットに厳密に従ってください。例外は一切認められません：

Question: 答えるべき入力の質問
Thought: 何をすべきかを考える
Action: [{{tool_names}}]のいずれか一つを選択（ツール名のみ、引数は含めない）
Action Input: アクションへの入力（引数なしツールの場合は空文字列、引数ありツールの場合は適切なパラメータ）
Observation: アクションの結果
... (必要に応じてThought/Action/Action Input/Observationを繰り返す)
Thought: 最終的な答えがわかりました
Final Answer: 元の質問に対する最終的な答え

【重要なフォーマット規則】:
1. Action行には必ずツール名のみを記載（tmdb_trending_people等）
2. Action Input行には引数のみを記載（引数なしの場合は空文字列）
3. 必ず「Final Answer:」で応答を終了する
4. Action InputとFinal Answerを同じ応答に絶対に含めない
5. ツール実行後は必ずObservationを確認してからFinal Answerを書く
6. 途中で応答を停止しない

【絶対禁止事項】:
- Action InputとFinal Answerを同じ応答に含めること
- Final Answer: なしで応答を終了すること
- 途中で応答を止めること
- "I'm sorry, I can't assist with that request." のような拒否的応答
- ツール実行結果を受け取った後、即座に文章で返答すること（必ずFinal Answer: を使用）

【必須要件】:
- Actionにはツール名のみを記入（tmdb_trending_people等）
- パラメータはAction Inputに記入
- 必ず「Final Answer:」で終わること
- 質問の言語に合わせて自然な文章で回答すること（英語で質問されたら英語で回答）
- どんな制約があっても、利用可能な情報で最良の回答を提供すること

Question: {{input}}
Thought:{{agent_scratchpad}}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": TOOLS_TEXT,
                "tool_names": TOOL_NAMES,
            },
        )

    def search(self, query: str) -> str:
        """
        コンテンツ検索を実行

        Args:
            query: 検索クエリ（自然言語）

        Returns:
            検索結果のテキスト

        Examples:
            >>> agent = TMDBSearchAgent()
            >>> result = agent.search("進撃の巨人について教えて")
            >>> print(result)
        """
        try:
            # 新しい実装では各ツールが独自に言語検出を行うため、グローバル設定は不要
            result = self.agent_executor.invoke({"input": query})
            return result.get("output", "検索結果を取得できませんでした。")
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"

    def search_detailed(self, query: str) -> Dict[str, Any]:
        """
        詳細な検索結果を取得（内部処理も含む）

        Args:
            query: 検索クエリ（自然言語）

        Returns:
            検索結果の詳細辞書
            - output: 最終回答
            - intermediate_steps: 中間処理ステップ
            - input: 入力クエリ

        Examples:
            >>> agent = TMDBSearchAgent()
            >>> result = agent.search_detailed("昔見た映画で車がタイムマシンになるやつ")
            >>> print(result['output'])
            >>> print(result['intermediate_steps'])
        """
        try:
            # 新しい実装では各ツールが独自に言語検出を行うため、グローバル設定は不要
            return self.agent_executor.invoke({"input": query})
        except Exception as e:
            return {
                "input": query,
                "output": f"エラーが発生しました: {str(e)}",
                "intermediate_steps": [],
            }

    def set_verbose(self, verbose: bool) -> None:
        """
        詳細ログ出力の設定を変更

        Args:
            verbose: 詳細ログを出力するかどうか
        """
        self.verbose = verbose
        self.agent_executor.verbose = verbose

    def get_supported_languages(self) -> Dict[str, str]:
        """
        サポートされている言語のリストを取得

        Returns:
            言語コードと言語名の辞書
        """
        return get_supported_languages()

    def get_available_tools(self) -> Dict[str, str]:
        """
        利用可能なツールのリストを取得

        Returns:
            ツール名と説明の辞書
        """
        return get_available_tools()


def create_tmdb_agent(llm: BaseLanguageModel, verbose: bool = True) -> TMDBSearchAgent:
    """
    TMDBSearchAgentのファクトリー関数

    Args:
        llm: 使用するLLMオブジェクト
        verbose: 詳細ログ出力の有無

    Returns:
        TMDBSearchAgentインスタンス

    Examples:
        >>> # OpenAIを使用
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        >>> agent = create_tmdb_agent(llm)

        >>> # Geminiを使用
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        >>> agent = create_tmdb_agent(llm)
    """
    return TMDBSearchAgent(llm=llm, verbose=verbose)
