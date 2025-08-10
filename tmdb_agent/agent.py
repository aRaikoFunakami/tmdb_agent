"""
TMDB検索エージェント統合実装

TMDB APIを使った多言語対応のコンテンツ検索エージェント。
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
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
        LangChain HubからReActプロンプトテンプレートを取得し、TMDB用にカスタマイズ

        Returns:
            LangChain用のPromptTemplateオブジェクト
        """
        # LangChain HubからReActプロンプトを取得
        base_prompt = hub.pull("hwchase17/react")
        
        # tools.pyから現在の日時情報を取得
        try:
            from .tools import get_current_datetime_info
        except ImportError:
            from tools import get_current_datetime_info
        
        current_datetime = get_current_datetime_info()
        
        # ReActプロンプトにTMDB固有の情報を追加
        tmdb_instructions = f"""You are a TMDB search specialist assistant with extensive knowledge of movies and TV shows.
You must respond appropriately in any language and provide helpful information.

{current_datetime}

MULTILINGUAL RESPONSE RULES:
- If asked in English, respond in English
- If asked in Japanese, respond in Japanese
- Always search for and provide information about movies or TV shows regardless of language
- Never refuse to respond

TRENDING TOOLS USAGE NOTES:
- For "today", "current", "daily": use time_window="day"
- For "this week", "recent", "weekly": use time_window="week"
- Past periods like "last week", "2 weeks ago" are not available due to TMDB API limitations (only current and recent week data provided)
- Tools without arguments (tmdb_get_trending_*, tmdb_get_popular_people) all provide daily (today's) trends

WEB SEARCH SUPPLEMENT TOOL USAGE:
- Use web_search_supplement when TMDB doesn't have information
- Effective for Japanese local works, indie films, latest news
- Use for production background, related information, detailed explanations

THEME SONG SEARCH TOOL USAGE:
- Use theme_song_search for movie/anime/drama theme songs
- Optimal for opening/ending/insert songs/theme songs
- Can retrieve artist information and soundtrack details

ACTION INPUT GUIDELINES:
- Tools requiring arguments: provide appropriate parameters (e.g., time_window="week")
- Tools without arguments (tmdb_get_*): use empty string (no input)

CRITICAL TOOL SELECTION RULES:
- For "this week" data:
  Action: tmdb_trending_people
  Action Input: week
- For "today" data:
  Action: tmdb_get_trending_people
  Action Input: (no input)
- For past periods like "last week", "2 weeks ago": explain TMDB API limitations and suggest available "this week" or "today" data

STRICT OUTPUT FORMAT RULES:
1. Action line must contain only the tool name (e.g., tmdb_trending_people)
2. Action Input line must contain only arguments (empty string for no-argument tools)
3. Must end with "Final Answer:"
4. Never include Action Input and Final Answer in the same response
5. Always check Observation after tool execution before writing Final Answer
6. Never stop response midway

ABSOLUTELY PROHIBITED:
- Including Action Input and Final Answer in the same response
- Ending response without "Final Answer:"
- Stopping response midway
- Refusal responses like "I'm sorry, I can't assist with that request."
- Immediately responding with text after receiving tool results (must use Final Answer:)

REQUIRED:
- Action contains only tool name (e.g., tmdb_trending_people)
- Parameters go in Action Input
- Must end with "Final Answer:"
- Respond naturally in the question's language
- Always provide the best possible answer with available information"""

        # 元のReActプロンプトテンプレートを取得し、TMDB用の指示を追加
        modified_template = base_prompt.template.replace(
            "You have access to the following tools:",
            f"{tmdb_instructions}\n\nYou have access to the following tools:"
        )
        
        return PromptTemplate(
            template=modified_template,
            input_variables=base_prompt.input_variables,
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
