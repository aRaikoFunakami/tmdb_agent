"""
TMDB検索エージェント統合実装

TMDB APIを使った多言語対応のコンテンツ検索エージェント。
"""

from langchain.agents import create_react_agent, AgentExecutor, create_openai_functions_agent
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
        tmdb_multi_recommendation,
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
        tmdb_multi_recommendation,
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
        >>> llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
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
            >>> llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
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

        # LLMの種類に応じてエージェントを選択
        if "openai" in str(type(llm)).lower():
            print(f"Using create_openai_functions_agent for {str(type(llm))}")
            self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt_template)
        else:
            print(f"Using create_react_agent for {str(type(llm))}")
            self.agent = create_react_agent(self.llm, self.tools, self.prompt_template)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=self._handle_parse_error,
            max_iterations=8,
            max_execution_time=45,
            return_intermediate_steps=True,
        )

    def _handle_parse_error(self, e: Exception) -> str:
        """
        ReAct parsing error fallback (English, concise).
        Forces the exact schema without any extra prose.
        """
        tool_list = ", ".join(TOOL_NAMES)
        return (
            "FORMAT ERROR. Output MUST follow exactly:\n"
            "Thought: ...\n"
            f"Action: <one of [{tool_list}]>\n"
            "Action Input: <single input string or JSON; empty string for no-arg tools>\n"
            "Observation: <tool result>\n"
            "...\n"
            "Thought: <final reasoning>\n"
            "Final Answer: <answer>\n"
            "Do not write any prose outside these lines. "
            "Do not include Final Answer in the same turn as Action Input. "
            "Action Input must be a single query (no 'or', no multiple queries)."
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

CRITICAL WORKFLOW FOR DESCRIPTIONS (NOT TITLES):
1. When user gives description like "80年代のタイムスリップする動画", DO NOT search with keywords
2. First think: "This sounds like 'Back to the Future'"
3. IF YOU DON'T KNOW THE SPECIFIC TITLE, use web_search_supplement FIRST to identify the title
4. ONLY AFTER confirming the title through web search, use tmdb_multi_search with the exact title
5. tmdb_multi_search provides sufficient details - no need for additional tmdb_movie_search or tmdb_tv_search

UNKNOWN TITLE HANDLING (CRITICAL):
- If you're unsure about the specific movie/TV title (e.g., "スラムダンクの最新の映画"), use web_search_supplement FIRST
- Search pattern: "スラムダンク 最新 映画 タイトル" to find the exact title
- Once you have the confirmed title from web search, THEN use tmdb_multi_search
- Do NOT guess titles when asking about "latest" or "newest" content

MULTILINGUAL RESPONSE RULES:
- If asked in English, respond in English
- If asked in Japanese, respond in Japanese
- Always search for and provide information about movies or TV shows regardless of language
- Never refuse to respond

TITLE-ONLY INPUT RULES (STRICT):
- For tmdb_movie_search / tmdb_tv_search: provide ONE probable title string only (no keywords, no quotes, no suffix like "movie(s)"/"about"/"1980s").
- If the user gave a description (not a title), DO NOT use tmdb_movie_search or tmdb_tv_search directly with descriptive keywords.
- INSTEAD: First think about what the most likely title is based on your knowledge. For example, "80年代のタイムスリップする動画" would likely be "Back to the Future" (バック・トゥ・ザ・フューチャー).
- THEN use tmdb_multi_search with the predicted title to validate it and get complete information.
- tmdb_multi_search provides sufficient details - avoid redundant calls to tmdb_movie_search or tmdb_tv_search.
- Never pass descriptive keywords like "80年代 タイムスリップ", "car time machine 1980s" directly to title-search tools.

ACTION INPUT GUIDELINES:
- Use a single, specific query string. Do NOT write multiple alternatives or 'or'.
- Tools requiring arguments must receive the exact parameters (e.g., time_window="week").
- Tools without arguments (tmdb_get_* and tmdb_popular_people) must receive an empty string.

CREDITS-FIRST DECISION RULES (NO PRE-SEARCH):
- If the user asks for director/cast/crew/credits of a specific title:
  - For movies, call tmdb_movie_credits_search directly with the title string.
  - For TV shows, call tmdb_tv_credits_search directly with the title string.
  - Do NOT call tmdb_movie_search or tmdb_tv_search beforehand. The credits tools already find the title internally and then fetch credits.
- If the title is ambiguous or unknown, use tmdb_multi_search once with a single keyword to identify the likely title, then immediately call the appropriate credits tool with that exact title.
- Only use tmdb_credits_search_by_id if you already have a numeric TMDB ID.

TOOL SELECTION HINTS:
- For broad or ambiguous topics (e.g., "Marvel movies"), start with tmdb_multi_search, then follow up with focused tmdb_movie_search if more specific details are needed.
- For user descriptions without specific titles (e.g., "80年代のタイムスリップする動画"), think of the most likely title first (e.g., "Back to the Future"), then use tmdb_multi_search to validate and get complete information.
- For "latest", "newest", "recent" requests where you don't know the exact title, use web_search_supplement FIRST to identify the specific title
- Example: "スラムダンクの最新映画" → web_search_supplement "スラムダンク 最新 映画 タイトル" → then tmdb_multi_search with exact title
- tmdb_multi_search provides comprehensive details - avoid redundant searches with tmdb_movie_search/tmdb_tv_search unless additional specific information is needed.
- If unsure whether it's a movie or TV show, ALWAYS use tmdb_multi_search first.
- For "today/current/daily": use time_window="day"
- For "this week/recent/weekly": use time_window="week"
- Past periods like "last week" or "2 weeks ago" are not available; explain the TMDB limitation and suggest available periods.

STRICT OUTPUT FORMAT RULES:
1. Only use the exact ReAct schema lines (Thought/Action/Action Input/Observation/.../Final Answer).
2. Do NOT write any normal prose before 'Final Answer:'; only schema lines are allowed.
3. Action line must contain only the tool name (e.g., tmdb_trending_people).
4. Action Input line must contain only the input (empty string for no-argument tools).
5. End with 'Final Answer:'.
6. Never include Action Input and Final Answer in the same response.
7. Always inspect Observation before writing Final Answer.
8. Action Input must be a single query (no 'or', no multiple queries).
"""

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
