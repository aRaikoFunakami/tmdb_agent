from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import requests
import os
from datetime import datetime
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sudachipy import tokenizer, dictionary

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# 形態素解析して SearcH API に適した形式に変換するための関数
TOKENIZER = dictionary.Dictionary().create()
MODE = tokenizer.Tokenizer.SplitMode.B

def tokenize_text(text):
    return [m.surface() for m in TOKENIZER.tokenize(text, MODE)]

# Pydantic モデル定義（厳格な型チェックとJSONスキーマ生成）
class MovieSearchInput(BaseModel):
    """映画検索の入力パラメータ"""
    query: str = Field(
        description=(
            "検索する映画タイトルまたは最小限の関連キーワードのみを指定する。"
            "説明文・引用符・装飾・改行は不可。"
            "関係ない語句（例: 80年代, 車, 教えて等）は含めない。"
            "複数語は半角スペース区切り、中黒や句読点は使わない。"
        ),
        min_length=1,
        max_length=64,
        examples=[
            "バック トゥ ザ フューチャー",
            "Back to the Future",
            "ターミネーター 2",
        ],
    )
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合は自動検出。明示的に言語を指定したい場合に使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class TVSearchInput(BaseModel):
    """TV番組検索の入力パラメータ"""
    query: str = Field(
        description=(
            "検索するTV番組のタイトルまたは最小限の関連キーワードのみを指定する。"
            "説明文・引用符・装飾・改行は不可。"
            "関係ない語句（例: 80年代, 車, 教えて等）は含めない。"
            "複数語は半角スペース区切り、中黒や句読点は使わない。"
        ),
        min_length=1,
        max_length=64,
        examples=[
            "バック トゥ ザ フューチャー",
            "Back to the Future",
            "ターミネーター 2",
        ],
    )
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合は自動検出。明示的に言語を指定したい場合に使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class PersonSearchInput(BaseModel):
    """人物検索の入力パラメータ"""
    query: str = Field(description="検索する人物の名前", min_length=1)
    query: str = Field(
        description=(
            "検索する人物の名前のみを指定する。"
            "説明文・引用符・装飾・改行は不可。"
            "関係ない語句（例: 80年代, 映画, 教えて等）は含めない。"
            "複数語は半角スペース区切り、中黒や句読点は使わない。"
            "※一部の人物名ではミドルネームを含めると検索に失敗する場合があるため、"
            "可能な限り短く、代表的な表記のみを使用する（例: 「マイケル フォックス」は可、"
            "「マイケル ジェイ フォックス」は不可）。"
        ),
        min_length=1,
        max_length=64,
        examples=[
            "マイケル フォックス",
            "Tom Hanks",
            "山田 太郎",
        ],
    )
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合は自動検出。明示的に言語を指定したい場合に使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class MultiSearchInput(BaseModel):
    """マルチ検索の入力パラメータ"""
    query: str = Field(
        description=(
            "検索するキーワード（映画・TV番組・人物を横断検索）のうち1つを指定する。"
            "説明文・引用符・装飾・改行は不可。"
            "関係ない語句（例: 80年代, 映画, 教えて等）は含めない。"
            "複数語は半角スペース区切り、中黒や句読点は使わない。"
            "※一部の人物名ではミドルネームを含めると検索に失敗する場合があるため、"
            "可能な限り短く、代表的な表記のみを使用する（例: 「マイケル フォックス」は可、"
            "「マイケル ジェイ フォックス」は不可）。"
        ),
        min_length=1,
        max_length=64,
        examples=[
            "バック トゥ ザ フューチャー",
            "Back to the Future",
            "ターミネーター 2",
            "マイケル フォックス",
            "Tom Hanks",
            "山田 太郎",
        ],
    )
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合は自動検出。明示的に言語を指定したい場合に使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class CreditsSearchInput(BaseModel):
    """クレジット検索の入力パラメータ（タイトル検索）"""
    query: str = Field(description="検索する作品のタイトル", min_length=1)
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合は自動検出。明示的に言語を指定したい場合に使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class CreditsSearchByIdInput(BaseModel):
    """クレジット検索の入力パラメータ（ID検索）"""
    movie_id: Optional[int] = Field(default=None, description="TMDB映画ID")
    tv_id: Optional[int] = Field(default=None, description="TMDB TV番組ID")
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（例: ja-JP, en-US）。指定しない場合はen-USを使用。",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )
    
    def model_post_init(self, __context):
        """movie_idとtv_idのうち、どちらか一つが必須"""
        if not self.movie_id and not self.tv_id:
            raise ValueError('movie_idまたはtv_idのいずれかを指定してください')
        if self.movie_id and self.tv_id:
            raise ValueError('movie_idとtv_idの両方を同時に指定することはできません')
        if self.movie_id and self.movie_id <= 0:
            raise ValueError('movie_idは正の整数である必要があります')
        if self.tv_id and self.tv_id <= 0:
            raise ValueError('tv_idは正の整数である必要があります')

class WebSearchInput(BaseModel):
    """Web検索の入力パラメータ"""
    query: str = Field(description="Web検索するキーワード（映画・TV番組・人物の補完情報など）", min_length=1)

class ThemeSongSearchInput(BaseModel):
    """主題歌・楽曲検索の入力パラメータ"""
    query: str = Field(description="主題歌・楽曲を検索するキーワード（映画・アニメ・ドラマのタイトルや歌手名など）", min_length=1)

class PopularPeopleInput(BaseModel):
    """人気順人物リスト取得の入力パラメータ"""
    page: int = Field(default=1, description="取得するページ番号（1-500、デフォルト: 1）", ge=1, le=500)
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（ja-JP, en-US等）",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )

class TrendingInput(BaseModel):
    """トレンド検索の入力パラメータ"""
    time_window: str = Field(
        default="day", 
        description="時間枠（day: 日別・今日・直近、week: 週別・今週・最近1週間）。ユーザーが「今日」「直近」と言った場合は'day'、「今週」「最近」と言った場合は'week'を使用。TMDB APIの制限により、過去の特定期間（先週、2週間前など）は利用不可。", 
        pattern="^(day|week)$"
    )
    language_code: Optional[str] = Field(
        default=None, 
        description="検索言語コード（ja-JP, en-US等）",
        pattern="^[a-z]{2}-[A-Z]{2}$"
    )
    
    def __init__(self, **data):
        # 空文字列をデフォルト値に変換
        if "time_window" in data and (data["time_window"] == "" or data["time_window"] is None):
            data["time_window"] = "day"
        if "language_code" in data and data["language_code"] == "":
            data["language_code"] = None
        super().__init__(**data)


# 言語コードマッピング（ISO 639-1 + ISO 3166-1）- 共通定義
SUPPORTED_LANGUAGES = {
    "ja": "ja-JP",  # 日本語
    "en": "en-US",  # 英語
    "ko": "ja-JP",  # 韓国語 → 日本語に統一
    "zh": "ja-JP",  # 中国語（簡体字） → 日本語に統一
    "de": "de-DE",  # ドイツ語
}

# ツール情報の統一定義
TOOL_DESCRIPTIONS = {
    "tmdb_movie_search": "映画の具体的なタイトルで検索",
    "tmdb_tv_search": "TV番組の具体的なタイトルで検索", 
    "tmdb_person_search": "具体的な人名で検索",
    "tmdb_multi_search": "映画・TV・人物を横断検索",
    "tmdb_movie_credits_search": "映画の詳細なクレジット情報を取得（タイトル検索）",
    "tmdb_tv_credits_search": "TV番組の詳細なクレジット情報を取得（タイトル検索）",
    "tmdb_credits_search_by_id": "映画IDまたはTV番組IDを直接指定してクレジット情報を取得",
    "tmdb_popular_people": "人気順で人物リストを取得（ページ指定可能）",
    "tmdb_get_popular_people": "人気順で人物リストを取得（引数なし・シンプル版）",
    "tmdb_trending_all": "全コンテンツのトレンド（映画・TV・人物）を取得。time_window: 'day'=日別（今日・直近）、'week'=週別（今週・最近1週間）",
    "tmdb_trending_movies": "映画のトレンドを取得。time_window: 'day'=日別（今日・直近）、'week'=週別（今週・最近1週間）",
    "tmdb_trending_tv": "TV番組のトレンドを取得。time_window: 'day'=日別（今日・直近）、'week'=週別（今週・最近1週間）", 
    "tmdb_trending_people": "人物のトレンドを取得。time_window: 'day'=日別（今日・直近）、'week'=週別（今週・最近1週間）",
    "tmdb_get_trending_all": "全コンテンツの日別トレンドを取得（引数なし・シンプル版）- 今日・直近のトレンド",
    "tmdb_get_trending_movies": "映画の日別トレンドを取得（引数なし・シンプル版）- 今日・直近のトレンド",
    "tmdb_get_trending_tv": "TV番組の日別トレンドを取得（引数なし・シンプル版）- 今日・直近のトレンド",
    "tmdb_get_trending_people": "人物の日別トレンドを取得（引数なし・シンプル版）- 今日・直近のトレンド",
    "web_search_supplement": "TMDBで見つからない映画・TV・人物情報をWebから検索して補完",
    "theme_song_search": "映画・アニメ・ドラマの主題歌・エンディング・挿入歌や歌手情報をWebから検索",
}# プロンプト用のツール説明文
TOOL_NAMES = "tmdb_movie_search, tmdb_tv_search, tmdb_person_search, tmdb_multi_search, tmdb_movie_credits_search, tmdb_tv_credits_search, current_trends_search"


def get_supported_languages() -> dict:
    """
    サポートされている言語のリストを取得
    
    Returns:
        言語コードと言語名の辞書
    """
    language_names = {
        "ja-JP": "日本語",
        "en-US": "英語", 
        "de-DE": "ドイツ語",
    }
    return language_names


def get_available_tools() -> dict:
    """
    利用可能なツールのリストを取得
    
    Returns:
        ツール名と説明の辞書
    """
    return TOOL_DESCRIPTIONS.copy()


def detect_language_and_get_tmdb_code(query: str) -> str:
    """
    クエリの言語を検出してTMDB APIに適した言語コードを返す
    日本語、韓国語、中国語、ドイツ語、英語のみサポート
    日本語、韓国語、中国語は全て日本語として扱う
    TMDB_API_LANG環境変数が設定されている場合は強制的に優先される
    
    Args:
        query: 検索クエリ
        
    Returns:
        TMDB API用の言語コード
    """
    # TMDB_API_LANG環境変数をチェック（最優先）
    tmdb_api_lang = os.getenv("TMDB_API_LANG")
    if tmdb_api_lang:
        return tmdb_api_lang
    
    try:
        detected_lang = detect(query)
        # サポートされている言語のみ対応、それ以外は英語
        return SUPPORTED_LANGUAGES.get(detected_lang, "en-US")
    except LangDetectException:
        # 言語検出に失敗した場合は英語をデフォルトとする
        return "en-US"


def get_current_datetime_info() -> str:
    """
    現在の日時情報を取得してトレンドツール用のコンテキストを提供
    
    Returns:
        現在の日時情報を含む文字列
    """
    now = datetime.now()
    return f"現在の日時: {now.strftime('%Y年%m月%d日 %H:%M')} ({now.strftime('%A')})"


def get_language_code(query: str, provided_code: Optional[str] = None) -> str:
    """
    言語コードを決定する（グローバル状態に依存しない）
    
    Args:
        query: 検索クエリ
        provided_code: 明示的に指定された言語コード
        
    Returns:
        使用する言語コード
    """
    if provided_code:
        return provided_code
    return detect_language_and_get_tmdb_code(query)


# @tool デコレーターを使った新しいツール定義
@tool("tmdb_movie_search", args_schema=MovieSearchInput)
def tmdb_movie_search(query: str, language_code: Optional[str] = None) -> str:
    """TMDBで映画を検索します。具体的な映画タイトルやキーワードを使用してください。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)

    # 形態素解析して検索クエリを整形
    query = " ".join(tokenize_text(query))

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:5]
        
        if not results:
            return f"「{query}」に一致する映画が見つかりませんでした。より具体的なタイトルやキーワードを試してください。（検索言語: {lang_code}）"

        output = []
        for r in results:
            overview = r.get("overview", "あらすじ情報なし")
            if len(overview) > 100:
                overview = overview[:100] + "..."

            output.append(
                f"title: {r['title']}\n"
                f"original_title: {r.get('original_title', 'N/A')}\n"
                f"release_date: {r.get('release_date', 'N/A')}\n"
                f"vote_average: {r['vote_average']}\n"
                f"overview: {overview}\n"
            )

        # 検索に使用した言語コードを結果に含める
        output.append(f"language: {lang_code}")
        return "\n".join(output) + "\n"
        
    except Exception as e:
        return f"映画検索でエラーが発生しました: {str(e)}"


@tool("tmdb_person_search", args_schema=PersonSearchInput)
def tmdb_person_search(query: str, language_code: Optional[str] = None) -> str:
    """TMDBで人物（俳優、監督など）を検索します。具体的な人名を使用してください。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)

    # 形態素解析して検索クエリを整形
    query = " ".join(tokenize_text(query))
    
    url = "https://api.themoviedb.org/3/search/person"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:3]
        
        if not results:
            return f"「{query}」に一致する人物が見つかりませんでした。（検索言語: {lang_code}）"

        output = []
        for r in results:
            known_for_titles = [
                movie.get("title", movie.get("name", ""))
                for movie in r.get("known_for", [])[:3]
            ]
            known_for_str = (
                ", ".join(known_for_titles) if known_for_titles else "代表作情報なし"
            )

            output.append(
                f"person_name: {r['name']}\n"
                f"known_for_department: {r.get('known_for_department', 'N/A')}\n"
                f"known_for: {known_for_str}\n"
            )

        # 検索に使用した言語コードを結果に含める
        output.append(f"language: {lang_code}")
        return "\n".join(output)
        
    except Exception as e:
        return f"人物検索でエラーが発生しました: {str(e)}"


@tool("tmdb_tv_search", args_schema=TVSearchInput)
def tmdb_tv_search(query: str, language_code: Optional[str] = None) -> str:
    """TMDBでTV番組・ドラマ・アニメを検索します。具体的な番組タイトルを使用してください。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)

    # 形態素解析して検索クエリを整形
    query = " ".join(tokenize_text(query))
    
    url = "https://api.themoviedb.org/3/search/tv"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:5]
        
        if not results:
            return f"「{query}」に一致するTV番組が見つかりませんでした。より具体的なタイトルやキーワードを試してください。（検索言語: {lang_code}）"

        output = []
        for r in results:
            overview = r.get("overview", "あらすじ情報なし")
            if len(overview) > 100:
                overview = overview[:100] + "..."

            # TV番組の場合はfirst_air_dateを使用
            air_date = r.get("first_air_date", "N/A")

            output.append(
                f"name: {r['name']}\n"
                f"original_name: {r.get('original_name', 'N/A')}\n"
                f"air_date: {air_date}\n"
                f"vote_average: {r['vote_average']}\n"
                f"overview: {overview}\n"
            )

        # 検索に使用した言語コードを結果に含める
        output.append(f"language: {lang_code}")
        return "\n".join(output)
        
    except Exception as e:
        return f"TV番組検索でエラーが発生しました: {str(e)}"


@tool("tmdb_multi_search", args_schema=MultiSearchInput)
def tmdb_multi_search(query: str, language_code: Optional[str] = None) -> str:
    """TMDBで映画・TV番組・人物を横断検索します。コンテンツの種類が不明な場合に使用してください。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)

    # 形態素解析して検索クエリを整形
    query = " ".join(tokenize_text(query))
    
    url = "https://api.themoviedb.org/3/search/multi"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:6]
        
        if not results:
            return f"「{query}」に一致するコンテンツが見つかりませんでした。より具体的なタイトルやキーワードを試してください。（検索言語: {lang_code}）"

        output = []
        for r in results:
            media_type = r.get("media_type", "unknown")

            if media_type == "movie":
                output.append(
                    f"movie_title: {r['title']}\n"
                    f"release_date: {r.get('release_date', 'N/A')}\n"
                    f"vote_average: {r['vote_average']}\n"
                    f"overview: {r.get('overview', 'N/A')[:100]}...\n"
                )
            elif media_type == "tv":
                output.append(
                    f"tv_name: {r['name']}\n"
                    f"first_air_date: {r.get('first_air_date', 'N/A')}\n"
                    f"vote_average: {r['vote_average']}\n"
                    f"overview: {r.get('overview', 'N/A')[:100]}...\n"
                )
            elif media_type == "person":
                known_for_titles = [
                    item.get("title", item.get("name", ""))
                    for item in r.get("known_for", [])[:2]
                ]
                known_for_str = (
                    ", ".join(known_for_titles) if known_for_titles else "代表作情報なし"
                )
                output.append(
                    f"person_name: {r['name']}\n"
                    f"known_for_department: {r.get('known_for_department', 'N/A')}\n"
                    f"known_for: {known_for_str}\n"
                )

        # 検索に使用した言語コードを結果に含める
        output.append(f"language: {lang_code}")
        return "\n".join(output)
        
    except Exception as e:
        return f"マルチ検索でエラーが発生しました: {str(e)}"

def get_tmdb_movie_credits(movie_id: str, language_code: str = None) -> str:
    """映画IDに基づいて詳細なクレジット情報（キャストとクルー）を取得します。
    
    Args:
        movie_id: TMDB映画ID
        language_code: 言語コード（デフォルト: en-US）
        
    Returns:
        フォーマットされたクレジット情報の文字列
    """
    if language_code is None:
        language_code = "en-US"
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    params = {"api_key": TMDB_API_KEY, "language": language_code}
    
    try:
        res = requests.get(url, params=params).json()
        
        if "cast" not in res and "crew" not in res:
            return f"映画ID {movie_id} のクレジット情報が見つかりませんでした。"
        
        output = []
        output.append(f"movie_id: {movie_id}\n")

        # 監督とプロデューサーを取得
        crew = res.get("crew", [])
        directors = [person for person in crew if person.get("job") == "Director"]
        producers = [person for person in crew if person.get("job") == "Producer"]
        writers = [person for person in crew if person.get("job") in ["Writer", "Screenplay"]]
        
        if directors:
            director_names = [d["name"] for d in directors[:3]]
            output.append(f"director: {', '.join(director_names)}")
        
        if producers:
            producer_names = [p["name"] for p in producers[:3]]
            output.append(f"producer: {', '.join(producer_names)}")
            
        if writers:
            writer_names = [w["name"] for w in writers[:3]]
            output.append(f"writer: {', '.join(writer_names)}")

        # 主要キャストを取得（上位10名）
        cast = res.get("cast", [])[:10]
        if cast:
            output.append("\ncast:")
            for actor in cast:
                character = actor.get("character", "character not specified")
                output.append(f"  • {actor['name']} as {character}")
        
        # 検索に使用した言語コードを結果に含める
        output.append(f"\nlanguage: {language_code}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"映画クレジット取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"映画クレジット取得でエラーが発生しました: {str(e)}"


def get_tmdb_tv_credits(tv_id: str, language_code: str = None) -> str:
    """TV番組IDに基づいて詳細なクレジット情報（キャストとクルー）を取得します。
    
    Args:
        tv_id: TMDB TV番組ID
        language_code: 言語コード（デフォルト: en-US）
        
    Returns:
        フォーマットされたクレジット情報の文字列
    """
    if language_code is None:
        language_code = "en-US"
    
    url = f"https://api.themoviedb.org/3/tv/{tv_id}/credits"
    params = {"api_key": TMDB_API_KEY, "language": language_code}
    
    try:
        res = requests.get(url, params=params).json()
        
        if "cast" not in res and "crew" not in res:
            return f"TV番組ID {tv_id} のクレジット情報が見つかりませんでした。"
        
        output = []
        output.append(f"tv_id: {tv_id}\n")

        # クリエイターとプロデューサーを取得
        crew = res.get("crew", [])
        creators = [person for person in crew if person.get("job") in ["Creator", "Executive Producer"]]
        directors = [person for person in crew if person.get("job") == "Director"]
        writers = [person for person in crew if person.get("job") in ["Writer", "Screenplay"]]
        
        if creators:
            creator_names = [c["name"] for c in creators[:3]]
            output.append(f"creator: {', '.join(creator_names)}")
        
        if directors:
            director_names = [d["name"] for d in directors[:3]]
            output.append(f"director: {', '.join(director_names)}")
            
        if writers:
            writer_names = [w["name"] for w in writers[:3]]
            output.append(f"writer: {', '.join(writer_names)}")

        # 主要キャストを取得（上位10名）
        cast = res.get("cast", [])[:10]
        if cast:
            output.append("\ncast:")
            for actor in cast:
                character = actor.get("character", "character not specified")
                output.append(f"  • {actor['name']} as {character}")
        
        # 検索に使用した言語コードを結果に含める
        output.append(f"\nlanguage: {language_code}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"TV番組クレジット取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"TV番組クレジット取得でエラーが発生しました: {str(e)}"


@tool("tmdb_credits_search_by_id", args_schema=CreditsSearchByIdInput)
def tmdb_credits_search_by_id(movie_id: Optional[int] = None, tv_id: Optional[int] = None, language_code: Optional[str] = None) -> str:
    """映画IDまたはTV番組IDを直接指定して、詳細なクレジット情報を取得します。"""
    
    # 入力検証
    if not movie_id and not tv_id:
        return "エラー: movie_idまたはtv_idのいずれかを指定してください。"
    if movie_id and tv_id:
        return "エラー: movie_idとtv_idの両方を同時に指定することはできません。"
    
    # 言語コードの設定（デフォルト: en-US）
    lang_code = language_code or "en-US"
    
    try:
        if movie_id:
            # 映画のクレジット情報を取得
            credits_info = get_tmdb_movie_credits(str(movie_id), lang_code)
            return credits_info
        elif tv_id:
            # TV番組のクレジット情報を取得
            credits_info = get_tmdb_tv_credits(str(tv_id), lang_code)
            return credits_info
            
    except Exception as e:
        return f"ID指定クレジット検索でエラーが発生しました: {str(e)}"


@tool("tmdb_movie_credits_search", args_schema=CreditsSearchInput)
def tmdb_movie_credits_search(query: str, language_code: Optional[str] = None) -> str:
    """映画の監督、キャスト、スタッフなどの詳細なクレジット情報を取得します。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)
    
    # まず映画を検索
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])
        
        if not results:
            return f"「{query}」に一致する映画が見つかりませんでした。より具体的なタイトルを試してください。（検索言語: {lang_code}）"
        
        # 最も関連性の高い結果（最初の結果）を使用
        movie = results[0]
        movie_id = movie["id"]
        
        output = []
        output.append(f"title: {movie['title']} ({movie.get('release_date', 'N/A')})")
        output.append(f"original_title: {movie.get('original_title', 'N/A')}")
        output.append(f"overview: {movie.get('overview', 'N/A')[:100]}...")
        output.append(f"release_date: {movie.get('release_date', 'N/A')}")
        output.append(f"vote_average: {movie['vote_average']}/10")
        output.append("")
        
        # クレジット情報を取得
        credits_info = get_tmdb_movie_credits(str(movie_id), lang_code)
        output.append(credits_info)
        
        return "\n".join(output)
        
    except Exception as e:
        return f"映画検索・クレジット取得でエラーが発生しました: {str(e)}"


@tool("tmdb_tv_credits_search", args_schema=CreditsSearchInput)
def tmdb_tv_credits_search(query: str, language_code: Optional[str] = None) -> str:
    """TV番組・ドラマ・アニメのクリエイター、キャスト、スタッフなどの詳細なクレジット情報を取得します。"""
    # 言語コードを決定
    lang_code = get_language_code(query, language_code)
    
    # まずTV番組を検索
    url = "https://api.themoviedb.org/3/search/tv"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])
        
        if not results:
            return f"「{query}」に一致するTV番組が見つかりませんでした。より具体的なタイトルを試してください。（検索言語: {lang_code}）"
        
        # 最も関連性の高い結果（最初の結果）を使用
        tv_show = results[0]
        tv_id = tv_show["id"]
        
        output = []
        output.append(f"name: {tv_show['name']}")
        output.append(f"original_name: {tv_show.get('original_name', 'N/A')}")
        output.append(f"overview: {tv_show.get('overview', 'N/A')[:100]}...")
        output.append(f"first_air_date: {tv_show.get('first_air_date', 'N/A')}")
        output.append(f"vote_average: {tv_show['vote_average']}/10")
        output.append("")
        
        # クレジット情報を取得
        credits_info = get_tmdb_tv_credits(str(tv_id), lang_code)
        output.append(credits_info)
        
        return "\n".join(output)
        
    except Exception as e:
        return f"TV番組検索・クレジット取得でエラーが発生しました: {str(e)}"


@tool("tmdb_popular_people", args_schema=PopularPeopleInput)
def tmdb_popular_people(page: int = 1, language_code: Optional[str] = None) -> str:
    """TMDBで人気順の人物リスト（俳優・監督・その他業界人）を取得します。デフォルトでページ1、上位15人を表示。"""
    
    # 言語コードの決定（優先順位: 1.明示的指定 2.環境変数 3.デフォルト ja-JP）
    if language_code:
        lang_code = language_code
    else:
        # TMDB_API_LANG環境変数をチェック
        tmdb_api_lang = os.getenv("TMDB_API_LANG")
        lang_code = tmdb_api_lang if tmdb_api_lang else "ja-JP"
    
    # ページ番号の検証
    if page < 1:
        page = 1
    elif page > 500:  # TMDBの制限
        page = 500
    
    url = "https://api.themoviedb.org/3/person/popular"
    params = {
        "api_key": TMDB_API_KEY, 
        "language": lang_code,
        "page": page
    }
    
    try:
        res = requests.get(url, params=params).json()
        
        if "results" not in res:
            return f"人気順人物リストの取得に失敗しました。（ページ: {page}, 言語: {lang_code}）"
        
        results = res.get("results", [])
        total_pages = res.get("total_pages", 0)
        total_results = res.get("total_results", 0)
        
        if not results:
            return f"ページ {page} に人物データが見つかりませんでした。（総ページ数: {total_pages}）"

        output = []
        output.append(f"page_info: {page}/{total_pages}）")
        output.append(f"total_results: {total_results:,}")
        output.append("")
        
        for i, person in enumerate(results[:15], 1):  # 上位15人を表示
            # 代表作を取得（最大3作品）
            known_for_titles = []
            for work in person.get("known_for", [])[:3]:
                title = work.get("title") or work.get("name", "")
                if title:
                    # 映画かTV番組かを判別
                    media_type = work.get("media_type", "")
                    if media_type == "movie":
                        known_for_titles.append(f"movie_title: {title}")
                    elif media_type == "tv":
                        known_for_titles.append(f"tv_show_title: {title}")
                    else:
                        known_for_titles.append(title)
            
            known_for_str = ", ".join(known_for_titles) if known_for_titles else "代表作情報なし"
            
            # 人気スコア（小数点1桁まで）
            popularity = person.get("popularity", 0)
            
            output.append(
                f"{i:2d}. people_name: {person.get('name', 'N/A')}\n"
                f"    known_for_department: {person.get('known_for_department', 'N/A')}\n"
                f"    popularity: {popularity:.1f}\n"
                f"    known_for: {known_for_str}\n"
            )
        
        # ページネーション情報
        if total_pages > 1:
            output.append(f"page_info: {page}/{total_pages}")
            if page < total_pages:
                output.append(f"To view the next page, specify page={page+1}")
        
        # 検索に使用した言語コードを結果に含める
        output.append(f"language: {lang_code}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"人気順人物リスト取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"人気順人物リスト取得でエラーが発生しました: {str(e)}"


# 引数なしバージョンのシンプルなツールも追加
@tool("tmdb_get_popular_people")
def tmdb_get_popular_people() -> str:
    """引数なしで人気順人物リストを取得（デフォルト設定）
    
    このツールは引数を取りません。Action Input は空文字列にしてください。
    例: Action Input: （何も入力しない）
    """
    return tmdb_popular_people.invoke({"page": 1, "language_code": None})


@tool("web_search_supplement", args_schema=WebSearchInput)
def web_search_supplement(query: str) -> str:
    """TMDBで見つからない映画・TV番組・人物の情報をWebから検索して補完します。
    
    使用場面:
    - TMDB検索で結果が見つからない場合
    - より最新の情報が必要な場合
    - 日本の作品やローカル情報が必要な場合
    - 製作背景や関連ニュースが必要な場合
    """
    # APIキーが設定されているかチェック
    if not os.getenv("TAVILY_API_KEY"):
        return "Web検索を利用するには、TAVILY_API_KEYを設定してください。現在はTMDBデータのみで検索を行ってください。"

    try:
        from langchain_tavily import TavilySearch

        # Tavilyツールを初期化
        tavily_tool = TavilySearch(
            max_results=4,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )

        # 映画・TV・人物関連のクエリに特化
        enhanced_query = f"{query} 映画 テレビ番組 俳優 監督 詳細 情報"

        # 検索を実行
        results = tavily_tool.invoke({"query": enhanced_query})

        # resultsが辞書の場合、results部分を取得
        if isinstance(results, dict):
            search_results = results.get("results", [])
        else:
            search_results = results

        if not search_results:
            return f"「{query}」に関するWeb情報は見つかりませんでした。"

        # 結果をフォーマット
        formatted_results = []
        for i, result in enumerate(search_results[:4], 1):
            title = result.get("title", "不明なタイトル")
            content = result.get("content", "")
            url = result.get("url", "")

            # 内容を適切な長さに制限
            content_preview = content[:200] if content else "内容情報なし"
            if len(content) > 200:
                content_preview += "..."

            formatted_result = f"{i}. **{title}**\n{content_preview}"
            if url:
                formatted_result += f"\n🔗 詳細: {url}"
            formatted_results.append(formatted_result)

        return f"Web search results (supplementary information for '{query}'):\n\n" + "\n\n".join(formatted_results)

    except ImportError:
        return "Web検索ツールが利用できません。langchain-tavilyがインストールされているか確認してください。"
    except Exception as e:
        return f"Web検索でエラーが発生しました: {str(e)[:100]}... TMDBデータで代替検索を試してください。"


@tool("tmdb_trending_all", args_schema=TrendingInput)
def tmdb_trending_all(time_window: str = "day", language_code: Optional[str] = None) -> str:
    """TMDBで全コンテンツ（映画・TV番組・人物）のトレンドを取得します。
    
    time_window パラメータの使い方:
    - 'day': 日別トレンド（今日・直近24時間のトレンド）
    - 'week': 週別トレンド（今週・最近1週間のトレンド）
    
    ユーザーが「今日」「直近」と言った場合は time_window='day' を使用。
    ユーザーが「今週」「最近」「この週」と言った場合は time_window='week' を使用。
    
    """ + get_current_datetime_info()
    
    # time_windowの検証と修正（空文字列対応）
    if not time_window or time_window.strip() == "" or time_window not in ["day", "week"]:
        time_window = "day"
    
    # 言語コードの決定（優先順位: 1.明示的指定 2.環境変数 3.デフォルト ja-JP）
    if language_code:
        lang_code = language_code
    else:
        tmdb_api_lang = os.getenv("TMDB_API_LANG")
        lang_code = tmdb_api_lang if tmdb_api_lang else "ja-JP"
    
    url = f"https://api.themoviedb.org/3/trending/all/{time_window}"
    params = {"api_key": TMDB_API_KEY, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:10]  # 上位10件
        
        if not results:
            return f"トレンドデータが見つかりませんでした。（時間枠: {time_window}, 言語: {lang_code}）"

        output = []
        output.append(f"{time_window} Trend (All Contents)")
        output.append("")
        
        for i, item in enumerate(results, 1):
            media_type = item.get("media_type", "unknown")
            
            if media_type == "movie":
                title = item.get("title", "タイトル不明")
                release_date = item.get("release_date", "N/A")
                vote_average = item.get("vote_average", 0)
                overview = item.get("overview", "")[:100]
                
                output.append(
                    f"{i:2d}. movie_title: {title} ({release_date})\n"
                    f"    vote_average: {vote_average:.1f}/10\n"
                    f"    overview: {overview}{'...' if len(overview) >= 100 else ''}\n"
                )
                
            elif media_type == "tv":
                title = item.get("name", "タイトル不明")
                first_air_date = item.get("first_air_date", "N/A")
                vote_average = item.get("vote_average", 0)
                overview = item.get("overview", "")[:100]
                
                output.append(
                    f"{i:2d}. tv_show_title: {title} ({first_air_date})\n"
                    f"    vote_average: {vote_average:.1f}/10\n"
                    f"    overview: {overview}{'...' if len(overview) >= 100 else ''}\n"
                )
                
            elif media_type == "person":
                name = item.get("name", "名前不明")
                known_for_department = item.get("known_for_department", "N/A")
                popularity = item.get("popularity", 0)
                
                # 代表作を取得
                known_for_titles = []
                for work in item.get("known_for", [])[:2]:
                    work_title = work.get("title") or work.get("name", "")
                    if work_title:
                        known_for_titles.append(work_title)
                
                known_for_str = ", ".join(known_for_titles) if known_for_titles else "代表作情報なし"
                
                output.append(
                    f"{i:2d}. person_name: {name}\n"
                    f"    known_for_department: {known_for_department}\n"
                    f"    popularity: {popularity:.1f}\n"
                    f"    known_for: {known_for_str}\n"
                )

        output.append(f"language: {lang_code}")
        output.append(f"time_window: {time_window}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"全コンテンツトレンド取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"全コンテンツトレンド取得でエラーが発生しました: {str(e)}"


@tool("tmdb_trending_movies", args_schema=TrendingInput)
def tmdb_trending_movies(time_window: str = "day", language_code: Optional[str] = None) -> str:
    """TMDBで映画のトレンドを取得します。
    
    time_window パラメータの使い方:
    - 'day': 日別トレンド（今日・直近24時間のトレンド映画）
    - 'week': 週別トレンド（今週・最近1週間のトレンド映画）
    
    ユーザーが「今日」「直近」と言った場合は time_window='day' を使用。
    ユーザーが「今週」「最近」「この週」と言った場合は time_window='week' を使用。
    
    """ + get_current_datetime_info()
    
    # time_windowの検証と修正（空文字列対応）
    if not time_window or time_window.strip() == "" or time_window not in ["day", "week"]:
        time_window = "day"
    
    # 言語コードの決定
    if language_code:
        lang_code = language_code
    else:
        tmdb_api_lang = os.getenv("TMDB_API_LANG")
        lang_code = tmdb_api_lang if tmdb_api_lang else "ja-JP"
    
    url = f"https://api.themoviedb.org/3/trending/movie/{time_window}"
    params = {"api_key": TMDB_API_KEY, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:10]  # 上位10件
        
        if not results:
            return f"映画のトレンドデータが見つかりませんでした。（時間枠: {time_window}, 言語: {lang_code}）"

        output = []
        time_window_jp = "Daily" if time_window == "day" else "Weekly"
        output.append(f"{time_window_jp} Trending Movies")
        output.append("")
        
        for i, movie in enumerate(results, 1):
            title = movie.get("title", "タイトル不明")
            release_date = movie.get("release_date", "N/A")
            vote_average = movie.get("vote_average", 0)
            popularity = movie.get("popularity", 0)
            overview = movie.get("overview", "")[:150]
            
            output.append(
                f"{i:2d}. title: {title}\n"
                f"    release_date: {release_date}\n"
                f"    vote_average: {vote_average:.1f}/10\n"
                f"    popularity: {popularity:.1f}\n"
                f"    overview: {overview}{'...' if len(overview) >= 150 else ''}\n"
            )
        
        output.append(f"language: {lang_code}")
        output.append(f"time_window: {time_window_jp}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"映画トレンド取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"映画トレンド取得でエラーが発生しました: {str(e)}"


@tool("tmdb_trending_tv", args_schema=TrendingInput)
def tmdb_trending_tv(time_window: str = "day", language_code: Optional[str] = None) -> str:
    """TMDBでTV番組のトレンドを取得します。
    
    time_window パラメータの使い方:
    - 'day': 日別トレンド（今日・直近24時間のトレンドTV番組）
    - 'week': 週別トレンド（今週・最近1週間のトレンドTV番組）
    
    ユーザーが「今日」「直近」と言った場合は time_window='day' を使用。
    ユーザーが「今週」「最近」「この週」と言った場合は time_window='week' を使用。
    
    """ + get_current_datetime_info()
    
    # time_windowの検証と修正（空文字列対応）
    if not time_window or time_window.strip() == "" or time_window not in ["day", "week"]:
        time_window = "day"
    
    # 言語コードの決定
    if language_code:
        lang_code = language_code
    else:
        tmdb_api_lang = os.getenv("TMDB_API_LANG")
        lang_code = tmdb_api_lang if tmdb_api_lang else "ja-JP"
    
    url = f"https://api.themoviedb.org/3/trending/tv/{time_window}"
    params = {"api_key": TMDB_API_KEY, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:10]  # 上位10件
        
        if not results:
            return f"TV番組のトレンドデータが見つかりませんでした。（時間枠: {time_window}, 言語: {lang_code}）"

        output = []
        output.append(f"{time_window} Trending TV Shows")
        output.append("")
        
        for i, tv_show in enumerate(results, 1):
            name = tv_show.get("name", "タイトル不明")
            first_air_date = tv_show.get("first_air_date", "N/A")
            vote_average = tv_show.get("vote_average", 0)
            popularity = tv_show.get("popularity", 0)
            overview = tv_show.get("overview", "")[:150]
            
            output.append(
                f"{i:2d}. name: {name} ({first_air_date})\n"
                f"    vote_average: {vote_average:.1f}/10\n"
                f"    popularity: {popularity:.1f}\n"
                f"    overview: {overview}{'...' if len(overview) >= 150 else ''}\n"
            )

        output.append(f"language: {lang_code}")
        output.append(f"time_window: {time_window}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"TV番組トレンド取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"TV番組トレンド取得でエラーが発生しました: {str(e)}"


@tool("tmdb_trending_people", args_schema=TrendingInput)
def tmdb_trending_people(time_window: str = "day", language_code: Optional[str] = None) -> str:
    """TMDBで人物のトレンドを取得します。
    
    time_window パラメータの使い方:
    - 'day': 日別トレンド（今日・直近24時間のトレンド人物）
    - 'week': 週別トレンド（今週・最近1週間のトレンド人物）
    
    ユーザーが「今日」「直近」と言った場合は time_window='day' を使用。
    ユーザーが「今週」「最近」「この週」と言った場合は time_window='week' を使用。
    
    """ + get_current_datetime_info()
    
    # time_windowの検証と修正（空文字列対応）
    if not time_window or time_window.strip() == "" or time_window not in ["day", "week"]:
        time_window = "day"
    
    # 言語コードの決定
    if language_code:
        lang_code = language_code
    else:
        tmdb_api_lang = os.getenv("TMDB_API_LANG")
        lang_code = tmdb_api_lang if tmdb_api_lang else "ja-JP"
    
    url = f"https://api.themoviedb.org/3/trending/person/{time_window}"
    params = {"api_key": TMDB_API_KEY, "language": lang_code}
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])[:15]  # 上位15件
        
        if not results:
            return f"人物のトレンドデータが見つかりませんでした。（時間枠: {time_window}, 言語: {lang_code}）"

        output = []

        output.append(f"{time_window} Trending People")
        output.append("")
        
        for i, person in enumerate(results, 1):
            name = person.get("name", "名前不明")
            known_for_department = person.get("known_for_department", "N/A")
            popularity = person.get("popularity", 0)
            
            # 代表作を取得（最大3作品）
            known_for_titles = []
            for work in person.get("known_for", [])[:3]:
                work_title = work.get("title") or work.get("name", "")
                if work_title:
                    media_type = work.get("media_type", "")
                    if media_type == "movie":
                        known_for_titles.append(f"movie_work_title: {work_title}")
                    elif media_type == "tv":
                        known_for_titles.append(f"tv_work_title: {work_title}")
                    else:
                        known_for_titles.append(f"work_title: {work_title}")

            known_for_str = ", ".join(known_for_titles) if known_for_titles else "代表作情報なし"
            
            output.append(
                f"{i:2d}. name: {name}\n"
                f"    known_for_department: {known_for_department}\n"
                f"    popularity: {popularity:.1f}\n"
                f"    known_for: {known_for_str}\n"
            )

        output.append(f"language: {lang_code}")
        output.append(f"time_window: {time_window}")
        return "\n".join(output)
        
    except requests.RequestException as e:
        return f"人物トレンド取得でネットワークエラーが発生しました: {str(e)}"
    except Exception as e:
        return f"人物トレンド取得でエラーが発生しました: {str(e)}"


# 引数なしバージョンのシンプルなツールも追加（LangChainエージェント互換）
@tool("tmdb_get_trending_all")
def tmdb_get_trending_all() -> str:
    """引数なしで全コンテンツの日別トレンドを取得（デフォルト設定）
    
    このツールは引数を取りません。Action Input は空文字列にしてください。
    例: Action Input: （何も入力しない）
    """
    return tmdb_trending_all.invoke({"time_window": "day", "language_code": None})

@tool("tmdb_get_trending_movies")
def tmdb_get_trending_movies() -> str:
    """引数なしで映画の日別トレンドを取得（デフォルト設定）
    
    このツールは引数を取りません。Action Input は空文字列にしてください。
    例: Action Input: （何も入力しない）
    """
    return tmdb_trending_movies.invoke({"time_window": "day", "language_code": None})

@tool("tmdb_get_trending_tv")
def tmdb_get_trending_tv() -> str:
    """引数なしでTV番組の日別トレンドを取得（デフォルト設定）
    
    このツールは引数を取りません。Action Input は空文字列にしてください。
    例: Action Input: （何も入力しない）
    """
    return tmdb_trending_tv.invoke({"time_window": "day", "language_code": None})

@tool("tmdb_get_trending_people")
def tmdb_get_trending_people() -> str:
    """引数なしで人物の日別トレンドを取得（デフォルト設定）
    
    このツールは引数を取りません。Action Input は空文字列にしてください。
    例: Action Input: （何も入力しない）
    """
    return tmdb_trending_people.invoke({"time_window": "day", "language_code": None})


@tool("theme_song_search", args_schema=ThemeSongSearchInput)
def theme_song_search(query: str) -> str:
    """映画・アニメ・ドラマの主題歌・エンディング・挿入歌や歌手情報をWebから検索します。
    
    使用場面:
    - 映画・アニメ・ドラマの主題歌を知りたい場合
    - 特定の楽曲がどの作品で使われているか調べたい場合
    - 主題歌を歌っているアーティスト・歌手の情報が欲しい場合
    - サウンドトラック情報が必要な場合
    """
    # APIキーが設定されているかチェック
    if not os.getenv("TAVILY_API_KEY"):
        return "主題歌検索を利用するには、TAVILY_API_KEYを設定してください。現在はTMDBデータのみで検索を行ってください。"

    try:
        from langchain_tavily import TavilySearch

        # Tavilyツールを初期化
        tavily_tool = TavilySearch(
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )

        # 主題歌・楽曲関連のクエリに特化
        enhanced_query = f"{query} 主題歌 エンディング 挿入歌 テーマソング 歌手 アーティスト サウンドトラック"

        # 検索を実行
        results = tavily_tool.invoke({"query": enhanced_query})

        # resultsが辞書の場合、results部分を取得
        if isinstance(results, dict):
            search_results = results.get("results", [])
        else:
            search_results = results

        if not search_results:
            return f"「{query}」に関する主題歌・楽曲情報は見つかりませんでした。"

        # 結果をフォーマット
        formatted_results = []
        for i, result in enumerate(search_results[:5], 1):
            title = result.get("title", "不明なタイトル")
            content = result.get("content", "")
            url = result.get("url", "")

            # 内容を適切な長さに制限
            content_preview = content[:250] if content else "内容情報なし"
            if len(content) > 250:
                content_preview += "..."

            formatted_result = f"{i}. **{title}**\n{content_preview}"
            formatted_results.append(formatted_result)

        return f"🎵 主題歌・楽曲検索結果（「{query}」）：\n\n" + "\n\n".join(formatted_results)

    except ImportError:
        return "主題歌検索ツールが利用できません。langchain-tavilyがインストールされているか確認してください。"
    except Exception as e:
        return f"主題歌検索でエラーが発生しました: {str(e)[:100]}... TMDBデータで代替検索を試してください。"


# 旧Tool定義を削除し、新しい@toolで定義されたツールのリストを作成

# ツールリスト（@toolデコレーターで定義されたツール）
TOOLS = [
    tmdb_movie_search,
    tmdb_person_search,
    tmdb_tv_search,
    tmdb_multi_search,
    tmdb_movie_credits_search,
    tmdb_tv_credits_search,
    tmdb_credits_search_by_id,
    tmdb_popular_people,
    tmdb_get_popular_people,
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
]

# プロンプト用のツール説明文
TOOLS_TEXT = "tmdb_movie_search: 映画の具体的なタイトルで検索\ntmdb_tv_search: TV番組の具体的なタイトルで検索\ntmdb_person_search: 具体的な人名で検索\ntmdb_multi_search: 映画・TV・人物を横断検索\ntmdb_movie_credits_search: 映画の詳細なクレジット情報を取得（タイトル検索）\ntmdb_tv_credits_search: TV番組の詳細なクレジット情報を取得（タイトル検索）\ntmdb_credits_search_by_id: 映画IDまたはTV番組IDを直接指定してクレジット情報を取得\ntmdb_popular_people: 人気順で人物リストを取得（ページ指定可能）\ntmdb_get_popular_people: 人気順で人物リストを取得（引数なし：Action Input は空で）\ntmdb_trending_all: 全コンテンツのトレンド取得（time_window: day=今日・直近, week=今週・最近）\ntmdb_trending_movies: 映画のトレンド取得（time_window: day=今日・直近, week=今週・最近）\ntmdb_trending_tv: TV番組のトレンド取得（time_window: day=今日・直近, week=今週・最近）\ntmdb_trending_people: 人物のトレンド取得（time_window: day=今日・直近, week=今週・最近）\ntmdb_get_trending_all: 全コンテンツの日別トレンドを取得（引数なし：Action Input は空で）\ntmdb_get_trending_movies: 映画の日別トレンドを取得（引数なし：Action Input は空で）\ntmdb_get_trending_tv: TV番組の日別トレンドを取得（引数なし：Action Input は空で）\ntmdb_get_trending_people: 人物の日別トレンドを取得（引数なし：Action Input は空で）\nweb_search_supplement: TMDBで見つからない情報をWebから検索して補完\ntheme_song_search: 映画・アニメ・ドラマの主題歌・楽曲・歌手情報をWebから検索"
TOOL_NAMES = "tmdb_movie_search, tmdb_tv_search, tmdb_person_search, tmdb_multi_search, tmdb_movie_credits_search, tmdb_tv_credits_search, tmdb_credits_search_by_id, tmdb_popular_people, tmdb_get_popular_people, tmdb_trending_all, tmdb_trending_movies, tmdb_trending_tv, tmdb_trending_people, tmdb_get_trending_all, tmdb_get_trending_movies, tmdb_get_trending_tv, tmdb_get_trending_people, web_search_supplement, theme_song_search"


# エクスポート用の関数リスト
__all__ = [
    # @toolデコレーター定義のツール
    "tmdb_movie_search",
    "tmdb_person_search", 
    "tmdb_tv_search",
    "tmdb_multi_search",
    "tmdb_movie_credits_search",
    "tmdb_tv_credits_search",
    "tmdb_credits_search_by_id",
    "tmdb_popular_people",
    "tmdb_get_popular_people",
    "tmdb_trending_all",
    "tmdb_trending_movies",
    "tmdb_trending_tv",
    "tmdb_trending_people",
    "tmdb_get_trending_all",
    "tmdb_get_trending_movies",
    "tmdb_get_trending_tv",
    "tmdb_get_trending_people",
    "web_search_supplement",
    "theme_song_search",
    # ツールリスト
    "TOOLS",
    # ユーティリティ関数
    "get_supported_languages",
    "get_available_tools",
    "detect_language_and_get_tmdb_code",
    "get_language_code",
    "get_current_datetime_info",
    "TOOLS_TEXT",
    "TOOL_NAMES",
    "SUPPORTED_LANGUAGES",
]
