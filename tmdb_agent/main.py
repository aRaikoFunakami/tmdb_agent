"""
TMDB検索エージェントのデモ実行ファイル

このファイルは、TMDBSearchAgentのデモンストレーションを実行します。
実際のエージェント実装は tmdb_search_agent.py にあります。

実行モード:
- 自動テスト: 事前定義されたテストケースを順次実行
- チャット形式: ユーザーとの対話形式でメモリ機能付き
"""

import sys

# 相対インポートと絶対インポートの両方に対応
try:
    # パッケージとして実行される場合（相対インポート）
    from .agent import create_tmdb_agent
except ImportError:
    # 直接実行される場合（絶対インポート）
    from agent import create_tmdb_agent

# LangChainメモリのインポート - 新しいAPIを使用
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List


class WindowedChatHistory:
    """ウィンドウサイズ制限付きチャット履歴管理
    
    LangChainの新しいメモリAPIに対応したカスタム実装
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        """メッセージを追加し、ウィンドウサイズを維持"""
        self.messages.append(message)
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    
    def get_messages(self) -> List[BaseMessage]:
        """メッセージ履歴を取得"""
        return self.messages.copy()
    
    def clear(self) -> None:
        """メッセージ履歴をクリア"""
        self.messages.clear()


class TMDBChatSession:
    """
    メモリ機能付きTMDB検索チャットセッション
    
    新しいLangChain APIのWindowedChatHistoryを使用して
    短期記憶（会話履歴）を管理します。
    """
    
    def __init__(self, agent, memory_window: int = 10):
        """
        チャットセッションを初期化
        
        Args:
            agent: TMDBSearchAgentインスタンス
            memory_window: 記憶する会話のターン数（デフォルト: 10）
        """
        self.agent = agent
        self.memory = WindowedChatHistory(window_size=memory_window * 2)  # ユーザー+AI両方のメッセージを考慮
        self.turn_count = 0
    
    def chat(self, user_input: str) -> str:
        """
        ユーザー入力に対してメモリを考慮した応答を生成
        
        Args:
            user_input: ユーザーの入力
            
        Returns:
            エージェントの応答
        """
        self.turn_count += 1
        
        # メモリから会話履歴を取得
        chat_history = self.memory.get_messages()
        
        # 会話履歴がある場合は、コンテキストを含めたクエリを作成
        if chat_history:
            context_messages = []
            # 直近6メッセージ（3ターン分）のみ使用
            for msg in chat_history[-6:]:
                if isinstance(msg, HumanMessage):
                    context_messages.append(f"ユーザー: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_messages.append(f"AI: {msg.content}")
            
            context = "\n".join(context_messages)
            enhanced_query = f"""
前回の会話:
{context}

現在の質問: {user_input}

上記の会話履歴を考慮して、現在の質問に答えてください。前回の検索結果と関連がある場合は、それを参考にしてください。
"""
        else:
            enhanced_query = user_input
        
        # エージェントに問い合わせ
        response = self.agent.search(enhanced_query)
        
        # メモリに会話を保存
        self.memory.add_message(HumanMessage(content=user_input))
        self.memory.add_message(AIMessage(content=response))
        
        return response
    
    def get_memory_stats(self) -> dict:
        """
        メモリの統計情報を取得
        
        Returns:
            メモリ統計情報の辞書
        """
        return {
            "total_turns": self.turn_count,
            "messages_in_memory": len(self.memory.get_messages()),
            "memory_window": self.memory.window_size // 2  # ユーザー+AIペアでカウント
        }
    
    def clear_memory(self):
        """メモリをクリア"""
        self.memory.clear()
        self.turn_count = 0


def list_available_tests():
    """利用可能なテストケースの一覧を表示"""
    test_cases = [
        "映画クレジット情報取得",
        "映画検索（曖昧な説明からの推論）",
        "TV番組検索",
        "マルチ検索（映画・TV混在）",
        "人物検索",
        "人気順人物リスト取得",
        "全コンテンツトレンド取得",
        "映画トレンド取得",
        "TV番組トレンド取得",
        "人物トレンド取得",
        "多言語対応テスト",
        "主題歌検索テスト（アニメ）",
        "主題歌検索テスト（映画）",
        "制作会社検索テスト",
        "制作会社による映画検索テスト",
        "推薦機能テスト（映画）",
        "推薦機能テスト（TV番組）",
        "推薦機能テスト（映画とTV両方）",
        "高機能推薦テスト（複数タイトル統合）",
        "高機能推薦テスト（アニメ複数）",
    ]
    
    print("=== 利用可能なテストケース ===")
    for i, title in enumerate(test_cases, 1):
        print(f"{i:2d}. {title}")
    print(f"\n総計: {len(test_cases)}個のテストケース")
    print("\n使用例:")
    print("  python main.py --auto 1,3,5      # テスト1,3,5を実行")
    print("  python main.py --auto 1-5        # テスト1から5まで実行")  
    print("  python main.py --auto all        # 全テストを実行")


def parse_test_selection(selection_str):
    """
    テスト選択文字列を解析して番号リストを返す
    
    Args:
        selection_str: "1,3,5" or "1-5" or "all" 形式の文字列
        
    Returns:
        テスト番号のリスト
    """
    if not selection_str or selection_str.lower() == 'all':
        return None  # 全テスト実行
    
    test_numbers = []
    
    # カンマ区切りで分割
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        
        # ハイフン区切りの範囲指定をチェック
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                test_numbers.extend(range(start, end + 1))
            except ValueError:
                print(f"⚠️ 不正な範囲指定: {part}")
        else:
            # 単一の番号
            try:
                test_numbers.append(int(part))
            except ValueError:
                print(f"⚠️ 不正な番号: {part}")
    
    return sorted(list(set(test_numbers)))  # 重複除去とソート


def run_auto_tests(selected_tests=None, debug_mode=False):
    """
    事前定義されたテストケースを自動実行
    
    Args:
        selected_tests: 実行するテストケースの番号リスト（None の場合は全て実行）
        debug_mode: デバッグモード（詳細ログ出力）
    
    複数のテストケースを使って、エージェントの機能をデモンストレーションします。
    """
    print("=== TMDB検索エージェント 自動テスト ===")
    if debug_mode:
        print("🔍 デバッグモード: 詳細ログを出力します")

    # OpenAI LLMを作成
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)  # 温度を下げて一貫性を向上

    # エージェントを作成
    agent = create_tmdb_agent(llm, verbose=True)

    # テストケース
    test_cases = [
        {
            "title": "映画クレジット情報取得",
            "query": "バック・トゥ・ザ・フューチャーの監督と出演者を教えて。",
        },
        {
            "title": "映画検索（曖昧な説明からの推論）",
            "query": "昔見た映画で、車がタイムマシンになって未来に行くやつ。80年代っぽい雰囲気だったかも。",
        },
        {
            "title": "TV番組検索", 
            "query": "進撃の巨人のアニメについて詳しく教えて。"
        },
        {
            "title": "マルチ検索（映画・TV混在）",
            "query": "スターウォーズについて教えて。映画もTV番組もあるよね？",
        },
        {
            "title": "人物検索", 
            "query": "新海誠監督について教えて。代表作も調べて。"
        },
        {
            "title": "人気順人物リスト取得",
            "query": "今人気の俳優や女優を教えて。トップ10くらいで。"
        },
        {
            "title": "全コンテンツトレンド取得",
            "query": "今日のトレンド（映画・TV・人物）を教えて。"
        },
        {
            "title": "映画トレンド取得",
            "query": "トレンド映画を教えて。"
        },
        {
            "title": "TV番組トレンド取得",
            "query": "話題のTV番組を教えて。"
        },
        {
            "title": "人物トレンド取得",
            "query": "今話題の人物（俳優・監督など）を教えて。"
        },
        {
            "title": "多言語対応テスト", 
            "query": "Show me movies produced by Marvel Studios"
        },
        {
            "title": "主題歌検索テスト（アニメ）",
            "query": "鬼滅の刃の主題歌を教えて。歌手は誰？"
        },
        {
            "title": "主題歌検索テスト（映画）",
            "query": "君の名はの主題歌とRADWIMPSについて詳しく"
        },
        {
            "title": "制作会社検索テスト",
            "query": "Studio Ghibliという制作会社について教えて。"
        },
        {
            "title": "制作会社による映画検索テスト",
            "query": "Marvel Studiosが制作した映画を人気順で教えて。"
        },
        {
            "title": "推薦機能テスト（映画）",
            "query": "バック・トゥ・ザ・フューチャーに似た映画を5つ推薦して。"
        },
        {
            "title": "推薦機能テスト（TV番組）",
            "query": "進撃の巨人に似たアニメを3つ推薦して。"
        },
        {
            "title": "推薦機能テスト（映画とTV両方）",
            "query": "スター・ウォーズに似た作品を映画とTV番組両方から推薦して。"
        },
        {
            "title": "高機能推薦テスト（複数タイトル統合）",
            "query": "バック・トゥ・ザ・フューチャー、ターミネーター、スター・ウォーズに似た作品を統合して推薦して。最終的に7つの作品を選んで。"
        },
        {
            "title": "高機能推薦テスト（アニメ複数）",
            "query": "進撃の巨人を見たことがあります。鬼滅の刃に興味があります。ワンピースを検索しました。おすすめのアニメを教えて。"
        },
    ]

    # 選択されたテストケースのみ実行
    if selected_tests:
        # 指定されたテスト番号を検証
        valid_tests = []
        for test_num in selected_tests:
            if 1 <= test_num <= len(test_cases):
                valid_tests.append(test_num)
            else:
                print(f"⚠️ テスト番号 {test_num} は存在しません（1-{len(test_cases)}の範囲で指定してください）")
        
        if not valid_tests:
            print("❌ 実行可能なテストがありません")
            return
            
        selected_tests = valid_tests
        print(f"📋 選択されたテスト: {selected_tests}")
    else:
        selected_tests = list(range(1, len(test_cases) + 1))
        print(f"📋 全テストを実行します（{len(test_cases)}件）")

    # 各テストケースを実行
    for i, test_case in enumerate(test_cases, 1):
        if i in selected_tests:
            print(f"\n=== テスト {i}: {test_case['title']} ===")
            print(f"クエリ: {test_case['query']}")
            
            if debug_mode:
                # デバッグモードでは詳細な実行情報を取得
                print(f"🔍 クエリ: {test_case['query']}")
                print("🔍 実行開始...")
                
                try:
                    # 詳細な結果を取得
                    detailed_result = agent.search_detailed(test_case["query"])
                    
                    print("🔍 実行完了")
                    print(f"🔍 中間ステップ数: {len(detailed_result.get('intermediate_steps', []))}")
                    
                    # 中間ステップの詳細を表示
                    for step_idx, step in enumerate(detailed_result.get('intermediate_steps', []), 1):
                        print(f"🔍 ステップ {step_idx}: {step}")
                    
                    result = detailed_result.get("output", "結果を取得できませんでした")
                    
                except Exception as e:
                    print(f"🔍 エラー詳細: {type(e).__name__}: {str(e)}")
                    # エラーの場合も通常の方法で再試行
                    result = agent.search(test_case["query"])
            else:
                # 通常モード
                result = agent.search(test_case["query"])
            
            print(f"結果: {result}")

            if i < max(selected_tests):
                print("\n" + "=" * 60 + "\n")


def run_chat_mode():
    """
    インタラクティブなチャット形式でのテスト
    
    ユーザーとの対話形式で、メモリ機能付きの会話を提供します。
    """
    print("=== TMDB検索エージェント チャットモード ===")
    print("メモリ機能付きでTMDBについて何でも聞いてください！")
    print("終了するには 'quit', 'exit', 'q' を入力してください。")
    print("メモリをクリアするには 'clear' を入力してください。")
    print("メモリ統計を見るには 'stats' を入力してください。")
    print("=" * 60)

    # OpenAI LLMを作成
    from langchain_openai import ChatOpenAI
    
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    except Exception as e:
        print(f"❌ LLMの初期化に失敗しました: {e}")
        print("OPENAI_API_KEYが設定されているか確認してください。")
        return

    # エージェントとチャットセッションを作成
    agent = create_tmdb_agent(llm, verbose=True)  # チャットモードではverbose=Falseに
    chat_session = TMDBChatSession(agent, memory_window=10)

    print("✅ チャットセッション開始！")
    
    while True:
        try:
            # ユーザー入力を取得
            user_input = input("\n🎬 あなた: ").strip()
            
            # 終了コマンド
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 チャットセッションを終了します。ありがとうございました！")
                break
            
            # メモリクリアコマンド
            if user_input.lower() == 'clear':
                chat_session.clear_memory()
                print("🧹 メモリをクリアしました。")
                continue
            
            # 統計表示コマンド
            if user_input.lower() == 'stats':
                stats = chat_session.get_memory_stats()
                print("📊 メモリ統計:")
                print(f"   - 総会話ターン数: {stats['total_turns']}")
                print(f"   - メモリ内メッセージ数: {stats['messages_in_memory']}")
                print(f"   - メモリウィンドウサイズ: {stats['memory_window']}")
                continue
            
            # 空入力をスキップ
            if not user_input:
                print("💭 何か質問してください...")
                continue
            
            # エージェントに問い合わせ
            print("\n🤖 AI: 調べています...")
            response = chat_session.chat(user_input)
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 チャットセッションを終了します。")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            print("もう一度お試しください。")


def show_help():
    """ヘルプメッセージを表示"""
    print("=== TMDB検索エージェント ===")
    print()
    print("使用方法:")
    print("  python main.py [オプション] [テスト選択]")
    print()
    print("オプション:")
    print("  --auto, -a [選択]  自動テストを実行")
    print("                     選択形式:")
    print("                       なし または all : 全テスト実行")
    print("                       1,3,5          : テスト1,3,5を実行")
    print("                       1-5            : テスト1から5まで実行")
    print("                       1,3-5,8        : テスト1,3,4,5,8を実行")
    print("  --debug [選択]     デバッグモードで自動テストを実行（詳細ログ付き）")
    print("  --list, -l         利用可能なテストケース一覧を表示")
    print("  --chat, -c         チャット形式で実行")
    print("  --help, -h         このヘルプを表示")
    print()
    print("デフォルト: チャット形式で実行")
    print()
    print("使用例:")
    print("  python main.py --auto              # 全テスト実行")
    print("  python main.py --auto 1,3,5        # 特定テスト実行")
    print("  python main.py --auto 1-5          # 範囲指定実行")
    print("  python main.py --debug 3,13        # 問題のあるテストをデバッグモードで実行")
    print("  python main.py --list              # テスト一覧表示")
    print("  python main.py --chat              # チャットモード")
    print()
    print("チャットモードでのコマンド:")
    print("  quit/exit/q    終了")
    print("  clear          メモリをクリア")
    print("  stats          メモリ統計を表示")

def main():
    """メイン関数 - コマンドライン引数を処理して適切なモードを実行"""
    
    # コマンドライン引数を処理
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--auto', '-a']:
            # テスト選択の解析
            if len(sys.argv) > 2:
                selection = sys.argv[2]
                selected_tests = parse_test_selection(selection)
            else:
                selected_tests = None  # 全テスト実行
            
            run_auto_tests(selected_tests, debug_mode=False)
            
        elif arg == '--debug':
            # デバッグモードでのテスト実行
            if len(sys.argv) > 2:
                selection = sys.argv[2]
                selected_tests = parse_test_selection(selection)
            else:
                selected_tests = None  # 全テスト実行
            
            run_auto_tests(selected_tests, debug_mode=True)
            
        elif arg in ['--list', '-l']:
            list_available_tests()
            
        elif arg in ['--chat', '-c']:
            run_chat_mode()
            
        elif arg in ['--help', '-h']:
            show_help()
            
        else:
            print(f"❌ 不明なオプション: {arg}")
            print("使用可能なオプション: --auto, --debug, --list, --chat, --help")
            show_help()
    else:
        # デフォルトはチャットモード
        run_chat_mode()


if __name__ == "__main__":
    main()
