#!/usr/bin/env python3
"""
CineBot サンプル実行スクリプト

音声対応映画・TV番組レコメンデーションボットのデモンストレーション
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tmdb_agent.cine_bot import create_cine_bot, test_cine_bot


async def demo_text_input():
    """テキスト入力でのCineBotデモ"""
    print("🎬 CineBot テキスト入力デモ開始")
    print("=" * 50)
    
    # CineBotを作成
    bot = create_cine_bot(verbose=True)
    
    # テスト質問リスト
    test_questions = [
        "こんにちは！80年代で面白い映画を教えて",
        "タイムスリップ系の映画でおすすめはある？",
        "ナウシカが好きなんだけど、似たような作品ある？",
        "最新のトレンド映画はなに？",
        "感動できる映画を教えて",
    ]
    
    # テキスト入力ストリームを模擬
    async def text_input_stream():
        for question in test_questions:
            print(f"\n👤 ユーザー: {question}")
            # ユーザーのテキスト入力をJSON形式に変換
            yield json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}]
                }
            })
            await asyncio.sleep(2)  # 質問間の間隔
    
    # 出力処理
    async def handle_output(chunk: str):
        try:
            # JSONとしてパースを試行
            data = json.loads(chunk)
            if data.get("type") == "response.audio.delta":
                # 音声データの場合は無視（テキストデモなので）
                pass
            else:
                print(f"🎵 システム応答: {chunk}")
        except json.JSONDecodeError:
            # テキスト応答の場合
            if chunk.strip():
                print(f"🤖 CineBot: {chunk}")
    
    try:
        # CineBotに接続してデモ実行
        await bot.aconnect(text_input_stream(), handle_output)
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("\n🎬 CineBot テキスト入力デモ終了")


async def demo_recommendation_tool():
    """レコメンデーション機能の直接テスト"""
    print("\n🔧 CineBot レコメンデーション機能テスト")
    print("=" * 50)
    
    await test_cine_bot()


def check_environment():
    """環境設定の確認"""
    print("🔍 環境設定チェック")
    print("=" * 30)
    
    # 必要な環境変数をチェック
    required_env_vars = ["OPENAI_API_KEY", "TMDB_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if os.getenv(var):
            print(f"✅ {var}: 設定済み")
        else:
            print(f"❌ {var}: 未設定")
            missing_vars.append(var)
    
    if missing_vars:
        print("\n⚠️ 以下の環境変数を設定してください:")
        for var in missing_vars:
            print(f"   export {var}=your_api_key_here")
        return False
    
    print("✅ 環境設定OK")
    return True


async def main():
    """メイン実行関数"""
    print("🎬 CineBot デモンストレーション")
    print("=" * 50)
    print("音声対応映画・TV番組レコメンデーションボット")
    print("=" * 50)
    
    # 環境チェック
    if not check_environment():
        print("\n❌ 環境設定が不完全です。上記の指示に従って設定してください。")
        return
    
    try:
        # レコメンデーション機能テスト
        await demo_recommendation_tool()
        
        # テキスト入力デモ（オプション）
        print("\n" + "=" * 50)
        response = input("テキスト入力デモを実行しますか？ (y/N): ")
        if response.lower() in ['y', 'yes']:
            await demo_text_input()
    
    except KeyboardInterrupt:
        print("\n\n👋 デモを終了します")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Python3.7+のasyncio互換性
    try:
        asyncio.run(main())
    except AttributeError:
        # Python3.6以下の場合
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
