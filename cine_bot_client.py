#!/usr/bin/env python3
"""
CineBot WebSocket クライアント サンプル

StarletteベースのCineBotサーバーに接続してテストするためのクライアント
"""

import asyncio
import json
import logging
import sys
from typing import Optional, Any

try:
    import websockets
except ImportError:
    print("❌ websocketsライブラリがインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("uv add websockets")
    print("または:")
    print("pip install websockets")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CineBotClient:
    """CineBot WebSocketクライアント"""
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.websocket: Optional[Any] = None
        
    async def connect(self):
        """サーバーに接続"""
        try:
            self.websocket = await websockets.connect(self.url)
            logger.info(f"✅ CineBotサーバーに接続しました: {self.url}")
            return True
        except Exception as e:
            logger.error(f"❌ 接続に失敗しました: {e}")
            return False
    
    async def disconnect(self):
        """サーバーから切断"""
        if self.websocket:
            await self.websocket.close()
            logger.info("✅ サーバーから切断しました")
    
    async def send_text_message(self, text: str):
        """テキストメッセージを送信"""
        if not self.websocket:
            logger.error("❌ サーバーに接続されていません")
            return
        
        try:
            # OpenAI Realtime API形式でメッセージを送信
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 送信: {text}")
            
        except Exception as e:
            logger.error(f"❌ メッセージ送信エラー: {e}")
    
    async def listen_for_responses(self):
        """サーバーからの応答を受信"""
        if not self.websocket:
            logger.error("❌ サーバーに接続されていません")
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_response(data)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ JSON以外のメッセージ: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("📋 サーバーとの接続が閉じられました")
        except Exception as e:
            logger.error(f"❌ 受信エラー: {e}")
    
    async def handle_response(self, data: dict):
        """サーバーからの応答を処理"""
        response_type = data.get("type", "unknown")
        
        if response_type == "connection_established":
            logger.info(f"🎬 {data.get('message', '接続確立')}")
            
        elif response_type == "text_response":
            content = data.get("content", "")
            if content.strip():
                print(f"\n🤖 CineBot: {content}")
                
        elif response_type == "response.audio.delta":
            # 音声データの場合（実際の実装では音声再生処理を行う）
            logger.debug("🔊 音声データを受信")
            
        else:
            # その他の応答
            logger.debug(f"📨 応答 ({response_type}): {json.dumps(data, ensure_ascii=False, indent=2)}")


async def interactive_mode(client: CineBotClient):
    """対話モード"""
    print("\n🎬 CineBot 対話モード")
    print("=" * 40)
    print("映画やTV番組について何でもお聞きください！")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("=" * 40)
    
    # 応答受信タスクを開始
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    try:
        while True:
            try:
                # ユーザー入力を取得
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\n👤 あなた: "
                )
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 またお会いしましょう！")
                    break
                
                if user_input.strip():
                    await client.send_text_message(user_input)
                    
            except (EOFError, KeyboardInterrupt):
                print("\n👋 対話を終了します")
                break
                
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass


async def test_mode(client: CineBotClient):
    """テストモード - 自動で複数の質問を送信"""
    print("\n🧪 CineBot テストモード")
    print("=" * 40)
    
    test_questions = [
        "こんにちは！80年代で面白い映画を教えて",
        "タイムスリップ系の映画でおすすめはある？",
        "ナウシカが好きなんだけど、似たような作品ある？",
        "最新のトレンド映画はなに？",
    ]
    
    # 応答受信タスクを開始
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    try:
        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] テスト質問: {question}")
            await client.send_text_message(question)
            await asyncio.sleep(3)  # 応答を待つ
            
        print("\n✅ すべてのテスト質問を送信しました")
        await asyncio.sleep(5)  # 最後の応答を待つ
        
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass


async def main():
    """メイン実行関数"""
    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("CineBot WebSocketクライアント")
            print("\n使用方法:")
            print("  python cine_bot_client.py [URL] [MODE]")
            print("\n引数:")
            print("  URL   : WebSocketサーバーのURL (デフォルト: ws://localhost:8000/ws)")
            print("  MODE  : 実行モード (interactive, test) (デフォルト: interactive)")
            print("\n例:")
            print("  python cine_bot_client.py")
            print("  python cine_bot_client.py ws://localhost:8000/ws interactive")
            print("  python cine_bot_client.py ws://localhost:8000/ws test")
            return
    
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
    mode = sys.argv[2] if len(sys.argv) > 2 else "interactive"
    
    print("🎬 CineBot WebSocketクライアント")
    print(f"接続先: {url}")
    print(f"モード: {mode}")
    
    # クライアントを作成
    client = CineBotClient(url)
    
    try:
        # サーバーに接続
        if not await client.connect():
            return
        
        # モードに応じて実行
        if mode == "test":
            await test_mode(client)
        else:
            await interactive_mode(client)
            
    except KeyboardInterrupt:
        print("\n\n👋 クライアントを終了します")
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AttributeError:
        # Python3.6以下の場合
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
