#!/usr/bin/env python3
"""
CineBot WebSocket サーバー (Starlette + Uvicorn版)

音声対応映画・TV番組レコメンデーションボットのWebSocketサーバー実装
StarletteとUvicornを使用してリアルタイムで音声入力を受け取り、音声またはテキストで応答します。
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Set

from starlette.applications import Starlette
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
import uvicorn

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tmdb_agent.cine_bot import create_cine_bot
except ImportError:
    # 直接インポートを試行
    sys.path.insert(0, str(Path(__file__).parent))
    from tmdb_agent.cine_bot import create_cine_bot

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 接続されたクライアントを管理
connected_clients: Set[WebSocket] = set()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続のエンドポイント"""
    # Accept with query params
    await websocket.accept()
    connected_clients.add(websocket)

    logger.info(f"Client {websocket.client} connected. Total clients: {len(connected_clients)}")

    try:
        # 言語パラメータ取得（例: ws://host/ws?language=ja）
        language = websocket.query_params.get("language", "ja")
        logger.info(f"Language param from client: {language}")

        # CineBotインスタンスを作成（languageを渡す）
        cine_bot = create_cine_bot(verbose=True, language=language)

        # 入力ストリームを作成
        input_queue = asyncio.Queue()

        # 接続確認メッセージを送信
        await websocket.send_json({
            "type": "connection_established",
            "message": "🎬 Connected to CineBot. Ask anything about movies or TV shows!",
            "timestamp": asyncio.get_event_loop().time(),
            "language": language
        })

        # クライアントからのメッセージを受信するタスク
        async def receive_messages():
            try:
                while True:
                    message = await websocket.receive_text()
                    logger.debug(f"Received message from {websocket.client}: {message[:100]}")
                    await input_queue.put(message)
            except WebSocketDisconnect:
                logger.info(f"Client {websocket.client} disconnected")
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
            finally:
                await input_queue.put(None)  # 終了シグナル

        # 入力ストリームジェネレーター
        async def input_stream():
            while True:
                message = await input_queue.get()
                if message is None:
                    break
                yield message

        # 出力処理
        async def send_output_chunk(chunk: str):
            try:
                # WebSocketが開いているかチェック
                if websocket.client_state.name == "CONNECTED":
                    try:
                        # JSONとしてパースを試行
                        data = json.loads(chunk)
                        await websocket.send_json(data)
                    except json.JSONDecodeError:
                        # テキスト応答の場合
                        if chunk.strip():
                            await websocket.send_json({
                                "type": "text_response",
                                "content": chunk,
                                "timestamp": asyncio.get_event_loop().time()
                            })

                    logger.debug(f"Sent chunk to {websocket.client}: {chunk[:100]}...")
                else:
                    logger.info(f"Cannot send to {websocket.client}: connection not open")
            except Exception as e:
                logger.error(f"Error sending chunk: {e}: {chunk}")

        # メッセージ受信タスクを開始
        receive_task = asyncio.create_task(receive_messages())

        # CineBotとの接続を開始
        cinebot_task = asyncio.create_task(
            cine_bot.aconnect(input_stream(), send_output_chunk)
        )

        # どちらかのタスクが完了するまで待機
        done, pending = await asyncio.wait(
            [receive_task, cinebot_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # 未完了のタスクをキャンセル
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info(f"Client {websocket.client} disconnected")
    except Exception as e:
        logger.error(f"Error handling client {websocket.client}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client {websocket.client} removed. Total clients: {len(connected_clients)}")


async def health_check(request):
    """ヘルスチェックエンドポイント"""
    return JSONResponse({
        "status": "healthy",
        "service": "CineBot WebSocket Server",
        "connected_clients": len(connected_clients),
        "timestamp": asyncio.get_event_loop().time()
    })


async def homepage(request):
    with open("static/index.html") as f:
        html = f.read()
        return HTMLResponse(html)
    
# Starletteアプリケーションを作成
app = Starlette(
    routes=[
        Route('/', homepage),
        Route('/health', health_check),
        WebSocketRoute('/ws', websocket_endpoint),
    ]
)
app.mount("/", StaticFiles(directory="static"), name="static")

def check_environment():
    """環境設定の確認"""
    logger.info("🔍 環境設定チェック")
    
    # 必要な環境変数をチェック
    required_env_vars = ["OPENAI_API_KEY", "TMDB_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if os.getenv(var):
            logger.info(f"✅ {var}: 設定済み")
        else:
            logger.error(f"❌ {var}: 未設定")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("⚠️ 以下の環境変数を設定してください:")
        for var in missing_vars:
            logger.error(f"   export {var}=your_api_key_here")
        return False
    
    logger.info("✅ 環境設定OK")
    return True


def main():
    """メイン実行関数"""
    # 環境チェック
    if not check_environment():
        logger.error("❌ 環境設定が不完全です。上記の指示に従って設定してください。")
        sys.exit(1)
    
    # コマンドライン引数の処理
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    logger.info(f"🎬 CineBot WebSocket Server starting on http://{host}:{port}")
    logger.info(f"WebSocket endpoint: ws://{host}:{port}/ws")
    logger.info("サーバーを停止するには Ctrl+C を押してください")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("サーバーを停止しています...")
    except Exception as e:
        logger.error(f"サーバーエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
