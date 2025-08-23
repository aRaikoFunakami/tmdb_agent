# CineBot - 音声対応映画・TV番組レコメンデーションエージェント

CineBotは、OpenAI Realtime APIを使用した音声対応の映画・TV番組レコメンデーションAIエージェントです。TMDB APIと連携して、ユーザーの好みに基づいて最適な作品を提案します。

## 特徴

- 🎙️ **音声対応**: OpenAI Realtime APIによる自然な音声会話
- 🎬 **映画・TV番組検索**: TMDB APIを活用した豊富なコンテンツデータベース
- 🤖 **インテリジェントレコメンデーション**: ユーザーの好みに基づく個人化された提案
- 🌐 **多言語対応**: 日本語・英語等での自然な会話
- 📱 **リアルタイム**: WebSocketによるストリーミング会話

## セットアップ

### 1. 環境要件

- Python 3.11以上
- OpenAI API キー
- TMDB API キー

### 2. インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/tmdb_agent.git
cd tmdb_agent

# 依存関係をインストール
pip install -r requirements.txt

# または uvを使用
uv sync
```

### 3. 環境変数の設定

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export TMDB_API_KEY="your_tmdb_api_key_here"
```

#### APIキーの取得方法

**OpenAI API キー:**
1. [OpenAI Platform](https://platform.openai.com/)にアクセス
2. アカウントを作成/ログイン
3. API Keys セクションで新しいキーを生成

**TMDB API キー:**
1. [TMDB](https://www.themoviedb.org/)でアカウント作成
2. [API Settings](https://www.themoviedb.org/settings/api)で申請
3. API Keyを取得

## 使用方法

### 1. 基本的なテスト

```bash
# レコメンデーション機能のテスト
python demo_cine_bot.py
```

### 2. WebSocketサーバーの起動

```bash
# サーバーを起動 (Starlette + Uvicorn版)
python cine_bot_server.py

# カスタムホストとポートで起動
python cine_bot_server.py 0.0.0.0 8080

# サーバー起動後、ブラウザで以下にアクセス:
# http://localhost:8000 (テストページ)
# http://localhost:8000/health (ヘルスチェック)
```

### 3. WebSocketクライアントでのテスト

```bash
# 対話モードでクライアントを起動
python cine_bot_client.py

# テストモードで自動テスト実行
python cine_bot_client.py ws://localhost:8000/ws test
```

### 3. プログラムでの使用

```python
import asyncio
from tmdb_agent.cine_bot import create_cine_bot

async def main():
    # CineBotを作成
    bot = create_cine_bot(verbose=True)
    
    # 入力ストリームを模擬
    async def input_stream():
        yield '{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "80年代で面白い映画ある？"}]}}'
    
    # 出力処理
    async def handle_output(chunk: str):
        print(f"出力: {chunk}")
    
    # エージェントに接続
    await bot.aconnect(input_stream(), handle_output)

# 実行
asyncio.run(main())
```

## 対話例

### 基本的な質問

**ユーザー**: "80年代で面白い映画ある？"

**CineBot**: "80年代の名作映画をいくつかご紹介しますね！まず「バック・トゥ・ザ・フューチャー」は絶対におすすめです。マイケル・J・フォックス演じるマーティが、偶然タイムマシンのデロリアンで1955年にタイムスリップしてしまい..."

### レコメンデーション

**ユーザー**: "ナウシカ好きなんだけど、おすすめの映画ある？"

**CineBot**: "ナウシカがお好きでしたら、同じ宮崎駿監督の「もののけ姫」はいかがでしょうか？自然と人間の関係を深く描いた作品で、ナウシカと同様に強い女性主人公が活躍します..."

### 最新トレンド

**ユーザー**: "最新のトレンド映画教えて"

**CineBot**: "現在人気の映画をご紹介しますね！今週のトレンドでは..."

## アーキテクチャ

```
CineBot (Starlette + Uvicorn版)
├── tmdb_agent/
│   ├── agent.py          # 基本TMDBエージェント
│   ├── tools.py          # TMDB API連携ツール
│   └── cine_bot.py       # 音声対応レコメンデーションエージェント
├── langchain_openai_voice/
│   ├── __init__.py       # OpenAI Voice React Agent
│   └── utils.py          # ユーティリティ関数
├── demo_cine_bot.py      # デモスクリプト
├── cine_bot_server.py    # Starlette WebSocketサーバー
└── cine_bot_client.py    # WebSocketクライアント
```

## 主要コンポーネント

### CineBot クラス

- OpenAI Realtime APIとの統合
- 音声入力/出力の処理
- TMDB ツールとの連携

### レコメンデーション機能

- ユーザーの好みに基づく検索戦略
- 年代別、ジャンル別、テーマ別検索
- 最新トレンド情報の提供

### TMDB ツール群

- 映画・TV番組・人物検索
- トレンド情報取得
- 詳細情報の提供

### Starlette WebSocketサーバー

- 高性能なASGIベースWebSocketサーバー
- 複数クライアント同時接続対応
- ヘルスチェックとテストページ提供
- Uvicornによる本番運用対応

## 開発・カスタマイズ

### カスタムインストラクションの追加

```python
custom_instructions = """
あなたは特にアクション映画を重視して
レコメンドしてください。
"""

bot = create_cine_bot(instructions=custom_instructions)
```

### 新しいツールの追加

```python
from langchain_core.tools import tool

@tool
def custom_recommendation_tool(query: str) -> str:
    """カスタムレコメンデーション機能"""
    # カスタムロジックを実装
    return "カスタム推薦結果"

# ツールリストに追加
bot.tools.append(custom_recommendation_tool)
```

## トラブルシューティング

### 一般的な問題

1. **API キーエラー**
   ```
   export OPENAI_API_KEY="your_key_here"
   export TMDB_API_KEY="your_key_here"
   ```

2. **websockets インストールエラー**
   ```bash
   pip install websockets
   ```

3. **Python バージョンエラー**
   - Python 3.11以上が必要です

### ログの確認

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ライセンス

MIT License

## 貢献

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## サポート

- Issues: [GitHub Issues](https://github.com/your-username/tmdb_agent/issues)
- Discussions: [GitHub Discussions](https://github.com/your-username/tmdb_agent/discussions)
