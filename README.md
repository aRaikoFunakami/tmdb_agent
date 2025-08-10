# TMDB Agent

TMDB Agentは、[TMDB API](https://www.themoviedb.org/documentation/api)を利用して映画、TV番組、人物の情報を検索するための多機能エージェントです。このエージェントは多言語対応で、映画のクレジット情報、トレンド情報、制作会社による映画検索など、さまざまな機能を提供します。

## 主な機能

- 映画、TV番組、人物の検索
- 制作会社の検索とその映画一覧の取得
- トレンド情報の取得（映画、TV番組、人物）
- 主題歌や楽曲情報の検索
- 多言語対応（日本語、英語など）

## 必要条件

- Python 3.11以上
- [uv](https://github.com/tonybaloney/uv) パッケージマネージャー
- 以下の環境変数を設定してください：
  - `TMDB_API_KEY`: TMDB APIを利用するためのAPIキー。
  - `TAVILY_API_KEY`: TAVILY関連のAPIキー。
  - `TMDB_API_LANG`: TMDB APIの言語設定（例: `ja-JP`）。
  - `OPENAI_API_KEY`: OpenAIのAPIキー。

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/aRaikoFunakami/tmdb_agent.git
cd tmdb_agent
```

### 2. 必要なパッケージのインストール

`uv` を使用して依存関係を管理します。

#### `uv` のインストール

`uv` がインストールされていない場合、以下のコマンドでインストールしてください。

```bash
pip install uv
```

#### 依存関係のインストール

以下のコマンドで必要なパッケージをインストールします。

```bash
uv sync
```

### 3. 環境変数の設定

以下の環境変数を設定してください。

#### TMDB APIキー

```bash
export TMDB_API_KEY="your_tmdb_api_key"
```

#### TAVILY APIキー

```bash
export TAVILY_API_KEY="your_tavily_api_key"
```

#### TMDB APIの言語設定（任意）

```bash
export TMDB_API_LANG="ja-JP"
```

#### OpenAI APIキー

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## 実行方法

### 自動テストの実行

`main.py` には自動テスト機能が組み込まれています。以下のコマンドでテストを実行できます。

```bash
uv run python -m tmdb_agent.main --auto
```

### チャットモードの実行

インタラクティブなチャット形式でエージェントを利用するには、以下のコマンドを実行します。

```bash
uv run python -m tmdb_agent.main --chat
```

#### サンプル入力

以下は、チャットモードでのサンプル入力例です：

```text
🎬 あなた:  最近の人気映画を教えて
🤖 AI: 最近の人気映画トップ10は以下の通りです。

1. ジュラシック・ワールド／復活の大地
2. Weapons
3. ザ・ピックアップ ~恋の強盗大作戦~
4. 28年後...
5. サンダーボルツ*
6. ヒックとドラゴン
7. トリプルX
8. 哪吒之魔童闹海
9. The Occupant
10. ファンタスティック４：ファースト・ステップ

これらは2025年8月10日時点での人気映画です。
```

```text
🎬 あなた: スターウォーズ１の詳細を教えて
🤖 AI: 「スター・ウォーズ エピソード１／ファントム・メナス」（1999年5月19日公開）は、ジョージ・ルーカス監督のSF映画です。物語は銀河共和国と通商連合の対立を描き、ジェダイ騎士クワイ＝ガン・ジン（リーアム・ニーソン）と弟子オビ＝ワン・ケノービ（ユアン・マクレガー）が惑星ナブーの危機に立ち向かいます。ナタリー・ポートマンがパドメ・アミダラ役、ジェイク・ロイドがアナキン・スカイウォーカー役を務めています。評価は6.565/10です。
```

## プロジェクト構成

```text
tmdb_agent/
├── tmdb_agent/
│   ├── agent.py          # TMDBエージェントのメインロジック
│   ├── main.py           # エントリーポイント
│   ├── tools.py          # TMDB APIを利用するツール群
│   └── __init__.py       # パッケージ初期化
├── auto_test.sh          # 自動テストスクリプト
├── pyproject.toml        # プロジェクト設定ファイル
├── README.md             # このファイル
└── uv.lock               # uvによる依存関係ロックファイル
```

## 開発者向け情報

### 新しい依存関係の追加

新しいパッケージを追加する場合、以下のコマンドを使用してください。

```bash
uv add <パッケージ名>
```

### 依存関係の更新

依存関係を更新するには、以下のコマンドを使用します。

```bash
uv sync
```

### テストの実行

`auto_test.sh` を使用して自動テストを実行できます。

```bash
bash auto_test.sh
```

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。
