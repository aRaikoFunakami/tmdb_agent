#!/usr/bin/env bash
set -euo pipefail

# デフォルト値
DEFAULT_PARALLEL=15
DEFAULT_OUTDIR="/tmp/tmdb_agent.main.$(date +%Y%m%d-%H%M%S)"
DEFAULT_TEST_RANGE="1-15"

# ヘルプメッセージ
function show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -p, --parallel <number>    並列実行数 (デフォルト: $DEFAULT_PARALLEL)"
  echo "  -o, --outdir <directory>   出力ディレクトリ (デフォルト: $DEFAULT_OUTDIR)"
  echo "  -r, --range <range>        テスト番号の範囲 (例: 1-5, デフォルト: $DEFAULT_TEST_RANGE)"
  echo "  -h, --help                 このヘルプメッセージを表示"
}

# 引数のパース
PARALLEL=$DEFAULT_PARALLEL
OUTDIR=$DEFAULT_OUTDIR
TEST_RANGE=$DEFAULT_TEST_RANGE

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--parallel)
      PARALLEL="$2"
      shift 2
      ;;
    -o|--outdir)
      OUTDIR="$2"
      shift 2
      ;;
    -r|--range)
      TEST_RANGE="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# 出力ディレクトリの作成
mkdir -p "$OUTDIR"

# テスト番号の範囲を解析
IFS="-" read -r START_TEST END_TEST <<< "$TEST_RANGE"

# 並列実行
pids=()
for i in $(seq "$START_TEST" "$END_TEST"); do
  echo "=== Running test $i ==="
  uv run python -m tmdb_agent.main --auto "$i" > "$OUTDIR/test_${i}.log" 2>&1 &
  pids+=($!)

  # 実行中のジョブ数が上限に達したら全部待つ
  if (( ${#pids[@]} >= PARALLEL )); then
    wait "${pids[@]}"
    pids=()
  fi
done

# 残りのジョブも待つ
if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

# 統合ログ作成
MERGED_LOG="$OUTDIR/all_tests.log"
cat > "$MERGED_LOG" <<'PROMPT'
# ChatGPTへの依頼（このファイルを解析してください）

以下は自動実行したTMDBエージェントのテスト統合ログです。  
**やってほしいこと**：
1. テスト番号ごとに結果を**表形式**でまとめてください（列例：テスト番号／テスト名（判別できれば）／ステータス（成功/失敗/要確認）／主な事象・不具合内容／重大度（高/中/低）／根拠ログ抜粋）。
2. 次に、**不具合があったテストのみ**を抽出した短い表を作ってください。
3. 代表的な不具合について、**原因の仮説と具体的な修正案**を箇条書きで提示してください。
4. 期間指定の不整合（例：TMDBのday/week取得と回答文言「昨日/一昨日」の食い違い）、APIバリデーション（例：pageの型エラー）、事実誤り（例：主題歌と歌手の対応）などに特に注意してください。
5. ログの**該当箇所は引用**（短く行番号や前後の文脈が分かる形）し、根拠を明示してください。

出力は**日本語**で、最初に総括（1〜3行）→表→抽出表→改善提案の順でお願いします。

--- 以下、生ログ ---
PROMPT

for i in $(seq "$START_TEST" "$END_TEST"); do
  {
    echo ""
    echo "===== Test $i start ====="
    cat "$OUTDIR/test_${i}.log"
    echo "===== Test $i end ====="
  } >> "$MERGED_LOG"
done

echo "全テスト完了。統合ログファイル: $MERGED_LOG"