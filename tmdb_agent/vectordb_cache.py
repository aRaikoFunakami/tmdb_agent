import os
import json
import time
import unicodedata
from typing import Any, Dict, Optional, List
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import hashlib

# パラメータ
DEFAULT_TAU = 0.90
FRESHNESS_WEIGHT = 0.99  # 時間減衰の重み
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_cache")
MODEL_NAME = "all-MiniLM-L6-v2"

# 正規化関数
def normalize_text(text: str) -> str:
    # 小文字化、全角半角統一、空白・記号除去
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = ' '.join(text.split())
    return text


def param_hash(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

class VectorDBCache:

    def search_with_score(self, query: str, meta: Dict[str, Any], now: Optional[int] = None):
        """
        キャッシュヒット時は (value, True, score)、ヒットしなければ (None, False, best_score) を返す
        """
        norm_query = normalize_text(query)
        now = now or int(time.time())
        # メタデータによるフィルタは行わず、全件から類似検索
        results = self.collection.query(
            query_texts=[norm_query],
            n_results=5
        )
        best_score = 0
        best_id = None
        for i, score in enumerate(results.get("distances", [[1]])[0]):
            cos_sim = 1 - score
            if cos_sim > best_score:
                best_score = cos_sim
                best_id = results["ids"][0][i]
        if best_id and best_score >= self.tau:
            value_path = os.path.join(self.persist_dir, f"{best_id}.json")
            if os.path.exists(value_path):
                with open(value_path, "r", encoding="utf-8") as f:
                    return json.load(f), True, best_score
        return None, False, best_score
    
    def __init__(self, tau: float = DEFAULT_TAU, persist_dir: Optional[str] = None):
        self.tau = tau
        self.persist_dir = persist_dir or CHROMA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="location_search_cache",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        )

    def _make_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        # 必須キー: locale, region, user, provider, version, param_hash
        keys = ["locale", "region", "user", "provider", "version", "param_hash"]
        return {k: meta.get(k, "") for k in keys}

    def add(self, query: str, meta: Dict[str, Any], value: Any, ttl: int = 3600):
        norm_query = normalize_text(query)
        meta = self._make_meta(meta)
        now = int(time.time())
        doc_id = f"{meta['param_hash']}_{now}"
        self.collection.add(
            documents=[norm_query],
            metadatas=[{
                **meta,
                "saved_at": now,
                "ttl": ttl
            }],
            ids=[doc_id],
            embeddings=[self.model.encode(norm_query)]
        )
        # valueは別ファイルに保存
        with open(os.path.join(self.persist_dir, f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    def search(self, query: str, meta: Dict[str, Any], now: Optional[int] = None) -> Optional[Any]:
        norm_query = normalize_text(query)
        now = now or int(time.time())
        # メタデータによるフィルタは行わず、全件から類似検索
        results = self.collection.query(
            query_texts=[norm_query],
            n_results=5
        )
        # スコア計算: cos_sim * freshness_weight
        best_score = 0
        best_id = None
        for i, score in enumerate(results.get("distances", [[1]])[0]):
            # Chromaは距離（小さいほど近い）なのでcos_sim = 1 - score
            cos_sim = 1 - score
            if cos_sim > best_score and cos_sim >= self.tau:
                best_score = cos_sim
                best_id = results["ids"][0][i]
        if best_id:
            # valueをロード
            value_path = os.path.join(self.persist_dir, f"{best_id}.json")
            if os.path.exists(value_path):
                with open(value_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

# テスト用
if __name__ == "__main__":
    cache = VectorDBCache()
    meta = {}
    query = "東京都の観光地を教えて"
    value = {"result": "東京タワー, 浅草寺, 上野公園"}
    cache.add(query, meta, value)



    # search_with_scoreのテスト（tau=0.90）
    print("\n--- search_with_score test (tau=0.90) ---")
    val, hit, score = cache.search_with_score("東京都の観光スポットを教えて", meta)
    print(f"Hit: {hit}, Score: {score:.4f}, Value: {val}")

    # 正規化の効き目テスト（全角・半角・大文字小文字・記号違い）
    print("\n--- normalization test ---")
    test_queries = [
        "東京都の観光スポットを教えて",
        "東京都ノ観光スポットヲ教エテ",  # カタカナ混じり
        "  東京都の観光スポットを教えて  ",  # 前後空白
        "東京都の観光スポットを教えて!",
        "ＴＯＫＹＯ都の観光スポットを教えて",  # 全角英字
        "tokyo都の観光スポットを教えて",  # 半角英字
    ]
    for q in test_queries:
        v, h, s = cache.search_with_score(q, meta)
        print(f"Query: '{q}' -> Hit: {h}, Score: {s:.4f}")

    # メタ情報違い（userだけ変える）
    print("\n--- meta info mismatch test ---")
    meta2 = meta.copy()
    meta2["user"] = "otheruser"
    v2, h2, s2 = cache.search_with_score("東京都の観光スポットを教えて", meta2)
    print(f"Meta user changed -> Hit: {h2}, Score: {s2:.4f}")

    # 完全に異なるクエリ
    print("\n--- completely different query test ---")
    v3, h3, s3 = cache.search_with_score("大阪のグルメを教えて", meta)
    print(f"Different query -> Hit: {h3}, Score: {s3:.4f}")

    # search_with_scoreのテスト
    print("\n--- search_with_score test (tau=0.90) ---")
    val, hit, score = cache.search_with_score("東京都の観光スポットを教えて", meta)
    print(f"Hit: {hit}, Score: {score:.4f}, Value: {val}")

    # しきい値を高くしてヒットしないこともテスト
    cache_high_tau = VectorDBCache(tau=0.99)
    cache_high_tau.add(query, meta, value)
    val2, hit2, score2 = cache_high_tau.search_with_score("東京都の観光スポットを教えて", meta)
    print(f"Hit with tau=0.99: {hit2}, Score: {score2:.4f}, Value: {val2}")
