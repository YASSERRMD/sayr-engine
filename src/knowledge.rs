use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::error::Result;

#[derive(Clone, Debug)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: Value,
}

#[derive(Clone, Debug)]
pub struct ScoredDocument {
    pub document: Document,
    pub score: f32,
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

#[async_trait]
pub trait OpenAiEmbeddingClient: Send + Sync {
    async fn embed(&self, model: &str, input: &str) -> Result<Vec<f32>>;
}

/// Embedder that delegates to an OpenAI-compatible embedding client.
pub struct OpenAiEmbedder<C> {
    client: Arc<C>,
    model: String,
}

impl<C> OpenAiEmbedder<C> {
    pub fn new(client: Arc<C>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

#[async_trait]
impl<C> Embedder for OpenAiEmbedder<C>
where
    C: OpenAiEmbeddingClient,
{
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.client.embed(&self.model, text).await
    }
}

#[async_trait]
pub trait TransformerClient: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Embedder that wraps a transformer runtime (e.g., candle, ort, ggml).
pub struct TransformerEmbedder<C> {
    client: Arc<C>,
}

impl<C> TransformerEmbedder<C> {
    pub fn new(client: Arc<C>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl<C> Embedder for TransformerEmbedder<C>
where
    C: TransformerClient,
{
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.client.embed(text).await
    }
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add(&self, document: Document, embedding: Vec<f32>) -> Result<()>;
    async fn search(
        &self,
        embedding: Vec<f32>,
        params: SearchParams,
    ) -> Result<Vec<ScoredDocument>>;
}

/// Basic whitespace tokenizer with hashed buckets for deterministic embeddings.
pub struct WhitespaceEmbedder {
    buckets: usize,
}

impl Default for WhitespaceEmbedder {
    fn default() -> Self {
        Self { buckets: 32 }
    }
}

impl WhitespaceEmbedder {
    pub fn new(buckets: usize) -> Self {
        Self { buckets }
    }
}

#[async_trait]
impl Embedder for WhitespaceEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.buckets];

        for token in text.split_whitespace() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            token.hash(&mut hasher);
            let idx = (hasher.finish() as usize) % self.buckets;
            vector[idx] += 1.0;
        }

        Ok(vector)
    }
}

#[derive(Default)]
pub struct InMemoryVectorStore {
    entries: RwLock<Vec<(Document, Vec<f32>)>>,
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&self, document: Document, embedding: Vec<f32>) -> Result<()> {
        self.entries.write().await.push((document, embedding));
        Ok(())
    }

    async fn search(
        &self,
        embedding: Vec<f32>,
        params: SearchParams,
    ) -> Result<Vec<ScoredDocument>> {
        let entries = self.entries.read().await;
        let mut scored: Vec<ScoredDocument> = entries
            .iter()
            .map(|(doc, stored)| ScoredDocument {
                document: doc.clone(),
                score: similarity(stored, &embedding, params.similarity),
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(params.top_k);
        Ok(scored)
    }
}

fn similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    match metric {
        SimilarityMetric::Cosine => {
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a.sqrt() * norm_b.sqrt())
            }
        }
        SimilarityMetric::DotProduct => dot,
        SimilarityMetric::Euclidean => {
            // Invert distance so higher is better while keeping the return type consistent.
            let mut squared_distance = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let diff = x - y;
                squared_distance += diff * diff;
            }
            1.0 / (1.0 + squared_distance.sqrt())
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SimilarityMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

#[derive(Clone, Debug)]
pub struct SearchParams {
    pub top_k: usize,
    pub similarity: SimilarityMetric,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            top_k: 5,
            similarity: SimilarityMetric::Cosine,
        }
    }
}

#[async_trait]
pub trait PgVectorClient: Send + Sync {
    async fn upsert(&self, document: &Document, embedding: &[f32]) -> Result<()>;
    async fn query(&self, embedding: &[f32], params: SearchParams) -> Result<Vec<ScoredDocument>>;
}

/// Adapter for Postgres/pgvector style databases.
pub struct PgVectorStore<C> {
    client: Arc<C>,
}

impl<C> PgVectorStore<C> {
    pub fn new(client: Arc<C>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl<C> VectorStore for PgVectorStore<C>
where
    C: PgVectorClient,
{
    async fn add(&self, document: Document, embedding: Vec<f32>) -> Result<()> {
        self.client.upsert(&document, &embedding).await
    }

    async fn search(
        &self,
        embedding: Vec<f32>,
        params: SearchParams,
    ) -> Result<Vec<ScoredDocument>> {
        self.client.query(&embedding, params).await
    }
}

#[async_trait]
pub trait QdrantClient: Send + Sync {
    async fn upsert(&self, document: &Document, embedding: &[f32]) -> Result<()>;
    async fn query(&self, embedding: &[f32], params: SearchParams) -> Result<Vec<ScoredDocument>>;
}

/// Adapter for Qdrant (or other HTTP/gRPC vector databases).
pub struct QdrantStore<C> {
    client: Arc<C>,
}

impl<C> QdrantStore<C> {
    pub fn new(client: Arc<C>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl<C> VectorStore for QdrantStore<C>
where
    C: QdrantClient,
{
    async fn add(&self, document: Document, embedding: Vec<f32>) -> Result<()> {
        self.client.upsert(&document, &embedding).await
    }

    async fn search(
        &self,
        embedding: Vec<f32>,
        params: SearchParams,
    ) -> Result<Vec<ScoredDocument>> {
        self.client.query(&embedding, params).await
    }
}

pub trait DocumentChunker: Send + Sync {
    fn chunk(&self, document: &Document) -> Vec<Document>;
}

/// Token (word) based chunker with sliding window overlap.
pub struct SlidingWindowChunker {
    pub max_tokens: usize,
    pub overlap: usize,
}

impl Default for SlidingWindowChunker {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            overlap: 32,
        }
    }
}

impl DocumentChunker for SlidingWindowChunker {
    fn chunk(&self, document: &Document) -> Vec<Document> {
        if document.text.is_empty() {
            return vec![document.clone()];
        }

        let tokens: Vec<&str> = document.text.split_whitespace().collect();
        if tokens.len() <= self.max_tokens {
            return vec![document.clone()];
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut chunk_index = 0usize;

        while start < tokens.len() {
            let end = usize::min(start + self.max_tokens, tokens.len());
            let text = tokens[start..end].join(" ");
            let mut metadata = document.metadata.clone();

            if let Value::Object(map) = &mut metadata {
                map.insert("chunk_index".to_string(), Value::from(chunk_index as u64));
                map.insert("source_id".to_string(), Value::from(document.id.clone()));
            } else {
                metadata = json!({
                    "chunk_index": chunk_index,
                    "source_id": document.id
                });
            }

            chunks.push(Document {
                id: format!("{}::{}", document.id, chunk_index),
                text,
                metadata,
            });

            if end == tokens.len() {
                break;
            }

            start = end.saturating_sub(self.overlap.min(end - start));
            chunk_index += 1;
        }

        chunks
    }
}

pub type Reranker = Arc<dyn Fn(&ScoredDocument) -> f32 + Send + Sync>;

pub struct KnowledgeBase<E: Embedder, S: VectorStore> {
    embedder: Arc<E>,
    store: Arc<S>,
    config: RetrievalConfig,
    chunker: Option<Arc<dyn DocumentChunker>>,
}

impl<E: Embedder, S: VectorStore> KnowledgeBase<E, S> {
    pub fn new(embedder: Arc<E>, store: Arc<S>) -> Self {
        Self {
            embedder,
            store,
            config: RetrievalConfig::default(),
            chunker: None,
        }
    }

    pub fn with_reranker(mut self, reranker: Reranker) -> Self {
        self.config.reranker = Some(reranker);
        self
    }

    pub fn with_chunker(mut self, chunker: Arc<dyn DocumentChunker>) -> Self {
        self.chunker = Some(chunker);
        self
    }

    pub fn with_config(mut self, config: RetrievalConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub async fn add_document(&self, document: Document) -> Result<()> {
        let chunks = if let Some(chunker) = &self.chunker {
            chunker.chunk(&document)
        } else {
            vec![document]
        };

        for chunk in chunks {
            let embedding = self.embedder.embed(&chunk.text).await?;
            self.store.add(chunk, embedding).await?;
        }

        Ok(())
    }

    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ScoredDocument>> {
        let overrides = RetrievalOverrides {
            top_k: Some(top_k),
            ..Default::default()
        };
        self.retrieve_with_overrides(query, overrides).await
    }

    pub async fn retrieve_with_overrides(
        &self,
        query: &str,
        overrides: RetrievalOverrides,
    ) -> Result<Vec<ScoredDocument>> {
        let embedding = self.embedder.embed(query).await?;
        let params = SearchParams {
            top_k: overrides.top_k.unwrap_or(self.config.top_k),
            similarity: overrides.similarity.unwrap_or(self.config.similarity),
        };
        let mut scored = self.store.search(embedding, params).await?;

        if let Some(reranker) = overrides.reranker.or_else(|| self.config.reranker.clone()) {
            for doc in scored.iter_mut() {
                doc.score = reranker(doc);
            }
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(scored)
    }

    pub async fn evaluate(
        &self,
        query: &str,
        relevant_document_ids: &[String],
        overrides: RetrievalOverrides,
    ) -> Result<RetrievalEvaluation> {
        let retrieved = self.retrieve_with_overrides(query, overrides).await?;
        let retrieved_ids: HashSet<String> =
            retrieved.iter().map(|d| d.document.id.clone()).collect();
        let relevant: HashSet<String> = relevant_document_ids.iter().cloned().collect();

        let hits = relevant.intersection(&retrieved_ids).count() as f32;
        let precision = if retrieved.is_empty() {
            0.0
        } else {
            hits / retrieved.len() as f32
        };
        let recall = if relevant.is_empty() {
            0.0
        } else {
            hits / relevant.len() as f32
        };

        Ok(RetrievalEvaluation {
            retrieved,
            precision,
            recall,
        })
    }
}

#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<String>>;
}

#[async_trait]
impl<E, S> Retriever for KnowledgeBase<E, S>
where
    E: Embedder,
    S: VectorStore,
{
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let docs = KnowledgeBase::retrieve(self, query, top_k).await?;
        Ok(docs.into_iter().map(|d| d.document.text).collect())
    }
}

#[derive(Clone)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub similarity: SimilarityMetric,
    pub reranker: Option<Reranker>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            similarity: SimilarityMetric::Cosine,
            reranker: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct RetrievalOverrides {
    pub top_k: Option<usize>,
    pub similarity: Option<SimilarityMetric>,
    pub reranker: Option<Reranker>,
}

pub struct RetrievalEvaluation {
    pub retrieved: Vec<ScoredDocument>,
    pub precision: f32,
    pub recall: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEmbedder;

    #[async_trait]
    impl Embedder for TestEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32])
        }
    }

    #[tokio::test]
    async fn chunks_documents() {
        let embedder = Arc::new(TestEmbedder);
        let store = Arc::new(InMemoryVectorStore::default());
        let kb: KnowledgeBase<_, _> =
            KnowledgeBase::new(embedder, store).with_chunker(Arc::new(SlidingWindowChunker {
                max_tokens: 2,
                overlap: 0,
            }));

        kb.add_document(Document {
            id: "doc".into(),
            text: "a b c d".into(),
            metadata: Value::Null,
        })
        .await
        .unwrap();

        let scored = kb.retrieve("a b", 10).await.unwrap();
        assert_eq!(scored.len(), 2);
    }

    #[tokio::test]
    async fn evaluates_precision_recall() {
        let embedder = Arc::new(TestEmbedder);
        let store = Arc::new(InMemoryVectorStore::default());
        let kb: KnowledgeBase<_, _> = KnowledgeBase::new(embedder, store);

        kb.add_document(Document {
            id: "d1".into(),
            text: "hello world".into(),
            metadata: Value::Null,
        })
        .await
        .unwrap();
        kb.add_document(Document {
            id: "d2".into(),
            text: "other".into(),
            metadata: Value::Null,
        })
        .await
        .unwrap();

        let report = kb
            .evaluate(
                "hello",
                &[String::from("d1")],
                RetrievalOverrides {
                    top_k: Some(1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(report.recall, 1.0);
        assert_eq!(report.precision, 1.0);
    }
}
