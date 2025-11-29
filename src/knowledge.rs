use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
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
pub trait VectorStore: Send + Sync {
    async fn add(&self, document: Document, embedding: Vec<f32>) -> Result<()>;
    async fn search(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>>;
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

    async fn search(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>> {
        let entries = self.entries.read().await;
        let mut scored: Vec<ScoredDocument> = entries
            .iter()
            .map(|(doc, stored)| ScoredDocument {
                document: doc.clone(),
                score: cosine_similarity(stored, &embedding),
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

pub type Reranker = Arc<dyn Fn(&ScoredDocument) -> f32 + Send + Sync>;

pub struct KnowledgeBase<E: Embedder, S: VectorStore> {
    embedder: Arc<E>,
    store: Arc<S>,
    reranker: Option<Reranker>,
}

impl<E: Embedder, S: VectorStore> KnowledgeBase<E, S> {
    pub fn new(embedder: Arc<E>, store: Arc<S>) -> Self {
        Self {
            embedder,
            store,
            reranker: None,
        }
    }

    pub fn with_reranker(mut self, reranker: Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub async fn add_document(&self, document: Document) -> Result<()> {
        let embedding = self.embedder.embed(&document.text).await?;
        self.store.add(document, embedding).await
    }

    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ScoredDocument>> {
        let embedding = self.embedder.embed(query).await?;
        let mut scored = self.store.search(embedding, top_k).await?;

        if let Some(reranker) = &self.reranker {
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
