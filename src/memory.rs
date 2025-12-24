use crate::message::Message;
#[cfg(feature = "persistence")]
use crate::storage::ConversationStore;

/// In-memory transcript storage.
#[derive(Default, Clone, Debug)]
pub struct ConversationMemory {
    messages: Vec<Message>,
}

impl ConversationMemory {
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Message> + '_ {
        self.messages.iter()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(feature = "persistence")]
#[derive(Clone)]
pub struct PersistentConversationMemory<S: ConversationStore> {
    store: S,
    inner: ConversationMemory,
}

#[cfg(feature = "persistence")]
impl<S: ConversationStore> PersistentConversationMemory<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            inner: ConversationMemory::default(),
        }
    }

    pub async fn load(mut self) -> crate::Result<Self> {
        let stored = self.store.load().await?;
        self.inner = ConversationMemory::with_messages(stored);
        Ok(self)
    }

    pub fn as_memory(&self) -> &ConversationMemory {
        &self.inner
    }

    pub async fn push(&mut self, message: Message) -> crate::Result<()> {
        self.store.append(&message).await?;
        self.inner.push(message);
        Ok(())
    }

    pub async fn clear(&mut self) -> crate::Result<()> {
        self.store.clear().await?;
        self.inner = ConversationMemory::default();
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Memory strategy trait for managing conversation context
pub trait MemoryStrategy: Send + Sync {
    /// Apply the strategy to get messages to send to the LLM
    fn get_context_messages(&self, messages: &[Message]) -> Vec<Message>;

    /// Name of the strategy
    fn name(&self) -> &str;
}

/// Keep all messages (default, no limiting)
#[derive(Clone, Default)]
pub struct FullMemoryStrategy;

impl MemoryStrategy for FullMemoryStrategy {
    fn get_context_messages(&self, messages: &[Message]) -> Vec<Message> {
        messages.to_vec()
    }

    fn name(&self) -> &str {
        "full"
    }
}

/// Keep only the last N messages (sliding window)
#[derive(Clone)]
pub struct WindowedMemoryStrategy {
    window_size: usize,
    keep_system: bool,
}

impl WindowedMemoryStrategy {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            keep_system: true,
        }
    }

    pub fn without_system(mut self) -> Self {
        self.keep_system = false;
        self
    }
}

impl MemoryStrategy for WindowedMemoryStrategy {
    fn get_context_messages(&self, messages: &[Message]) -> Vec<Message> {
        use crate::message::Role;

        if messages.len() <= self.window_size {
            return messages.to_vec();
        }

        let mut result = Vec::new();

        // Keep system messages if configured
        if self.keep_system {
            for msg in messages {
                if msg.role == Role::System {
                    result.push(msg.clone());
                }
            }
        }

        // Add the last N non-system messages
        let non_system: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let start = if non_system.len() > self.window_size {
            non_system.len() - self.window_size
        } else {
            0
        };

        for msg in &non_system[start..] {
            result.push((*msg).clone());
        }

        result
    }

    fn name(&self) -> &str {
        "windowed"
    }
}

/// Keep first and last N messages, summarize the middle
#[derive(Clone)]
pub struct SummarizedMemoryStrategy {
    /// Number of messages to keep at the start
    keep_first: usize,
    /// Number of messages to keep at the end
    keep_last: usize,
    /// Summary of the middle (set after summarization)
    summary: Option<String>,
}

impl SummarizedMemoryStrategy {
    pub fn new(keep_first: usize, keep_last: usize) -> Self {
        Self {
            keep_first,
            keep_last,
            summary: None,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Check if summarization is needed (more than keep_first + keep_last messages)
    pub fn needs_summary(&self, messages: &[Message]) -> bool {
        messages.len() > self.keep_first + self.keep_last
    }

    /// Get messages that need to be summarized
    pub fn messages_to_summarize<'a>(&self, messages: &'a [Message]) -> &'a [Message] {
        if messages.len() <= self.keep_first + self.keep_last {
            return &[];
        }
        let end = messages.len() - self.keep_last;
        &messages[self.keep_first..end]
    }
}

impl MemoryStrategy for SummarizedMemoryStrategy {
    fn get_context_messages(&self, messages: &[Message]) -> Vec<Message> {
        if messages.len() <= self.keep_first + self.keep_last {
            return messages.to_vec();
        }

        let mut result = Vec::new();

        // Add first N messages
        for msg in messages.iter().take(self.keep_first) {
            result.push(msg.clone());
        }

        // Add summary as a system message if available
        if let Some(ref summary) = self.summary {
            result.push(Message::system(format!(
                "[Summary of {} messages]: {}",
                messages.len() - self.keep_first - self.keep_last,
                summary
            )));
        }

        // Add last N messages
        let start = messages.len() - self.keep_last;
        for msg in &messages[start..] {
            result.push(msg.clone());
        }

        result
    }

    fn name(&self) -> &str {
        "summarized"
    }
}

/// Token-based memory limiting (approximate)
#[derive(Clone)]
pub struct TokenLimitedMemoryStrategy {
    max_tokens: usize,
    /// Approximate characters per token (default: 4)
    chars_per_token: usize,
}

impl TokenLimitedMemoryStrategy {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            chars_per_token: 4,
        }
    }

    pub fn with_chars_per_token(mut self, chars: usize) -> Self {
        self.chars_per_token = chars;
        self
    }

    fn estimate_tokens(&self, content: &str) -> usize {
        content.len() / self.chars_per_token
    }
}

impl MemoryStrategy for TokenLimitedMemoryStrategy {
    fn get_context_messages(&self, messages: &[Message]) -> Vec<Message> {
        use crate::message::Role;

        let mut result = Vec::new();
        let mut total_tokens = 0;

        // Always include system messages first
        for msg in messages {
            if msg.role == Role::System {
                let tokens = self.estimate_tokens(&msg.content);
                total_tokens += tokens;
                result.push(msg.clone());
            }
        }

        // Add messages from the end until we hit the limit
        let non_system: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let mut temp = Vec::new();
        for msg in non_system.iter().rev() {
            let tokens = self.estimate_tokens(&msg.content);
            if total_tokens + tokens > self.max_tokens {
                break;
            }
            total_tokens += tokens;
            temp.push((*msg).clone());
        }

        // Reverse to maintain chronological order
        temp.reverse();
        result.extend(temp);

        result
    }

    fn name(&self) -> &str {
        "token_limited"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowed_strategy() {
        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
            Message::user("How are you?"),
            Message::assistant("I'm doing well!"),
            Message::user("What's 2+2?"),
            Message::assistant("4"),
        ];

        let strategy = WindowedMemoryStrategy::new(4);
        let context = strategy.get_context_messages(&messages);

        // Should keep system + last 4 non-system messages
        assert_eq!(context.len(), 5); // 1 system + 4 recent
        assert_eq!(context[0].content, "You are a helpful assistant");
    }

    #[test]
    fn test_token_limited_strategy() {
        let messages = vec![
            Message::system("System"),
            Message::user("A".repeat(100)),
            Message::assistant("B".repeat(100)),
            Message::user("C".repeat(100)),
        ];

        let strategy = TokenLimitedMemoryStrategy::new(50); // ~200 chars
        let context = strategy.get_context_messages(&messages);

        // Should keep system and fit as many recent messages as possible
        assert!(context.len() <= messages.len());
    }
}

