use crate::message::Message;
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

/// A conversation memory that persists messages through a pluggable backend.
#[derive(Clone, Debug)]
pub struct PersistentConversationMemory<S: ConversationStore> {
    store: S,
    inner: ConversationMemory,
}

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
