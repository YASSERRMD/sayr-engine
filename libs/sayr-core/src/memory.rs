use crate::message::Message;

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

    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}
