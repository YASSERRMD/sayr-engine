use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Role {
    Admin,
    User,
    Service,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Action {
    SendMessage,
    CallTool(String),
    ReadTranscript,
    ManageDeployment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Principal {
    pub id: String,
    pub role: Role,
    pub tenant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrivacyRule {
    pub field: String,
    pub redaction: String,
}

#[derive(Default, Clone)]
pub struct AccessController {
    rules: Arc<RwLock<HashMap<Role, HashSet<Action>>>>,
    privacy: Arc<RwLock<Vec<PrivacyRule>>>,
}

impl AccessController {
    pub fn new() -> Self {
        let controller = Self::default();
        controller.allow(Role::Admin, Action::ManageDeployment);
        controller.allow(Role::Admin, Action::ReadTranscript);
        controller
    }

    pub fn allow(&self, role: Role, action: Action) {
        let mut rules = self.rules.write().unwrap();
        rules.entry(role).or_default().insert(action);
    }

    pub fn authorize(&self, principal: &Principal, action: &Action) -> bool {
        let rules = self.rules.read().unwrap();
        if let Some(actions) = rules.get(&principal.role) {
            actions.contains(action)
        } else {
            false
        }
    }

    pub fn add_privacy_rule(&mut self, rule: PrivacyRule) {
        self.privacy.write().unwrap().push(rule);
    }

    pub fn scrub_payload(&self, payload: &mut serde_json::Value) {
        let rules = self.privacy.read().unwrap();
        for rule in rules.iter() {
            if let Some(obj) = payload.as_object_mut() {
                if obj.contains_key(&rule.field) {
                    obj.insert(
                        rule.field.clone(),
                        serde_json::Value::String(rule.redaction.clone()),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denies_missing_action() {
        let controller = AccessController::new();
        let user = Principal {
            id: "user1".into(),
            role: Role::User,
            tenant: None,
        };
        assert!(!controller.authorize(&user, &Action::ManageDeployment));
    }

    #[test]
    fn scrubs_fields() {
        let mut controller = AccessController::new();
        controller.add_privacy_rule(PrivacyRule {
            field: "secret".into(),
            redaction: "***".into(),
        });
        let mut payload = serde_json::json!({"secret": "value", "other": "ok"});
        controller.scrub_payload(&mut payload);
        assert_eq!(payload["secret"], "***");
    }
}
