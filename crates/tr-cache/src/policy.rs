//! Per-role policies: what happens to chunks (and the snapshots at their ends) of each role.
use crate::role::Role;
use serde::{Deserialize, Serialize};

/// The policy for one role. Defaults: persisted, priority 5, no expiry, snapshots taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolePolicy {
    /// Store chunks of this role at all. A chunk that is not stored breaks the chain, so nothing
    /// *after* it is restorable either; `persist: false` on `reasoning` means prefixes are
    /// resumable up to the first think block only.
    pub persist: bool,
    /// Eviction order, 0..=9: everything of a lower priority is evicted (least recently used
    /// first) before anything of a higher one.
    pub priority: u8,
    /// Idle expiry in seconds: an object unused for longer is dropped regardless of the budget.
    pub ttl: Option<u64>,
    /// Take a snapshot where a prompt span of this role ends (with the `message` policy).
    pub snapshot: bool,
}

pub const PRIORITY_MAX: u8 = 9;

impl Default for RolePolicy {
    fn default() -> RolePolicy {
        RolePolicy { persist: true, priority: 5, ttl: None, snapshot: true }
    }
}

/// Policies for the five roles plus the one for objects without a role.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policies {
    pub system: RolePolicy,
    pub user: RolePolicy,
    pub reasoning: RolePolicy,
    pub tool: RolePolicy,
    pub assistant: RolePolicy,
}

impl Policies {
    pub fn get(&self, role: Role) -> &RolePolicy {
        match role {
            Role::System => &self.system,
            Role::User => &self.user,
            Role::Reasoning => &self.reasoning,
            Role::Tool => &self.tool,
            Role::Assistant => &self.assistant,
        }
    }
    pub fn get_mut(&mut self, role: Role) -> &mut RolePolicy {
        match role {
            Role::System => &mut self.system,
            Role::User => &mut self.user,
            Role::Reasoning => &mut self.reasoning,
            Role::Tool => &mut self.tool,
            Role::Assistant => &mut self.assistant,
        }
    }
    /// The policy of an object: its role's, or the default for objects without one.
    pub fn of(&self, role: Option<Role>) -> RolePolicy {
        role.map(|r| *self.get(r)).unwrap_or_default()
    }
    pub fn any_ttl(&self) -> bool {
        Role::ALL.iter().any(|&r| self.get(r).ttl.is_some())
    }
    /// The roles whose policy differs from the default, for log lines.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        for &r in Role::ALL.iter() {
            let p = self.get(r);
            if *p == RolePolicy::default() {
                continue;
            }
            let mut f = Vec::new();
            if !p.persist {
                f.push("not persisted".to_string());
            }
            if p.priority != 5 {
                f.push(format!("priority {}", p.priority));
            }
            if let Some(t) = p.ttl {
                f.push(format!("ttl {t} s"));
            }
            if !p.snapshot {
                f.push("no snapshots".to_string());
            }
            parts.push(format!("{}: {}", r.name(), f.join(", ")));
        }
        if parts.is_empty() { "default for every role".into() } else { parts.join("; ") }
    }
}
