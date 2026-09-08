//! Persistent prefix cache for the inference engine: the sequence state of finished requests,
//! stored as role-tagged *rows chunks* (the position-indexed K/V and indexer rows of a token
//! range) and *snapshots* (the recurrent state at one position), behind a pluggable blob
//! [`Backend`] with an LRU byte budget.
//!
//! This crate knows nothing about the model: it moves `&[u8]` bodies with metadata. Objects are
//! addressed by a [`PrefixKey`], the chain hash of everything before a position (model identity,
//! token ids, image content), so equal keys mean equal state and a prefix of one conversation is
//! shared by every conversation that starts the same way. A restore point is a snapshot at
//! position P whose rows chunks chain back to position 0 without a gap ([`Cache::best_restore`]).
pub mod backend;
pub mod buf;
pub mod cache;
pub mod fs;
pub mod key;
pub mod object;
pub mod policy;
pub mod role;

pub use backend::{Backend, MemBackend};
pub use buf::AlignedBuf;
pub use cache::{Cache, RestorePlan, Stats};
pub use fs::FsBackend;
pub use key::{prefix_keys, root_key, PrefixKey};
pub use object::{Kind, ObjectMeta, ALIGN, FORMAT};
pub use policy::{Policies, RolePolicy, PRIORITY_MAX};
pub use role::{chunk_spans, Role, Span};
