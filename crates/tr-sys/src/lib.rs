//! tr-sys: topology, NUMA memory, pinned worker pool, barriers, AMX enablement.
pub mod amx;
pub mod topology;
pub mod numa;
pub mod affinity;
pub mod loader;
pub mod procinfo;
pub mod pool;
