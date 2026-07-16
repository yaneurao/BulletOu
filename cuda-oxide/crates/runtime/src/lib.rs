//! Host-side boundary for the future BulletOu cuda-oxide backend.
//!
//! This crate intentionally has no cuda-oxide dependency yet. CO-003 only
//! creates the isolated workspace boundary. CO-004 will add PTX loading and a
//! smoke kernel without touching the root BulletOu workspace.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    SkeletonOnly,
}

pub fn backend_status() -> BackendStatus {
    BackendStatus::SkeletonOnly
}
