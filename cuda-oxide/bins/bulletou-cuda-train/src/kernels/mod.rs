//! cuda-oxide kernel definitions for the experimental BulletOu fast backend.
//!
//! cuda-oxide only emits `#[kernel]` functions that live in the binary crate.
//! Keep host-only runtime layout code in `bulletou-cuda-oxide-runtime`, but put
//! device entry points here.

pub(crate) mod nnue;

pub(crate) use nnue::*;
