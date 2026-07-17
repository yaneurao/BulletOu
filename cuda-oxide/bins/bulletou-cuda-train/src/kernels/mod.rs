//! cuda-oxide kernel definitions for the experimental BulletOu fast backend.
//!
//! cuda-oxide only emits `#[kernel]` functions that live in the binary crate.
//! Keep host-only runtime layout code in `bulletou-cuda-oxide-runtime`, but put
//! device entry points here.

pub(crate) mod backward;
pub(crate) mod loss;
pub(crate) mod nnue;
pub(crate) mod optimizer;
pub(crate) mod sfnn;

#[allow(unused_imports)]
pub(crate) use backward::*;
#[allow(unused_imports)]
pub(crate) use loss::*;
#[allow(unused_imports)]
pub(crate) use nnue::*;
#[allow(unused_imports)]
pub(crate) use optimizer::*;
#[allow(unused_imports)]
pub(crate) use sfnn::*;
