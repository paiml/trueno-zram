//! Zero-dependency ublk implementation
//!
//! Direct kernel interface using only:
//! - nix::libc for ioctls
//! - io-uring for async I/O
//!
//! No libublk, no smol, no futures.

pub mod ctrl;
pub mod daemon;
pub mod io;
pub mod shim;
pub mod sys;

pub use ctrl::{CtrlError, DeviceConfig};
#[cfg(not(test))]
pub use ctrl::UblkCtrl;
#[cfg(test)]
pub use ctrl::MockUblkCtrl as UblkCtrl;
pub use daemon::DaemonError;
#[cfg(not(test))]
pub use daemon::{run_daemon, UblkDaemon};
#[cfg(test)]
pub use daemon::MockUblkDaemon as UblkDaemon;
pub use io::{build_commit_fetch_cmd, build_fetch_cmd, user_copy_offset, IoOp, IoRequest, IoResult};
#[cfg(not(test))]
pub use shim::RealKernelShim;
pub use shim::{CtrlShim, DaemonShim, IoUringOps};
pub use sys::*;
