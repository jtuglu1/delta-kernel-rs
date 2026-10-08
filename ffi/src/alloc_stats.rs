//! FFI access to process-wide Rust allocation counters.
//!
//! # Scope
//!
//! Counters record requested allocation sizes only, so they are a lower bound on process RSS:
//! C/`mmap` allocations and allocator overhead are not included. Values are process-global and
//! advisory, rather than measurements for a single operation. The counters use relaxed atomic
//! operations and do not form a coherent snapshot. See [`reset_native_memory_stats`] for the
//! concurrent-reset contract.

#[cfg(feature = "alloc-tracking")]
use delta_kernel::utils::alloc_tracking::TrackingAlloc;

#[cfg(feature = "alloc-tracking")]
#[global_allocator]
static GLOBAL_ALLOC: TrackingAlloc = TrackingAlloc::new();

/// Whether this library was built with allocation tracking.
///
/// The `*_native_bytes` getters return zero when this is false.
#[no_mangle]
pub extern "C" fn alloc_tracking_enabled() -> bool {
    cfg!(feature = "alloc-tracking")
}

/// Independently sampled allocation statistics written by [`reset_native_memory_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NativeMemoryStats {
    /// Reported minimum native bytes before the reset.
    pub min_bytes: u64,
    /// Native bytes sampled as live while resetting the measurement window.
    pub current_bytes: u64,
    /// Reported maximum native bytes before the reset.
    pub max_bytes: u64,
}

/// Reported minimum simultaneously live native bytes since the library was loaded or reset.
///
/// The value is process-wide, not per-operation, and counts requested Rust allocation sizes only.
/// The value remains zero until [`reset_native_memory_stats`] starts a new measurement window.
/// Its relationship to the current total is advisory; see the concurrent-reset contract.
/// Returns zero when built without `alloc-tracking`.
#[no_mangle]
pub extern "C" fn min_native_bytes() -> u64 {
    #[cfg(feature = "alloc-tracking")]
    {
        GLOBAL_ALLOC.min_usage()
    }
    #[cfg(not(feature = "alloc-tracking"))]
    {
        0
    }
}

/// Reported maximum simultaneously live native bytes since the library was loaded or reset.
///
/// The value is process-wide, not per-operation, and counts requested Rust allocation sizes only.
/// Its relationship to the current total is advisory; see [`reset_native_memory_stats`].
/// Returns zero when built without `alloc-tracking`.
#[no_mangle]
pub extern "C" fn max_native_bytes() -> u64 {
    #[cfg(feature = "alloc-tracking")]
    {
        GLOBAL_ALLOC.max_usage()
    }
    #[cfg(not(feature = "alloc-tracking"))]
    {
        0
    }
}

/// Native bytes currently live (allocated but not yet freed).
///
/// The value is process-wide and counts requested Rust allocation sizes only. Returns zero when
/// built without `alloc-tracking`.
#[no_mangle]
pub extern "C" fn current_native_bytes() -> u64 {
    #[cfg(feature = "alloc-tracking")]
    {
        GLOBAL_ALLOC.current_usage()
    }
    #[cfg(not(feature = "alloc-tracking"))]
    {
        0
    }
}

/// Samples the current live total, independently resets both extrema to that sample, and writes
/// the previous extrema and current sample to `stats`.
///
/// Concurrent allocation or reset can lose extrema or mix measurement windows. The returned fields
/// are not a coherent snapshot, and after a concurrent reset the reported minimum can remain above
/// current or the maximum below current. No automatic repair is guaranteed. Callers
/// requiring reliable windows must quiesce tracked allocation and serialize resets.
///
/// All fields are zero when built without `alloc-tracking`.
///
/// # Safety
///
/// `stats` must be non-null, aligned, and writable for a `NativeMemoryStats`. The storage may be
/// uninitialized and must not be accessed by other code for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn reset_native_memory_stats(stats: *mut NativeMemoryStats) {
    #[cfg(feature = "alloc-tracking")]
    let value = {
        let stats = GLOBAL_ALLOC.reset_stats();
        NativeMemoryStats {
            min_bytes: stats.min_bytes,
            current_bytes: stats.current_bytes,
            max_bytes: stats.max_bytes,
        }
    };
    #[cfg(not(feature = "alloc-tracking"))]
    let value = NativeMemoryStats::default();
    // SAFETY: the caller guarantees aligned, exclusively writable storage for the output.
    unsafe { stats.write(value) };
}

#[cfg(all(test, not(feature = "alloc-tracking")))]
mod disabled_tests {
    use std::mem::MaybeUninit;

    use rstest::rstest;

    use super::{
        alloc_tracking_enabled, current_native_bytes, max_native_bytes, min_native_bytes,
        reset_native_memory_stats, NativeMemoryStats,
    };

    #[rstest]
    fn getters_report_disabled_tracking(#[values(false, true)] initialized: bool) {
        assert!(!alloc_tracking_enabled());
        assert_eq!(min_native_bytes(), 0);
        assert_eq!(max_native_bytes(), 0);
        assert_eq!(current_native_bytes(), 0);
        let mut stats = if initialized {
            MaybeUninit::new(NativeMemoryStats {
                min_bytes: 1,
                current_bytes: 2,
                max_bytes: 3,
            })
        } else {
            MaybeUninit::uninit()
        };
        // SAFETY: stats is aligned, writable storage, and reset initializes all its fields.
        let stats = unsafe {
            reset_native_memory_stats(stats.as_mut_ptr());
            stats.assume_init()
        };
        assert_eq!(stats, NativeMemoryStats::default());
    }
}

#[cfg(all(test, feature = "alloc-tracking"))]
mod global_allocator_tests {
    use std::mem::MaybeUninit;

    use super::{
        alloc_tracking_enabled, current_native_bytes, max_native_bytes, min_native_bytes,
        reset_native_memory_stats, NativeMemoryStats,
    };

    // Far above incidental harness allocation, so the bounds below cannot be met by noise.
    const N: usize = 8 * 1024 * 1024;

    #[test]
    fn installed_global_allocator_accounts_a_large_allocation() {
        assert!(alloc_tracking_enabled());
        assert_eq!(min_native_bytes(), 0);

        let buf = vec![0u8; N];
        assert!(current_native_bytes() >= N as u64);
        assert!(max_native_bytes() >= N as u64);

        for mut stats in [
            MaybeUninit::uninit(),
            MaybeUninit::new(NativeMemoryStats {
                min_bytes: u64::MAX,
                current_bytes: u64::MAX,
                max_bytes: u64::MAX,
            }),
        ] {
            // SAFETY: stats is aligned, writable storage, and reset initializes all its fields.
            let stats = unsafe {
                reset_native_memory_stats(stats.as_mut_ptr());
                stats.assume_init()
            };
            assert!(stats.current_bytes >= N as u64);
            assert!(stats.current_bytes < u64::MAX);
            assert!(stats.max_bytes >= N as u64);
            assert!(stats.max_bytes < u64::MAX);
            assert!(stats.min_bytes < u64::MAX);
        }

        drop(buf);
    }
}
