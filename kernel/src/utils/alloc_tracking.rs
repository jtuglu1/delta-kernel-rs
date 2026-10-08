//! Requested Rust allocation sizes, independent of allocator installation or FFI.
//!
//! Installing [`TrackingAlloc`] as a global allocator is the consuming crate's responsibility.
//! Counters exclude allocator overhead, C allocations, and `mmap`, and are not an RSS measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

/// A system allocator with advisory current/minimum/maximum live-byte counters.
///
/// Accounting uses relaxed atomics without bookkeeping allocations or locks. Concurrent getters
/// do not form a coherent snapshot. See [`Self::reset_stats`] for the measurement-window contract.
pub struct TrackingAlloc {
    current: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

/// Independently sampled live-byte counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationStats {
    /// Reported minimum live bytes before the reset.
    pub min_bytes: u64,
    /// Live bytes sampled while resetting the window.
    pub current_bytes: u64,
    /// Reported maximum live bytes before the reset.
    pub max_bytes: u64,
}

impl TrackingAlloc {
    /// Returns an allocator with zeroed counters, without installing it globally.
    pub const fn new() -> Self {
        Self {
            current: AtomicU64::new(0),
            min: AtomicU64::new(0),
            max: AtomicU64::new(0),
        }
    }

    /// Returns the reported minimum requested live bytes since construction or reset.
    ///
    /// This remains zero until the first reset. The relationship to current usage is advisory.
    pub fn min_usage(&self) -> u64 {
        self.min.load(Ordering::Relaxed)
    }

    /// Returns the reported maximum requested live bytes since construction or reset.
    ///
    /// The relationship to current usage is advisory.
    pub fn max_usage(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    /// Returns the requested bytes currently live through this allocator.
    pub fn current_usage(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// Samples current usage, rebases both extrema independently, and returns the previous extrema.
    ///
    /// Concurrent allocation or reset can lose extrema, mix windows, and leave minimum above
    /// current or maximum below current. No automatic repair is guaranteed. Reliable windows
    /// require quiescing allocation through this allocator and serializing resets.
    pub fn reset_stats(&self) -> AllocationStats {
        let current_bytes = self.current_usage();
        AllocationStats {
            min_bytes: self.min.swap(current_bytes, Ordering::Relaxed),
            current_bytes,
            max_bytes: self.max.swap(current_bytes, Ordering::Relaxed),
        }
    }

    #[inline]
    fn grow_usage(&self, size: u64) {
        let previous = self.current.fetch_add(size, Ordering::Relaxed);
        self.max
            .fetch_max(previous.saturating_add(size), Ordering::Relaxed);
    }

    #[inline]
    fn shrink_usage(&self, size: u64) {
        let previous = self.current.fetch_sub(size, Ordering::Relaxed);
        self.min
            .fetch_min(previous.saturating_sub(size), Ordering::Relaxed);
    }

    #[inline]
    fn record_realloc(&self, succeeded: bool, old_size: u64, new_size: u64) {
        if !succeeded {
            return;
        }
        if new_size > old_size {
            self.grow_usage(new_size - old_size);
        } else if old_size > new_size {
            self.shrink_usage(old_size - new_size);
        }
    }
}

impl Default for TrackingAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: pointers and layouts are forwarded to `System` unchanged. Bookkeeping only updates
// atomics and does not access the allocated memory.
unsafe impl GlobalAlloc for TrackingAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.grow_usage(layout.size() as u64);
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        self.shrink_usage(layout.size() as u64);
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            self.grow_usage(layout.size() as u64);
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        self.record_realloc(!new_ptr.is_null(), layout.size() as u64, new_size as u64);
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::{AllocationStats, TrackingAlloc};

    #[test]
    fn allocator_tracks_current_minimum_and_maximum_usage() {
        let allocator = TrackingAlloc::new();
        let initial_layout = Layout::from_size_align(64, 8).unwrap();
        let grown_layout = Layout::from_size_align(128, 8).unwrap();
        let shrunk_layout = Layout::from_size_align(32, 8).unwrap();
        let zeroed_layout = Layout::from_size_align(16, 8).unwrap();

        unsafe {
            let ptr = allocator.alloc(initial_layout);
            assert!(!ptr.is_null());
            assert_eq!(allocator.current_usage(), 64);
            assert_eq!(allocator.min_usage(), 0);
            assert_eq!(allocator.max_usage(), 64);
            assert_eq!(
                allocator.reset_stats(),
                AllocationStats {
                    min_bytes: 0,
                    current_bytes: 64,
                    max_bytes: 64,
                }
            );

            let ptr = allocator.realloc(ptr, initial_layout, grown_layout.size());
            assert!(!ptr.is_null());
            assert_eq!(allocator.current_usage(), 128);
            assert_eq!(allocator.min_usage(), 64);
            assert_eq!(allocator.max_usage(), 128);

            let ptr = allocator.realloc(ptr, grown_layout, shrunk_layout.size());
            assert!(!ptr.is_null());
            assert_eq!(allocator.current_usage(), 32);
            assert_eq!(allocator.min_usage(), 32);
            assert_eq!(allocator.max_usage(), 128);

            allocator.dealloc(ptr, shrunk_layout);
            assert_eq!(allocator.current_usage(), 0);
            assert_eq!(allocator.min_usage(), 0);
            assert_eq!(allocator.max_usage(), 128);

            let ptr = allocator.alloc_zeroed(zeroed_layout);
            assert!(!ptr.is_null());
            assert_eq!(allocator.current_usage(), 16);
            assert_eq!(allocator.max_usage(), 128);
            for offset in 0..zeroed_layout.size() {
                assert_eq!(*ptr.add(offset), 0);
            }
            allocator.dealloc(ptr, zeroed_layout);
        }
    }

    #[test]
    fn reset_returns_distinct_extrema_and_rebases_both_to_current() {
        let allocator = TrackingAlloc::new();
        allocator.grow_usage(64);
        allocator.reset_stats();
        allocator.shrink_usage(32);
        allocator.grow_usage(96);
        allocator.shrink_usage(64);

        assert_eq!(
            allocator.reset_stats(),
            AllocationStats {
                min_bytes: 32,
                current_bytes: 64,
                max_bytes: 128,
            }
        );
        assert_eq!(allocator.min_usage(), 64);
        assert_eq!(allocator.current_usage(), 64);
        assert_eq!(allocator.max_usage(), 64);
        assert_eq!(
            allocator.reset_stats(),
            AllocationStats {
                min_bytes: 64,
                current_bytes: 64,
                max_bytes: 64,
            }
        );
    }

    #[test]
    fn failed_realloc_does_not_change_usage() {
        let allocator = TrackingAlloc::new();
        allocator.grow_usage(64);
        let before = allocator.reset_stats();

        allocator.record_realloc(false, 64, 128);

        assert_eq!(allocator.current_usage(), before.current_bytes);
        assert_eq!(allocator.min_usage(), before.current_bytes);
        assert_eq!(allocator.max_usage(), before.current_bytes);
    }

    #[test]
    fn concurrent_frees_lower_minimum_below_reset_baseline_and_balance_current_usage() {
        let allocator = Arc::new(TrackingAlloc::new());
        let layout = Layout::from_size_align(64, 8).unwrap();
        let barrier = Arc::new(Barrier::new(5));
        let threads = (0..4)
            .map(|_| {
                let allocator = Arc::clone(&allocator);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let ptr = unsafe { allocator.alloc(layout) };
                    assert!(!ptr.is_null());
                    barrier.wait();
                    barrier.wait();
                    unsafe { allocator.dealloc(ptr, layout) };
                    barrier.wait();
                    for _ in 0..32 {
                        unsafe {
                            let ptr = allocator.alloc(layout);
                            assert!(!ptr.is_null());
                            allocator.dealloc(ptr, layout);
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        // The barriers isolate reset and make the minimum reach zero before the loops.
        barrier.wait();
        assert_eq!(
            allocator.reset_stats(),
            AllocationStats {
                min_bytes: 0,
                current_bytes: 256,
                max_bytes: 256,
            }
        );
        assert_eq!(allocator.min_usage(), 256);
        barrier.wait();
        barrier.wait();

        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(allocator.current_usage(), 0);
        assert_eq!(allocator.min_usage(), 0);
        assert_eq!(allocator.max_usage(), 256);
    }
}
