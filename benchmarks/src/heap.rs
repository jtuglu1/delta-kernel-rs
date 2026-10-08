//! Heap profiles for complete workload executions, separate from Criterion timing.
//!
//! Counters include allocations on all threads when installed globally. Unrelated background work
//! can contaminate samples; live-byte extrema retain the shared allocator's advisory reset
//! contract.

use std::alloc::{GlobalAlloc, Layout};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use delta_kernel::utils::alloc_tracking::TrackingAlloc;
use serde::{Deserialize, Serialize};

use crate::runners::WorkloadRunner;

/// Tokio worker count used by the heap-profile harness for reproducible process-wide samples.
pub const RUNTIME_THREADS: usize = 2;
const REPETITIONS: usize = 5;

/// Live-byte tracking plus cumulative counters used only in heap-profile builds.
///
/// This type does not install itself globally. Bookkeeping uses atomics without allocating or
/// locking. Failed requests do not change totals. Bytes count successful allocation sizes plus
/// positive realloc growth; calls include successful alloc, alloc_zeroed, and realloc requests.
pub struct HeapAllocator {
    live: TrackingAlloc,
    allocated_bytes: AtomicU64,
    allocation_calls: AtomicU64,
}

/// One complete operation's heap usage, excluding harness setup and result reporting.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct HeapSample {
    /// Requested allocation bytes plus positive realloc growth during the operation.
    pub allocated_bytes: u64,
    /// Successful allocation and reallocation calls during the operation.
    pub allocation_calls: u64,
    /// Advisory maximum live bytes above the operation's starting baseline.
    pub peak_extra_live_bytes: u64,
    /// Advisory live bytes at the start of the operation.
    pub baseline_live_bytes: u64,
    /// Advisory live bytes when the operation finishes.
    pub end_live_bytes: u64,
}

/// Repeated heap samples for one workload after an unmeasured warm-up execution.
#[derive(Debug, Deserialize, Serialize)]
pub struct HeapProfile {
    /// Criterion benchmark name identifying the workload and harness configuration.
    pub name: String,
    /// Samples in execution order; each covers a single operation, not an iteration batch.
    pub samples: Vec<HeapSample>,
}

/// Versioned heap-profile sidecar consumed by the PR comparison script.
#[derive(Debug, Deserialize, Serialize)]
pub struct HeapReport {
    /// JSON schema version.
    pub schema_version: u32,
    /// Tokio worker count used during profiling.
    pub runtime_threads: usize,
    /// Profiles for workloads selected by the tag and Criterion name filters.
    pub workloads: Vec<HeapProfile>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AllocationTotals {
    bytes: u64,
    calls: u64,
}

impl HeapAllocator {
    /// Returns an uninstalled allocator with zeroed counters.
    pub const fn new() -> Self {
        Self {
            live: TrackingAlloc::new(),
            allocated_bytes: AtomicU64::new(0),
            allocation_calls: AtomicU64::new(0),
        }
    }

    /// Warms up `runner` once and returns repeated single-operation heap samples.
    ///
    /// Runner setup and sample storage are outside the measurement windows. The runner must finish
    /// its worker tasks and drop operation-owned results before returning. Unrelated background
    /// allocations and deferred cleanup can still affect process-wide counters.
    ///
    /// Returns the first execution error without reporting a partial profile.
    pub fn profile(
        &self,
        runner: &dyn WorkloadRunner,
    ) -> Result<HeapProfile, Box<dyn std::error::Error>> {
        let mut profile = HeapProfile {
            name: runner.name().to_owned(),
            samples: Vec::with_capacity(REPETITIONS),
        };
        runner.execute()?;
        for _ in 0..REPETITIONS {
            let sample = self.measure(|| runner.execute())?;
            profile.samples.push(sample);
        }
        Ok(profile)
    }

    /// Returns a single heap sample covering `operation`, or propagates its error.
    ///
    /// The closure must release operation-owned results and finish worker tasks before returning.
    /// Measurements through this allocator must be serialized. Reliable extrema require no
    /// concurrent allocation at reset boundaries; the method does not enforce quiescence.
    pub fn measure<E>(&self, operation: impl FnOnce() -> Result<(), E>) -> Result<HeapSample, E> {
        let baseline_live_bytes = self.live.reset_stats().current_bytes;
        let before = self.totals();
        operation()?;
        let after = self.totals();
        let peak_extra_live_bytes = self.live.max_usage().saturating_sub(baseline_live_bytes);
        let end_live_bytes = self.live.current_usage();
        Ok(HeapSample {
            allocated_bytes: after.bytes.wrapping_sub(before.bytes),
            allocation_calls: after.calls.wrapping_sub(before.calls),
            peak_extra_live_bytes,
            baseline_live_bytes,
            end_live_bytes,
        })
    }

    fn totals(&self) -> AllocationTotals {
        AllocationTotals {
            bytes: self.allocated_bytes.load(Ordering::Relaxed),
            calls: self.allocation_calls.load(Ordering::Relaxed),
        }
    }

    #[inline]
    fn record_allocation(&self, succeeded: bool, size: u64) {
        if succeeded {
            if size != 0 {
                self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
            }
            self.allocation_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    fn record_realloc(&self, succeeded: bool, old_size: u64, new_size: u64) {
        self.record_allocation(succeeded, new_size.saturating_sub(old_size));
    }
}

impl Default for HeapAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Writes `workloads` to `path`, creating parent directories and replacing any existing report.
///
/// Returns an error if no workloads ran, or if directory creation, serialization, or writing fails.
pub fn write_report(
    path: &Path,
    workloads: Vec<HeapProfile>,
) -> Result<(), Box<dyn std::error::Error>> {
    if workloads.is_empty() {
        return Err("no workloads matched the heap-profile filters".into());
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(
        &mut writer,
        &HeapReport {
            schema_version: 1,
            runtime_threads: RUNTIME_THREADS,
            workloads,
        },
    )?;
    // Flush explicitly so buffered write failures reach the caller.
    writer.into_inner()?;
    Ok(())
}

// SAFETY: all pointer/layout operations delegate to TrackingAlloc unchanged. Additional
// bookkeeping only updates atomics and never touches allocation contents.
unsafe impl GlobalAlloc for HeapAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.live.alloc(layout) };
        self.record_allocation(!ptr.is_null(), layout.size() as u64);
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.live.alloc_zeroed(layout) };
        self.record_allocation(!ptr.is_null(), layout.size() as u64);
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.live.dealloc(ptr, layout) };
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { self.live.realloc(ptr, layout, new_size) };
        self.record_realloc(!new_ptr.is_null(), layout.size() as u64, new_size as u64);
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::Arc;
    use std::thread;

    use super::{
        write_report, AllocationTotals, HeapAllocator, HeapReport, HeapSample, REPETITIONS,
    };
    use crate::runners::WorkloadRunner;

    struct AllocatingRunner<'a>(&'a HeapAllocator);

    impl WorkloadRunner for AllocatingRunner<'_> {
        fn name(&self) -> &str {
            "allocation-test"
        }

        fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
            let layout = Layout::from_size_align(64, 8)?;
            unsafe {
                let ptr = self.0.alloc(layout);
                assert!(!ptr.is_null());
                self.0.dealloc(ptr, layout);
            }
            Ok(())
        }
    }

    #[test]
    fn allocation_reallocation_and_free_track_churn_separately_from_live_bytes() {
        let allocator = HeapAllocator::new();
        let initial = Layout::from_size_align(64, 8).unwrap();
        let grown = Layout::from_size_align(128, 8).unwrap();
        let shrunk = Layout::from_size_align(32, 8).unwrap();
        let zeroed = Layout::from_size_align(16, 8).unwrap();
        let sample = allocator
            .measure(|| {
                unsafe {
                    let ptr = allocator.alloc(initial);
                    assert!(!ptr.is_null());
                    let ptr = allocator.realloc(ptr, initial, grown.size());
                    assert!(!ptr.is_null());
                    let ptr = allocator.realloc(ptr, grown, shrunk.size());
                    assert!(!ptr.is_null());
                    let ptr = allocator.realloc(ptr, shrunk, shrunk.size());
                    assert!(!ptr.is_null());
                    allocator.dealloc(ptr, shrunk);
                    let ptr = allocator.alloc_zeroed(zeroed);
                    assert!(!ptr.is_null());
                    for offset in 0..zeroed.size() {
                        assert_eq!(*ptr.add(offset), 0);
                    }
                    allocator.dealloc(ptr, zeroed);
                }
                Ok::<_, ()>(())
            })
            .unwrap();
        assert_eq!(
            sample,
            HeapSample {
                allocated_bytes: 144,
                allocation_calls: 5,
                peak_extra_live_bytes: 128,
                baseline_live_bytes: 0,
                end_live_bytes: 0,
            }
        );
    }

    #[test]
    fn failed_requests_do_not_change_totals() {
        let allocator = HeapAllocator::new();
        allocator.record_allocation(false, 64);
        for new_size in [32, 64, 128] {
            allocator.record_realloc(false, 64, new_size);
        }
        assert_eq!(allocator.totals(), AllocationTotals::default());
    }

    #[test]
    fn samples_exclude_existing_allocations_and_capture_repeated_churn() {
        let allocator = HeapAllocator::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let existing = unsafe { allocator.alloc(layout) };
        assert!(!existing.is_null());
        let runner = AllocatingRunner(&allocator);

        for _ in 0..2 {
            let sample = allocator.measure(|| runner.execute()).unwrap();
            assert_eq!(sample.allocated_bytes, 64);
            assert_eq!(sample.allocation_calls, 1);
            assert_eq!(sample.peak_extra_live_bytes, 64);
            assert_eq!(sample.baseline_live_bytes, 64);
            assert_eq!(sample.end_live_bytes, 64);
        }
        assert_eq!(allocator.totals().bytes, 192);
        unsafe { allocator.dealloc(existing, layout) };
    }

    #[test]
    fn profile_excludes_warm_up_and_round_trips_the_json_report() {
        let allocator = HeapAllocator::new();
        let profile = allocator.profile(&AllocatingRunner(&allocator)).unwrap();
        assert_eq!(profile.samples.len(), REPETITIONS);
        assert_eq!(allocator.totals().calls, (REPETITIONS + 1) as u64);
        for sample in &profile.samples {
            assert_eq!(sample.allocated_bytes, 64);
            assert_eq!(sample.allocation_calls, 1);
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profiles/report.json");
        write_report(&path, vec![profile]).unwrap();
        let report: HeapReport = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(report.schema_version, 1);
        assert_eq!(report.workloads[0].name, "allocation-test");
        assert_eq!(report.workloads[0].samples.len(), REPETITIONS);
        assert!(write_report(&directory.path().join("empty.json"), vec![]).is_err());
    }

    #[test]
    fn joined_worker_threads_contribute_to_the_same_cumulative_totals() {
        let allocator = Arc::new(HeapAllocator::new());
        let sample = allocator
            .measure(|| {
                let threads = (0..4)
                    .map(|_| {
                        let allocator = allocator.clone();
                        thread::spawn(move || {
                            let runner = AllocatingRunner(&allocator);
                            for _ in 0..32 {
                                runner.execute().unwrap();
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                for thread in threads {
                    thread.join().unwrap();
                }
                Ok::<_, ()>(())
            })
            .unwrap();
        assert_eq!(sample.allocated_bytes, 4 * 32 * 64);
        assert_eq!(sample.allocation_calls, 4 * 32);
        assert!(sample.peak_extra_live_bytes >= 64);
        assert_eq!(sample.end_live_bytes, 0);
    }

    #[test]
    fn operation_errors_are_propagated_without_a_partial_sample() {
        let allocator = HeapAllocator::new();
        assert_eq!(allocator.measure(|| Err("failed")), Err("failed"));
        assert_eq!(
            allocator
                .measure(|| Ok::<_, ()>(()))
                .unwrap()
                .allocated_bytes,
            0
        );
    }
}
