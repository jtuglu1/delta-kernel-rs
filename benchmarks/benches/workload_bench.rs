use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "heap-tracking")]
use delta_kernel_benchmarks::heap::{write_report, HeapAllocator, HeapProfile, RUNTIME_THREADS};
use delta_kernel_benchmarks::registry::BenchRegistry;
use delta_kernel_benchmarks::runners::{
    benchmark_name, configured_benchmark_name, create_read_runner, SnapshotConstructionRunner,
    WorkloadRunner,
};
use delta_kernel_benchmarks::utils::load_all_workloads;
use delta_kernel_workloads::models::{ReadOperation, Spec};
#[cfg(not(feature = "heap-tracking"))]
use test_utils::CountingReporter;

// Checked-in registry mapping each benchmark to its harness configs. Lives under the crate root
// (not the gitignored, downloaded `workloads/` dir), so it is loaded relative to
// CARGO_MANIFEST_DIR.
const REGISTRY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench-registry.json");

#[cfg(feature = "heap-tracking")]
#[global_allocator]
static HEAP_ALLOC: HeapAllocator = HeapAllocator::new();

struct Profiles {
    #[cfg(not(feature = "heap-tracking"))]
    reporter: CountingReporter,
    #[cfg(feature = "heap-tracking")]
    heap: Vec<HeapProfile>,
}

// Loads all workloads and sets up a shared runtime, then registers each as a top-level benchmark.
// For each workload, builds a runner that encapsulates the state (table info, engine, config, etc.)
// and execution logic. Timing builds print a separate IO profile; heap builds write repeated
// allocation profiles without collecting timing baselines.
fn workload_benchmarks(c: &mut Criterion) {
    #[cfg(feature = "heap-tracking")]
    assert!(
        std::env::args().any(|arg| arg == "--test"),
        "heap-tracking requires --test; run timing benchmarks without the feature"
    );

    let workloads = match load_all_workloads() {
        Ok(workloads) if !workloads.is_empty() => workloads,
        Ok(_) => panic!("No workloads found"),
        Err(e) => panic!("Failed to load workloads: {e}"),
    };

    let registry = BenchRegistry::load_from_path(Path::new(REGISTRY_PATH))
        .expect("Failed to load bench-registry.json");
    registry
        .validate(&workloads)
        .expect("bench-registry.json must match the loaded workload types");

    let mut profiles = Profiles {
        #[cfg(not(feature = "heap-tracking"))]
        reporter: CountingReporter::new(),
        #[cfg(feature = "heap-tracking")]
        heap: Vec::new(),
    };
    #[cfg(not(feature = "heap-tracking"))]
    let runtime = Arc::new(tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"));
    #[cfg(feature = "heap-tracking")]
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(RUNTIME_THREADS)
            .enable_all()
            .build()
            .expect("Failed to create heap profiling runtime"),
    );

    for workload in &workloads {
        let case_name = &workload.case_name;
        match &workload.spec {
            Spec::Read(read_spec) => {
                let configs = registry
                    .read_configs(workload)
                    .expect("loaded workload must have a registry table key");
                for operation in [ReadOperation::ReadMetadata] {
                    for config in &configs {
                        let name = configured_benchmark_name(
                            &workload.table_info,
                            case_name,
                            &config.name,
                        );
                        let runner = create_read_runner(
                            name,
                            read_spec,
                            operation,
                            config.clone(),
                            &workload.table_info,
                            runtime.clone(),
                        )
                        .expect("Failed to create read runner");
                        run_benchmark(c, runner.as_ref(), &mut profiles);
                    }
                }
            }
            Spec::SnapshotConstruction(snapshot_construction_spec) => {
                let name = benchmark_name(&workload.table_info, case_name);
                let runner = SnapshotConstructionRunner::setup(
                    name,
                    snapshot_construction_spec,
                    &workload.table_info,
                    runtime.clone(),
                )
                .expect("Failed to create snapshot construction runner");
                run_benchmark(c, &runner, &mut profiles);
            }
        }
    }
    #[cfg(feature = "heap-tracking")]
    {
        let path = std::env::var_os("BENCH_HEAP_OUTPUT")
            .unwrap_or_else(|| "target/heap-results.json".into());
        write_report(Path::new(&path), profiles.heap).expect("Failed to write heap profiles");
    }
}

// Registers a workload with Criterion and benchmarks its `execute()` function.
// After timing completes, runs an IO-profiling iteration. Heap builds use Criterion only for name
// filtering and run repeated heap samples instead. Filtered workloads produce neither profile.
fn run_benchmark(c: &mut Criterion, runner: &dyn WorkloadRunner, profiles: &mut Profiles) {
    let bench_ran = AtomicBool::new(false);
    c.bench_function(runner.name(), |b| {
        bench_ran.store(true, Ordering::Relaxed);
        #[cfg(not(feature = "heap-tracking"))]
        b.iter(|| runner.execute().expect("Benchmark execution failed"));
        // Criterion handles name filtering; heap measurement runs outside its iteration machinery.
        #[cfg(feature = "heap-tracking")]
        b.iter(|| ());
    });
    if bench_ran.load(Ordering::Relaxed) {
        #[cfg(not(feature = "heap-tracking"))]
        {
            profiles.reporter.reset();
            runner.execute().expect("IO profiling iteration failed");
            profiles.reporter.print_summary(runner.name());
        }
        #[cfg(feature = "heap-tracking")]
        profiles
            .heap
            .push(HEAP_ALLOC.profile(runner).expect("Heap profiling failed"));
    }
}

criterion_group!(benches, workload_benchmarks);
criterion_main!(benches);
