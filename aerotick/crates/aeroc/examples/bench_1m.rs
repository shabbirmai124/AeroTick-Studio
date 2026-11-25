//! 1,000,000-Tick Deterministic Replay Benchmark for AeroTick SoA VM.
//! Run with: `cargo run --release -p aeroc --example bench_1m`

use aerotick_lexer::lex;
use aerotick_mir::lower_to_mir;
use aerotick_parser::parse;
use aerotick_sema::analyze;
use aerotick_vm::{DeterministicVm, NumericMode};
use std::time::Instant;

fn main() {
    let src = include_str!("../../../examples/reference_body.aero");
    let (tokens, lex_errs) = lex(src);
    assert!(lex_errs.is_empty());
    let (ast, parse_errs) = parse(&tokens, src.len());
    assert!(parse_errs.is_empty());
    let (hir, sema_errs) = analyze(&ast.expect("valid reference AST"));
    assert!(sema_errs.is_empty());
    let mir = lower_to_mir(&hir);

    const TICKS: u64 = 1_000_000;
    const DT: f64 = 0.001;

    let mut vm_run1 = DeterministicVm::new(&mir, 1, DT, NumericMode::Float64).expect("vm1");
    let mut vm_run2 = DeterministicVm::new(&mir, 1, DT, NumericMode::Float64).expect("vm2");

    let start = Instant::now();
    for _ in 0..TICKS {
        vm_run1.step_tick();
    }
    let elapsed = start.elapsed();

    for _ in 0..TICKS {
        vm_run2.step_tick();
    }

    let hash1 = vm_run1.state_sha256_hex();
    let hash2 = vm_run2.state_sha256_hex();
    assert_eq!(hash1, hash2, "Determinism violation between replay runs!");

    let secs = elapsed.as_secs_f64().max(1e-9);
    println!("AeroTick 1,000,000-Tick Benchmark PASSED");
    println!("  Elapsed:    {:.2} ms", secs * 1000.0);
    println!("  Throughput: {:.0} ticks/s", (TICKS as f64) / secs);
    println!("  SHA-256:    {hash1}");
}
