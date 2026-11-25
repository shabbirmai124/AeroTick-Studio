#![deny(clippy::unwrap_used, clippy::expect_used)]
//! `aeroc` — AeroTick Ultra (v2.0) Deterministic Physics & Flight Simulation Compiler CLI.

use aerotick_abi::emit_c_header;
use aerotick_codegen_llvm::emit_llvm_ir;
use aerotick_mir::{MirProgram, lower_to_mir_with_opt};
use aerotick_sema::{HirProgram, analyze};
use aerotick_span::{Diagnostic, Severity};
use aerotick_vm::{DeterministicVm, NumericMode, VmCheckpoint, run_mir_tests};
use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliNumericMode {
    F64,
    Fixed64,
}

#[derive(Parser)]
#[command(
    name = "aeroc",
    version = "2.0.0",
    about = "AeroTick Ultra v2.0 Deterministic Physics DSL Compiler"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Lex, parse, and verify physical dimensions, 6-DoF types, and step safety in an .aero source file.
    Check {
        /// Path to the .aero file to verify.
        file: PathBuf,
    },
    /// Emit unit-erased SoA MIR (default) or deterministic LLVM IR (`--llvm`).
    EmitIr {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Emit LLVM IR instead of SoA MIR.
        #[arg(long)]
        llvm: bool,
        /// Run the 4-pass deterministic MIR optimizer (inlining, unrolling, constant folding, CSE/DCE).
        #[arg(long)]
        opt: bool,
        /// Print optimization & 64B cache-line utilization report.
        #[arg(long)]
        stats: bool,
        /// Optional output path for native object file compilation (requires `--features llvm`).
        #[arg(long)]
        obj: Option<PathBuf>,
    },
    /// Emit the portable C11/C++ compatible SoA ABI header (`aerotick.h`) with static assertions.
    EmitHeader {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Optional file path to write the generated C header instead of stdout.
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },
    /// Execute a compiled `.aero` system deterministically on the Swarm SoA VM or native Cranelift JIT (`--jit`).
    Run {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Number of simulation ticks to execute.
        #[arg(long, default_value_t = 1000)]
        ticks: u64,
        /// Fixed timestep in seconds.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Number of simulated entities in the SoA pool.
        #[arg(long, default_value_t = 1)]
        entities: usize,
        /// Worker threads for deterministic parallel Swarm VM chunk execution.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Compile & execute native machine code via pure-Rust Cranelift JIT.
        #[arg(long)]
        jit: bool,
        /// Numeric arithmetic mode (`f64` or `fixed64`).
        #[arg(long, value_enum, default_value_t = CliNumericMode::F64)]
        mode: CliNumericMode,
        /// Shorthand flag for Q32.32 fixed-point arithmetic (`--mode fixed64`).
        #[arg(long)]
        fixed64: bool,
    },
    /// Benchmark Float64 vs Q32.32 Fixed64 vs Cranelift JIT with bit-exact SHA-256 verification.
    Bench {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Number of benchmark ticks.
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        /// Fixed timestep in seconds.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Number of entities per tick.
        #[arg(long, default_value_t = 1)]
        entities: usize,
        /// Worker threads for deterministic parallel Swarm execution.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Include native Cranelift JIT in benchmark comparison.
        #[arg(long)]
        jit: bool,
        /// Run Float64 only (skip Q32.32 fixed-point baseline).
        #[arg(long)]
        fixed64: bool,
    },
    /// Verify deterministic checkpoint snapshot, `.aeroreplay` serialization, & bit-exact rollback replay.
    Replay {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Total ticks to simulate.
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        /// Fixed timestep in seconds.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Entities in the SoA buffer.
        #[arg(long, default_value_t = 64)]
        entities: usize,
        /// Capture checkpoint every N ticks.
        #[arg(long, default_value_t = 1000)]
        checkpoint_every: u64,
        /// Restore an earlier checkpoint and verify bit-exact SHA-256 match at target tick.
        #[arg(long)]
        verify_rollback: bool,
        /// Optional output path to save the final verified `.aeroreplay` binary checkpoint.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run built-in `.aero` `test "name" { ... assert_approx(...); }` blocks.
    Test {
        /// Path to the .aero source file containing test blocks.
        file: PathBuf,
    },
    /// Canonical `.aero` source formatter.
    Fmt {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Exit non-zero if file differs from canonical formatting without modifying it.
        #[arg(long)]
        check: bool,
        /// Write canonical formatted output back to the file in place (default unless `--check`).
        #[arg(long)]
        write: bool,
    },
}

struct CompiledUnit {
    hir: HirProgram,
    mir: MirProgram,
    warnings: Vec<Diagnostic>,
}

fn compile_source_with_opt(
    filename: &str,
    source: &str,
    optimize: bool,
) -> Result<CompiledUnit, String> {
    let (tokens, mut diags) = aerotick_lexer::lex(source);
    let (ast_opt, parse_diags) = aerotick_parser::parse(&tokens, source.len());
    diags.extend(parse_diags);

    if let Some(ast) = ast_opt {
        let (hir, sema_diags) = analyze(&ast);
        diags.extend(sema_diags);

        let has_errors = diags.iter().any(|d| d.severity == Severity::Error);
        if !has_errors {
            let mir = lower_to_mir_with_opt(&hir, optimize);
            return Ok(CompiledUnit {
                hir,
                mir,
                warnings: diags,
            });
        }
    }

    Err(format_diagnostics(filename, source, &diags))
}

fn compile_source(filename: &str, source: &str) -> Result<CompiledUnit, String> {
    compile_source_with_opt(filename, source, true)
}

fn format_diagnostics(filename: &str, source: &str, diags: &[Diagnostic]) -> String {
    let mut err_out = String::new();
    for diag in diags {
        match diag.render_to_string(filename, source) {
            Ok(rendered) => err_out.push_str(&rendered),
            Err(e) => err_out.push_str(&format!(
                "[{}] {}\n(render error: {e})\n",
                diag.code, diag.title
            )),
        }
    }
    err_out
}

/// Canonical formatter for `.aero` source files.
pub fn format_aero_source(source: &str) -> String {
    let mut out = Vec::new();
    let mut indent: usize = 0;
    for raw_line in source.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            if !out.last().is_some_and(|l: &String| l.is_empty()) {
                out.push(String::new());
            }
            continue;
        }
        let dedent = trimmed.starts_with('}') || trimmed.starts_with(')');
        if dedent {
            indent = indent.saturating_sub(1);
        }
        let pad = "    ".repeat(indent);
        out.push(format!("{pad}{trimmed}"));
        if trimmed.ends_with('{') || trimmed.ends_with('(') {
            indent += 1;
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    let mut joined = out.join("\n");
    joined.push('\n');
    joined
}

/// Runs checking pipeline on an in-memory source string and returns formatted summary.
pub fn check_source(filename: &str, source: &str) -> Result<String, String> {
    let unit = compile_source(filename, source)?;
    let mut out = format!(
        "Checked '{filename}' — 0 errors, {} warning(s).\n",
        unit.warnings.len()
    );
    for (name, dim, scale, _) in &unit.hir.unit_aliases {
        out.push_str(&format!("  unit {name} = {dim} (scale {scale})\n"));
    }
    for c in &unit.hir.consts {
        out.push_str(&format!("  const {}: {}\n", c.name, c.ty));
    }
    for st in &unit.hir.states {
        out.push_str(&format!(
            "  state {} ({} fields)\n",
            st.name,
            st.fields.len()
        ));
    }
    for f in &unit.hir.functions {
        out.push_str(&format!("  pure fn {} -> {}\n", f.name, f.return_type));
    }
    for sys in &unit.hir.systems {
        out.push_str(&format!(
            "  system {}({}: s) [step({}) -> {}]\n",
            sys.name, sys.dt_param, sys.state_name, sys.next_name
        ));
    }
    for t in &unit.hir.tests {
        out.push_str(&format!("  test \"{}\"\n", t.name));
    }
    for b in &unit.hir.top_bindings {
        out.push_str(&format!("  let {}: {}\n", b.name, b.ty));
    }
    for layout in &unit.mir.layouts {
        out.push_str(&format!(
            "  soa {} => {} bytes/entity ({} B pad, {}x64B lines, {:.1}% util)\n",
            layout.state_name,
            layout.total_bytes_per_entity,
            layout.total_padding_bytes_per_entity,
            layout.cache_lines_64b(),
            layout.cache_line_utilization_pct(),
        ));
    }
    if !unit.warnings.is_empty() {
        out.push('\n');
        out.push_str(&format_diagnostics(filename, source, &unit.warnings));
    }
    Ok(out)
}

fn read_input(file: &PathBuf) -> Result<(String, String), ExitCode> {
    let filename = file.display().to_string();
    match fs::read_to_string(file) {
        Ok(src) => Ok((filename, src)),
        Err(err) => {
            eprintln!("aeroc: failed to read '{filename}': {err}");
            Err(ExitCode::FAILURE)
        }
    }
}

fn run_vm_ticks(vm: &mut DeterministicVm, ticks: u64, threads: usize) -> Result<(), String> {
    for _ in 0..ticks {
        vm.step_tick_parallel(threads).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(feature = "jit")]
fn run_jit_bench(
    mir: &MirProgram,
    entities: usize,
    ticks: u64,
    dt: f64,
) -> Result<(f64, String, String), String> {
    let kernel =
        aerotick_codegen_cranelift::CraneliftJitKernel::compile(mir).map_err(|e| e.to_string())?;
    let mut vm =
        DeterministicVm::new(mir, entities, dt, NumericMode::Float64).map_err(|e| e.to_string())?;
    let start = Instant::now();
    for _ in 0..ticks {
        let (prev_ptrs, next_ptrs) = vm.field_ptrs_mut();
        unsafe {
            kernel.step(&prev_ptrs, &next_ptrs, vm.count, vm.dt);
        }
        vm.commit_jit_tick();
    }
    let elapsed_s = start.elapsed().as_secs_f64().max(1e-9);
    Ok((elapsed_s, vm.state_sha256_hex(), kernel.target_isa))
}

#[cfg(not(feature = "jit"))]
fn run_jit_bench(
    _mir: &MirProgram,
    _entities: usize,
    _ticks: u64,
    _dt: f64,
) -> Result<(f64, String, String), String> {
    Err("Cranelift JIT feature is disabled; rebuild with `--features jit`".to_string())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Check { file } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            match check_source(&filename, &source) {
                Ok(report) => {
                    print!("{report}");
                    ExitCode::SUCCESS
                }
                Err(err_report) => {
                    eprint!("{err_report}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::EmitIr {
            file,
            llvm,
            opt,
            stats,
            obj,
        } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let run_opt = opt || llvm || obj.is_some();
            let unit = match compile_source_with_opt(&filename, &source, run_opt) {
                Ok(u) => u,
                Err(err_report) => {
                    eprint!("{err_report}");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(obj_path) = obj {
                if let Err(err) = aerotick_codegen_llvm::compile_to_object(&unit.mir, &obj_path) {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
                println!("Wrote native object file to '{}'", obj_path.display());
            }
            if llvm {
                println!("{}", emit_llvm_ir(&unit.mir));
            } else {
                println!("{}", unit.mir.format_ir());
            }
            if stats {
                println!("\n{}", unit.mir.format_stats());
            }
            ExitCode::SUCCESS
        }
        Commands::EmitHeader { file, output } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            match compile_source(&filename, &source) {
                Ok(unit) => {
                    let header = emit_c_header(&unit.mir);
                    if let Some(out_path) = output {
                        if let Err(err) = fs::write(&out_path, header) {
                            eprintln!("aeroc: failed to write '{}': {err}", out_path.display());
                            return ExitCode::FAILURE;
                        }
                        println!("Wrote C11/C++ ABI header to '{}'", out_path.display());
                    } else {
                        println!("{header}");
                    }
                    ExitCode::SUCCESS
                }
                Err(err_report) => {
                    eprint!("{err_report}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Run {
            file,
            ticks,
            dt,
            entities,
            threads,
            jit,
            mode,
            fixed64,
        } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let unit = match compile_source(&filename, &source) {
                Ok(u) => u,
                Err(err_report) => {
                    eprint!("{err_report}");
                    return ExitCode::FAILURE;
                }
            };
            if jit {
                match run_jit_bench(&unit.mir, entities, ticks, dt) {
                    Ok((elapsed_s, sha, isa)) => {
                        println!("engine:   Cranelift Native JIT ({isa})");
                        println!("ticks:    {ticks}");
                        println!("entities: {entities}");
                        println!("elapsed:  {:.3} ms", elapsed_s * 1000.0);
                        println!("sha256:   {sha}");
                        return ExitCode::SUCCESS;
                    }
                    Err(e) => {
                        eprintln!("aeroc: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            let numeric_mode = if fixed64 || mode == CliNumericMode::Fixed64 {
                NumericMode::Fixed64
            } else {
                NumericMode::Float64
            };
            let mut vm = match DeterministicVm::new(&unit.mir, entities, dt, numeric_mode) {
                Ok(vm) => vm,
                Err(err) => {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(err) = run_vm_ticks(&mut vm, ticks, threads) {
                eprintln!("aeroc: {err}");
                return ExitCode::FAILURE;
            }
            println!("engine:  Swarm VM ({threads} thread(s), {numeric_mode:?})");
            println!("ticks:   {}", vm.tick);
            println!("sim_t:   {:.6} s", vm.sim_time);
            println!("sha256:  {}", vm.state_sha256_hex());
            if let Some(pos) = vm.read_entity0_vec3("pos") {
                println!("pos[0]:  ({:.6}, {:.6}, {:.6}) m", pos[0], pos[1], pos[2]);
            }
            if let Some(vel) = vm.read_entity0_vec3("vel") {
                println!("vel[0]:  ({:.6}, {:.6}, {:.6}) m/s", vel[0], vel[1], vel[2]);
            }
            if let Some(q) = vm.read_entity0_quat("quat") {
                println!(
                    "quat[0]: ({:.6}, {:.6}, {:.6}, {:.6})",
                    q[0], q[1], q[2], q[3]
                );
            }
            ExitCode::SUCCESS
        }
        Commands::Bench {
            file,
            ticks,
            dt,
            entities,
            threads,
            jit,
            fixed64,
        } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let unit = match compile_source(&filename, &source) {
                Ok(u) => u,
                Err(err_report) => {
                    eprint!("{err_report}");
                    return ExitCode::FAILURE;
                }
            };

            let bench_mode = |mode: NumericMode| -> Result<(f64, String), String> {
                let mut vm = DeterministicVm::new(&unit.mir, entities, dt, mode)
                    .map_err(|e| e.to_string())?;
                let start = Instant::now();
                run_vm_ticks(&mut vm, ticks, threads)?;
                let elapsed_s = start.elapsed().as_secs_f64().max(1e-9);
                Ok((elapsed_s, vm.state_sha256_hex()))
            };

            let (f64_s, f64_hash) = match bench_mode(NumericMode::Float64) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("aeroc: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let (_, f64_replay_hash) = match bench_mode(NumericMode::Float64) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("aeroc: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let (q32_s, q32_hash) = if !fixed64 {
                match bench_mode(NumericMode::Fixed64) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!("aeroc: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                (f64_s, f64_hash.clone())
            };

            println!(
                "AeroTick Ultra v2.0 Benchmark ({entities} entities x {ticks} ticks, {threads} thread(s))"
            );
            println!(
                "  Float64 VM     : {:>8.2} ms | {:>10.0} ticks/s | {:>12.0} ent-ticks/s | {}",
                f64_s * 1000.0,
                (ticks as f64) / f64_s,
                ((ticks as f64) * (entities as f64)) / f64_s,
                &f64_hash[..16]
            );
            if !fixed64 {
                println!(
                    "  Fixed64 Q32.32 : {:>8.2} ms | {:>10.0} ticks/s | {:>12.0} ent-ticks/s | {}",
                    q32_s * 1000.0,
                    (ticks as f64) / q32_s,
                    ((ticks as f64) * (entities as f64)) / q32_s,
                    &q32_hash[..16]
                );
            }

            if jit {
                match run_jit_bench(&unit.mir, entities, ticks, dt) {
                    Ok((jit_s, jit_hash, isa)) => {
                        println!(
                            "  Cranelift JIT  : {:>8.2} ms | {:>10.0} ticks/s | {:>12.0} ent-ticks/s | {} ({isa})",
                            jit_s * 1000.0,
                            (ticks as f64) / jit_s,
                            ((ticks as f64) * (entities as f64)) / jit_s,
                            &jit_hash[..16]
                        );
                        if jit_hash != f64_hash {
                            eprintln!("aeroc: JIT vs Float64 VM SHA-256 mismatch!");
                            return ExitCode::FAILURE;
                        }
                    }
                    Err(e) => {
                        eprintln!("aeroc: JIT failed: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }

            println!("  bit_exact_f64  : {}", f64_hash == f64_replay_hash);
            if f64_hash == f64_replay_hash {
                ExitCode::SUCCESS
            } else {
                eprintln!("aeroc: replay hash mismatch!");
                ExitCode::FAILURE
            }
        }
        Commands::Replay {
            file,
            ticks,
            dt,
            entities,
            checkpoint_every,
            verify_rollback,
            out,
        } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let unit = match compile_source(&filename, &source) {
                Ok(u) => u,
                Err(err_report) => {
                    eprint!("{err_report}");
                    return ExitCode::FAILURE;
                }
            };
            let mut vm = match DeterministicVm::new(&unit.mir, entities, dt, NumericMode::Float64) {
                Ok(vm) => vm,
                Err(err) => {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let step_every = checkpoint_every.max(1);
            let mut checkpoints = Vec::new();
            for t in 0..ticks {
                if t % step_every == 0 {
                    checkpoints.push(vm.save_checkpoint());
                }
                vm.step_tick();
            }
            let final_cp = vm.save_checkpoint();
            let final_hash = final_cp.sha256_hex.clone();
            println!(
                "Captured {} checkpoints over {ticks} ticks ({entities} entities)",
                checkpoints.len()
            );
            println!("  final SHA-256: {final_hash}");

            if let Some(out_path) = out {
                let blob = final_cp.to_bytes();
                if let Err(err) = VmCheckpoint::from_bytes(&blob) {
                    eprintln!("aeroc: checkpoint verification failed: {err}");
                    return ExitCode::FAILURE;
                }
                if let Err(err) = fs::write(&out_path, &blob) {
                    eprintln!("aeroc: failed to write '{}': {err}", out_path.display());
                    return ExitCode::FAILURE;
                }
                println!(
                    "  wrote .aeroreplay ({} bytes) to '{}'",
                    blob.len(),
                    out_path.display()
                );
            }

            if verify_rollback && let Some(mid_cp) = checkpoints.get(checkpoints.len() / 2) {
                let roundtrip_bytes = mid_cp.to_bytes();
                let decoded_cp = match VmCheckpoint::from_bytes(&roundtrip_bytes) {
                    Ok(cp) => cp,
                    Err(e) => {
                        eprintln!("aeroc: .aeroreplay decode failed: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                let remaining = ticks.saturating_sub(decoded_cp.tick);
                if let Err(e) = vm.restore_checkpoint(&decoded_cp) {
                    eprintln!("aeroc: rollback failed: {e}");
                    return ExitCode::FAILURE;
                }
                for _ in 0..remaining {
                    vm.step_tick();
                }
                let rollback_hash = vm.state_sha256_hex();
                println!(
                    "  rollback tick {} -> {ticks} SHA-256: {rollback_hash} (match={})",
                    decoded_cp.tick,
                    rollback_hash == final_hash
                );
                if rollback_hash != final_hash {
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Commands::Test { file } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let unit = match compile_source(&filename, &source) {
                Ok(u) => u,
                Err(err_report) => {
                    eprint!("{err_report}");
                    return ExitCode::FAILURE;
                }
            };
            let reports = run_mir_tests(&unit.mir);
            let mut failed = 0usize;
            for r in &reports {
                if r.passed {
                    println!("  [PASS] {}", r.name);
                } else {
                    failed += 1;
                    println!(
                        "  [FAIL] {}: {}",
                        r.name,
                        r.message.as_deref().unwrap_or("failed")
                    );
                }
            }
            println!(
                "test result: {} passed, {failed} failed",
                reports.len() - failed
            );
            if failed == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Commands::Fmt {
            file,
            check,
            write: _write,
        } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            let formatted = format_aero_source(&source);
            if check {
                if formatted == source {
                    println!("Formatted OK: '{filename}'");
                    ExitCode::SUCCESS
                } else {
                    eprintln!("aeroc fmt: '{filename}' is not formatted canonically");
                    ExitCode::FAILURE
                }
            } else if let Err(e) = fs::write(&file, formatted) {
                eprintln!("aeroc fmt: failed to write '{filename}': {e}");
                ExitCode::FAILURE
            } else {
                println!("Formatted '{filename}'");
                ExitCode::SUCCESS
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_reference_body_and_soa_layout() {
        let rb = include_str!("../../../examples/reference_body.aero");
        let out = match check_source("examples/reference_body.aero", rb) {
            Ok(o) => o,
            Err(e) => panic!("reference_body.aero failed: {e}"),
        };
        assert!(out.contains("soa Body => 104 bytes/entity"));
        assert!(out.contains("system Integrate(dt: s)"));
    }

    #[test]
    fn verifies_flagship_6dof_pid_and_invalid_examples() {
        for (name, src) in [
            (
                "examples/rocket_6dof.aero",
                include_str!("../../../examples/rocket_6dof.aero"),
            ),
            (
                "examples/quadcopter_pid.aero",
                include_str!("../../../examples/quadcopter_pid.aero"),
            ),
        ] {
            let unit = match compile_source(name, src) {
                Ok(u) => u,
                Err(e) => panic!("{name} failed to compile: {e}"),
            };
            let reports = run_mir_tests(&unit.mir);
            assert!(!reports.is_empty(), "expected inline tests in {name}");
            for r in reports {
                assert!(
                    r.passed,
                    "test '{}' in {name} failed: {:?}",
                    r.name, r.message
                );
            }
        }

        let bad = include_str!("../../../examples/invalid_physics.aero");
        let Err(err) = compile_source("examples/invalid_physics.aero", bad) else {
            panic!("expected invalid_physics.aero to fail compilation");
        };
        for code in ["E0202", "E0210", "E0301", "E0303", "E0305"] {
            assert!(err.contains(code), "missing expected {code} in {err}");
        }
    }

    #[test]
    fn emits_ir_and_c_header_and_verifies_replay_roundtrip() {
        let src = include_str!("../../../examples/reference_body.aero");
        let unit = match compile_source("examples/reference_body.aero", src) {
            Ok(u) => u,
            Err(e) => panic!("reference_body.aero failed: {e}"),
        };
        let llvm = emit_llvm_ir(&unit.mir);
        let header = emit_c_header(&unit.mir);
        assert!(llvm.contains("define void @aerotick_step_integrate"));
        assert!(llvm.contains("define void @aerotick_step_batch_integrate"));
        assert!(header.contains("void aerotick_step_integrate("));
        assert!(header.contains("AeroVec3Simd4"));

        let mut vm1 = match DeterministicVm::new(&unit.mir, 8, 0.01, NumericMode::Float64) {
            Ok(v) => v,
            Err(e) => panic!("vm1 init failed: {e}"),
        };
        let mut vm2 = match DeterministicVm::new(&unit.mir, 8, 0.01, NumericMode::Float64) {
            Ok(v) => v,
            Err(e) => panic!("vm2 init failed: {e}"),
        };
        for _ in 0..250 {
            vm1.step_tick();
            vm2.step_tick();
        }
        let cp_bytes = vm1.save_checkpoint().to_bytes();
        let decoded = match VmCheckpoint::from_bytes(&cp_bytes) {
            Ok(c) => c,
            Err(e) => panic!("checkpoint decode failed: {e}"),
        };
        for _ in 0..250 {
            vm1.step_tick();
        }
        if let Err(e) = vm2.restore_checkpoint(&decoded) {
            panic!("restore failed: {e}");
        }
        for _ in 0..250 {
            vm2.step_tick();
        }
        assert_eq!(vm1.state_sha256_hex(), vm2.state_sha256_hex());
    }
}
