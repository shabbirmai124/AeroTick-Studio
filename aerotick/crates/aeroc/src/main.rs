//! `aeroc` — AeroTick Deterministic Physics & Flight Simulation Compiler CLI (Milestones 1–4).

use aerotick_abi::emit_c_header;
use aerotick_codegen_llvm::emit_llvm_ir;
use aerotick_mir::{MirProgram, lower_to_mir};
use aerotick_sema::{HirProgram, analyze};
use aerotick_span::{Diagnostic, Severity};
use aerotick_vm::{DeterministicVm, NumericMode};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "aeroc", about = "AeroTick Deterministic Physics DSL Compiler")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Lex, parse, and verify physical dimensions + step safety in an .aero source file.
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
        /// Optional output path for native object file compilation (requires `--features llvm`).
        #[arg(long)]
        obj: Option<PathBuf>,
    },
    /// Emit the C99/C++ compatible SoA ABI header (`aerotick.h`).
    EmitHeader {
        /// Path to the .aero source file.
        file: PathBuf,
    },
    /// Execute a compiled `.aero` system deterministically on the pure-Rust SoA VM.
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
        /// Use Q32.32 fixed-point arithmetic instead of strict IEEE-754 float64.
        #[arg(long)]
        fixed64: bool,
    },
    /// Run a 1,000,000-tick deterministic benchmark and verify bit-exact SHA-256 replay hashes.
    Bench {
        /// Path to the .aero source file.
        file: PathBuf,
        /// Number of benchmark ticks.
        #[arg(long, default_value_t = 1_000_000)]
        ticks: u64,
        /// Fixed timestep in seconds.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Number of entities per tick.
        #[arg(long, default_value_t = 1)]
        entities: usize,
        /// Use Q32.32 fixed-point arithmetic instead of strict IEEE-754 float64.
        #[arg(long)]
        fixed64: bool,
    },
}

struct CompiledUnit {
    hir: HirProgram,
    mir: MirProgram,
    warnings: Vec<Diagnostic>,
}

fn compile_source(filename: &str, source: &str) -> Result<CompiledUnit, String> {
    let (tokens, mut diags) = aerotick_lexer::lex(source);
    let (ast_opt, parse_diags) = aerotick_parser::parse(&tokens, source.len());
    diags.extend(parse_diags);

    if let Some(ast) = ast_opt {
        let (hir, sema_diags) = analyze(&ast);
        diags.extend(sema_diags);

        let has_errors = diags.iter().any(|d| d.severity == Severity::Error);
        if !has_errors {
            let mir = lower_to_mir(&hir);
            return Ok(CompiledUnit {
                hir,
                mir,
                warnings: diags,
            });
        }
    }

    Err(format_diagnostics(filename, source, &diags))
}

fn format_diagnostics(filename: &str, source: &str, diags: &[Diagnostic]) -> String {
    let mut err_out = String::new();
    for diag in diags {
        match diag.render_to_string(filename, source) {
            Ok(rendered) => err_out.push_str(&rendered),
            Err(e) => err_out.push_str(&format!("[{}] {}\n(render error: {e})\n", diag.code, diag.title)),
        }
    }
    err_out
}

/// Runs checking pipeline on an in-memory source string and returns formatted summary.
pub fn check_source(filename: &str, source: &str) -> Result<String, String> {
    let unit = compile_source(filename, source)?;
    let mut out = format!(
        "Checked '{filename}' — 0 errors, {} warning(s).\n",
        unit.warnings.len()
    );
    for c in &unit.hir.consts {
        out.push_str(&format!("  const {}: {}\n", c.name, c.ty));
    }
    for st in &unit.hir.states {
        out.push_str(&format!("  state {} ({} fields)\n", st.name, st.fields.len()));
    }
    for sys in &unit.hir.systems {
        out.push_str(&format!(
            "  system {}({}: s) [step({}) -> {}]\n",
            sys.name, sys.dt_param, sys.state_name, sys.next_name
        ));
    }
    for b in &unit.hir.top_bindings {
        out.push_str(&format!("  let {}: {}\n", b.name, b.ty));
    }
    for layout in &unit.mir.layouts {
        out.push_str(&format!(
            "  soa {} => {} bytes/entity ({} B pad)\n",
            layout.state_name, layout.total_bytes_per_entity, layout.total_padding_bytes_per_entity
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
        Commands::EmitIr { file, llvm, obj } => {
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
            ExitCode::SUCCESS
        }
        Commands::EmitHeader { file } => {
            let Ok((filename, source)) = read_input(&file) else {
                return ExitCode::FAILURE;
            };
            match compile_source(&filename, &source) {
                Ok(unit) => {
                    println!("{}", emit_c_header(&unit.mir));
                    ExitCode::SUCCESS
                }
                Err(err_report) => {
                    eprint!("{err_report}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Run { file, ticks, dt, entities, fixed64 } => {
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
            let mode = if fixed64 { NumericMode::Fixed64 } else { NumericMode::Float64 };
            let mut vm = match DeterministicVm::new(&unit.mir, entities, dt, mode) {
                Ok(vm) => vm,
                Err(err) => {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
            };
            for _ in 0..ticks {
                vm.step_tick();
            }
            println!("ticks:   {}", vm.tick);
            println!("sim_t:   {:.6} s", vm.sim_time);
            println!("mode:    {mode:?}");
            println!("sha256:  {}", vm.state_sha256_hex());
            if let Some(pos) = vm.read_entity0_vec3("pos") {
                println!("pos[0]:  ({:.6}, {:.6}, {:.6}) m", pos[0], pos[1], pos[2]);
            }
            if let Some(vel) = vm.read_entity0_vec3("vel") {
                println!("vel[0]:  ({:.6}, {:.6}, {:.6}) m/s", vel[0], vel[1], vel[2]);
            }
            ExitCode::SUCCESS
        }
        Commands::Bench { file, ticks, dt, entities, fixed64 } => {
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
            let mode = if fixed64 { NumericMode::Fixed64 } else { NumericMode::Float64 };
            let mut vm_a = match DeterministicVm::new(&unit.mir, entities, dt, mode) {
                Ok(vm) => vm,
                Err(err) => {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let mut vm_b = match DeterministicVm::new(&unit.mir, entities, dt, mode) {
                Ok(vm) => vm,
                Err(err) => {
                    eprintln!("aeroc: {err}");
                    return ExitCode::FAILURE;
                }
            };

            let start = Instant::now();
            for _ in 0..ticks {
                vm_a.step_tick();
            }
            let elapsed = start.elapsed();

            for _ in 0..ticks {
                vm_b.step_tick();
            }

            let hash_a = vm_a.state_sha256_hex();
            let hash_b = vm_b.state_sha256_hex();
            let secs = elapsed.as_secs_f64().max(1e-9);
            let ticks_per_sec = (ticks as f64) / secs;

            println!("AeroTick Deterministic Benchmark");
            println!("  ticks:       {ticks}");
            println!("  entities:    {entities}");
            println!("  mode:        {mode:?}");
            println!("  elapsed:     {:.4} ms", secs * 1000.0);
            println!("  throughput:  {ticks_per_sec:.0} ticks/sec");
            println!("  run A hash:  {hash_a}");
            println!("  run B hash:  {hash_b}");
            println!("  bit_exact:   {}", hash_a == hash_b);

            if hash_a == hash_b {
                ExitCode::SUCCESS
            } else {
                eprintln!("aeroc: replay hash mismatch!");
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_source_accepts_milestone1_and_reference_body() {
        let m1 = include_str!("../../../examples/milestone1.aero");
        let res1 = check_source("examples/milestone1.aero", m1);
        assert!(res1.is_ok(), "{res1:?}");
        let out1 = res1.expect("ok");
        assert!(out1.contains("let velocity: Scalar<m/s>"));
        assert!(out1.contains("let force: Scalar<N (kg*m/s^2)>"));

        let rb = include_str!("../../../examples/reference_body.aero");
        let res2 = check_source("examples/reference_body.aero", rb);
        assert!(res2.is_ok(), "{res2:?}");
        assert!(res2.expect("ok").contains("soa Body => 104 bytes/entity"));
    }

    #[test]
    fn check_source_rejects_unit_mismatch_and_faults() {
        let src = include_str!("../../../examples/unit_mismatch.aero");
        let res = check_source("examples/unit_mismatch.aero", src);
        assert!(res.is_err());
        assert!(res.expect_err("expected error").contains("E0230"));

        let faults = include_str!("../../../examples/deterministic_faults.aero");
        let res_faults = check_source("examples/deterministic_faults.aero", faults);
        assert!(res_faults.is_err());
        let err = res_faults.expect_err("expected deterministic faults");
        assert!(err.contains("E0301"));
        assert!(err.contains("E0302"));
        assert!(err.contains("E0303"));
        assert!(err.contains("E0305"));
    }

    #[test]
    fn emits_ir_and_c_header_and_runs_deterministically() {
        let src = include_str!("../../../examples/reference_body.aero");
        let unit = compile_source("examples/reference_body.aero", src).expect("compiled");
        let llvm = emit_llvm_ir(&unit.mir);
        let header = emit_c_header(&unit.mir);
        assert!(llvm.contains("define void @aerotick_step_integrate"));
        assert!(header.contains("void aerotick_step_integrate("));

        let mut vm1 = DeterministicVm::new(&unit.mir, 1, 0.01, NumericMode::Float64).expect("vm1");
        let mut vm2 = DeterministicVm::new(&unit.mir, 1, 0.01, NumericMode::Float64).expect("vm2");
        for _ in 0..500 {
            vm1.step_tick();
            vm2.step_tick();
        }
        assert_eq!(vm1.state_sha256_hex(), vm2.state_sha256_hex());
    }
}
