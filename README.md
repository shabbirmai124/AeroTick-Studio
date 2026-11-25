# AeroTick Ultra (v2.0) — Deterministic 6-DoF Physics DSL & SoA Compiler

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**AeroTick Ultra (v2.0)** is a pure-Rust, deterministic, domain-specific systems programming language and 12-crate Cargo Edition 2024 compiler (`aeroc`) engineered for 6-DoF aerospace kinematics, flight simulation, robotics PID control, and multi-entity physics swarms.

AeroTick eliminates the three classic sources of physics simulation bugs at compile time:
1. **Unit-Mismatch Disasters** — First-class 7-component SI dimensional analysis (`[m, kg, s, A, K, mol, cd]` + `rad`) verifies equations, `sqrt`/`cbrt` parity (`E0210`), `Quat`/`Mat3` tensor transforms, scaled aerospace conversions (`km`, `cm`, `mm`, `ft`, `in`, `deg`, `rpm`, `kn`, `kN`, `bar`, `atm`, `g0`), and custom `unit` aliases at compile time, erasing all unit metadata before codegen (**zero runtime cost**).
2. **State Races & Order Coupling** — Double-buffered `step(prev) -> next` semantics enforce immutable `prev` reads (`E0301`) and write-before-read initialization of `next` (`E0302`, `W0301`), unlocking `noalias` SIMD vectorization and deterministic multi-core `rayon` Swarm scheduling.
3. **GC Jitter & Non-Deterministic Drift** — Heap allocations (`E0303`), recursion/mutual recursion (`E0304`), and unbounded loops (`while`/`loop` — `E0305`) are rejected inside hot kernels. Kernels compile to 32-byte aligned Structure-of-Arrays (SoA) buffers executed via **Cranelift pure-Rust JIT**, **LLVM 18+** (`"fp-contract"="off"`), or the **Rayon Swarm VM** (`f64` and `Q32.32` Fixed64) using pure-Rust `libm` math and canonical little-endian **SHA-256** `.aeroreplay` checkpoint rollback.

---

## Compiler Architecture

```text
  .aero Source
       │
       ├──► aerotick-lsp (stdio JSON-RPC: publishDiagnostics, SI Hover, Symbols)
       ▼
  [Phase 1: aerotick-lexer]        logos zero-copy DFA tokenizer + Spans (E0101)
       ▼
  [Phase 2: aerotick-parser]       chumsky 0.10 recursive parser + recovery (E0102)
       ▼
  [Phase 3: aerotick-sema]         7-dim SI + rad algebra, 6-DoF types, step safety
       ▼
  [Phase 4: aerotick-mir]          Unit erasure, 32B SoA layout + 4-pass optimizer:
                                     1. Pure Function Inlining
                                     2. Bounded Loop Unrolling (for i in 0..N)
                                     3. Non-Associative IEEE-754 Constant Folding (libm)
                                     4. CSE & Dead Code Elimination + 64B Cache-Line Stats
       │
       ├──────────────────────────────────┬──────────────────────────────────┐
       ▼                                  ▼                                  ▼
  [aerotick-codegen-cranelift]      [aerotick-codegen-llvm]           [aerotick-vm & aerotick-abi]
  Pure-Rust Native JIT              Strict IEEE-754 LLVM IR           Rayon Swarm VM + .aeroreplay
  (cranelift-jit 0.116 + libm)      + inkwell 0.11 native .o          + C11/C++ aerotick.h header
```

---

## Repository Structure

```text
/
├── Cargo.toml                         # Root Cargo Edition 2024 workspace (12 crates)
├── LICENSE                            # MIT License
├── README.md                          # Documentation
├── .gitignore                         # Rust build artifacts ignore rules
├── .github/workflows/ci.yml           # Linux/macOS/Windows CI matrix
├── crates/
│   ├── aerotick-span/                 # SourceMap, Span, line/col mapping, ariadne renderer
│   ├── aerotick-lexer/                # logos tokenizer (E0101)
│   ├── aerotick-ast/                  # Spanned AST (unit, const, state, fn, system, test)
│   ├── aerotick-parser/               # chumsky 0.10 parser + recovery (E0102)
│   ├── aerotick-sema/                 # 7-dim SI + rad verifier, 6-DoF & step safety, proptest
│   ├── aerotick-mir/                  # Unit-erased SoA MIR + 4-pass deterministic optimizer
│   ├── aerotick-codegen-llvm/         # Strict IEEE-754 LLVM IR + optional inkwell 0.11 .o
│   ├── aerotick-codegen-cranelift/    # Pure-Rust native JIT (cranelift-jit + libm)
│   ├── aerotick-vm/                   # Swarm SoA VM (f64 / Q32.32), rayon chunks, .aeroreplay
│   ├── aerotick-abi/                  # Portable C11/C++ aerotick.h generator (_Static_assert)
│   ├── aerotick-lsp/                  # JSON-RPC Language Server over stdio
│   └── aeroc/                         # Compiler CLI driver
├── examples/
│   ├── reference_body.aero            # Canonical rigid-body integrator (104 B/entity SoA)
│   ├── rocket_6dof.aero               # 6-DoF rigid-body rocket (Quat, Mat3, pure fn, tests)
│   ├── quadcopter_pid.aero            # PID altitude/attitude hover controller (rpm, clamp, tests)
│   └── invalid_physics.aero           # Compile-fail showcase (E0202, E0210, E0301, E0303, E0305)
└── bindings/
    ├── c_example/main.c               # Portable C11 double-buffered SoA host example
    └── godot_gdextension/README.md    # Godot 4 GDExtension C++ integration guide
```

---

## Language & Type System Highlights

- **6-DoF Primitives**: `Vec3<U>` (32B SIMD4 padded), `Quat` (32B unit quaternion `(w, x, y, z)` with Hamilton product `*`), and `Mat3<U>` (96B `3x4` padded inertia tensors & transforms with `Mat3<U1> * Vec3<U2> -> Vec3<U1*U2>`).
- **Deterministic Math & 6-DoF Intrinsics** (all transcendental math routed through pure-Rust `libm`):
  `sin`, `cos`, `tan`, `atan2`, `sqrt`, `cbrt`, `abs`, `min`, `max`, `clamp`, `select`, `dot`, `cross`, `norm`, `normalize`, `quat_conj`, `quat_norm`, `quat_normalize`, `quat_rotate`, `quat_integrate`, `mat3_inv`, `mat3_from_quat`, `mat3_transpose`, `mat3_diag`.
- **28 Built-In SI & Aerospace Units + Custom Aliases**:
  `m`, `km`, `cm`, `mm`, `ft`, `in`, `kg`, `g`, `s`, `ms`, `A`, `K`, `mol`, `cd`, `rad`, `deg`, `rpm`, `kn`, `N`, `kN`, `J`, `W`, `Pa`, `MPa`, `bar`, `atm`, `Hz`, `g0`, plus user-defined `unit Thrust = N;`, `unit Inertia = kg*m^2;`.
- **Pure Functions, Branchless Select & Bounded Loops**:
  Side-effect-free non-recursive `fn` helpers (statically inlined), deterministic `if cond { a } else { b }` expressions, statically bounded `for i in 0..N { ... }` loops, and built-in `test "name" { ... assert_approx(a, b, tol); }` verification blocks.

---

## Diagnostic Code Reference

| Code | Severity | Meaning |
| :--- | :--- | :--- |
| `E0101` | Error | Invalid lexical token |
| `E0102` | Error | Syntax error (recovered by `chumsky` parser) |
| `E0201` | Error | Unknown physical unit, symbol, function, or state field |
| `E0202` | Error | Dimensional mismatch (incompatible SI dimensions or angular units) |
| `E0203` | Error | Structural type mismatch (e.g., `Scalar` vs `Vec3` / `Quat` / `Mat3` / `Bool`) |
| `E0210` | Error | Non-integer resulting dimension in `sqrt` (must be even) or `cbrt` (must be divisible by 3) |
| `E0301` | Error | Attempted mutation of read-only `prev` state buffer inside `step` |
| `E0302` | Error | Reading an uninitialized `next` field before assigning it in the current tick |
| `E0303` | Error | Forbidden heap allocation (`alloc`/`malloc`/`Vec`/`Box`/`spawn`) inside `step` |
| `E0304` | Error | Direct or mutual recursion detected in pure `fn` call graph |
| `E0305` | Error | Unbounded loop (`while` / `loop`) forbidden inside deterministic kernels |
| `W0301` | Warning | State field never written to `next` inside `step` |

---

## Quick Start & Verification Commands

Run all commands directly from the repository root (`/`):

```bash
# 1. Formatting, strict Clippy (-D warnings, deny unwrap/expect), and full workspace tests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# 2. Type-check kernels & execute built-in physics test blocks
cargo run -p aeroc -- check examples/reference_body.aero
cargo run -p aeroc -- check examples/rocket_6dof.aero
cargo run -p aeroc -- check examples/quadcopter_pid.aero
cargo run -p aeroc -- test examples/rocket_6dof.aero
cargo run -p aeroc -- test examples/quadcopter_pid.aero

# 3. Inspect optimized SoA MIR + 64B cache-line report, strict LLVM IR, and C11 ABI header
cargo run -p aeroc -- emit-ir --opt --stats examples/rocket_6dof.aero
cargo run -p aeroc -- emit-ir --llvm examples/rocket_6dof.aero
cargo run -p aeroc -- emit-header examples/reference_body.aero -o bindings/c_example/aerotick.h

# 4. Run pure-Rust Cranelift JIT, multi-threaded Swarm VM, Benchmarks, and .aeroreplay Rollback
cargo run --release -p aeroc -- run examples/rocket_6dof.aero --jit --ticks 10000 --entities 1000
cargo run --release -p aeroc -- run examples/reference_body.aero --mode fixed64 --ticks 10000 --threads 4
cargo run --release -p aeroc -- bench examples/reference_body.aero --jit --entities 10000 --ticks 1000 --threads 4
cargo run --release -p aeroc -- replay examples/rocket_6dof.aero --ticks 10000 --checkpoint-every 1000 --verify-rollback --out rocket.aeroreplay

# 5. Format .aero source files & launch Language Server Protocol binary (stdio)
cargo run -p aeroc -- fmt --check examples/rocket_6dof.aero
cargo run -p aerotick-lsp
```

### Optional Native LLVM Object Emission (`--features llvm`)

With LLVM 18 installed on the host system, compile kernels directly to native PIC object files:

```bash
cargo run -p aeroc --features llvm -- emit-ir examples/reference_body.aero --llvm --obj body.o
cc -std=c11 -O3 bindings/c_example/main.c body.o -lm -o aerotick_c_host
./aerotick_c_host
```

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for full details.
