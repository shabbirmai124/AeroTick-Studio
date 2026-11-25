# AeroTick — Deterministic Real-Time Physics & Flight Simulation Language

**AeroTick** (`.aero`) is a compiled, domain-specific systems language written in Rust for 3D kinematics, aerospace guidance, robotics, and deterministic multi-entity simulation. It eliminates unit-mismatch failures, mid-frame state races, and heap jitter at compile time while lowering directly to Structure-of-Arrays (SoA) memory layouts, strict IEEE-754 LLVM IR, and zero-overhead C-ABI kernels.

## Core Design Pillars

1. **First-Class SI Dimensional Analysis (Zero-Cost Abstraction)**
   Every quantity carries an exact 7-component SI exponent vector `[m, kg, s, A, K, mol, cd]` plus angular tracking `rad`. Adding `10.0 m + 5.0 kg` or assigning `m/s` to `m/s^2` fails compilation with `ariadne` multi-span diagnostics. After verification, all unit tags are erased before MIR lowering.
2. **Tick-Driven Double-Buffered Execution (`step(prev) -> next`)**
   `prev` is strictly immutable (`E0301`), `next` fields are write-only until assigned (`E0302`), and incomplete `next` writes emit `W0301`.
3. **Zero Heap Allocation & Bounded Hot Loops**
   Dynamic allocation (`alloc`, `malloc`, `Vec`, `Box`, `spawn`), recursion (`E0304`), and unbounded loops (`while`, `loop` — `E0305`) are rejected inside `step`.
4. **Cache-Aligned Structure-of-Arrays (SoA) & Strict Determinism**
   `Vec3<Unit>` fields pack into 32-byte aligned 4-lane SIMD vectors (`<x, y, z, _pad>`). LLVM codegen emits `noalias readonly %prev`, `noalias writeonly %next`, and `"fp-contract"="off"` to prevent FMA contraction drift across Intel/ARM hosts. Pure-Rust VM supports both strict `Float64` and `Q32.32 Fixed64` modes with SHA-256 bit-exact replay hashes.

## Compiler Pipeline Architecture

```
  [.aero Source]
        │
        ▼  Phase 1: aerotick-lexer (logos 0.15)
  [Spanned Tokens]
        │
        ▼  Phase 2: aerotick-parser (chumsky 0.10 + error recovery)
  [Untyped AST]
        │
        ▼  Phase 3: aerotick-sema (SI DimVector + step verifier + ariadne diagnostics)
  [Typed HIR]
        │
        ▼  Phase 4: aerotick-mir (unit erasure + 32B SoA byte-layout calculator)
  [SoA MIR]
        ├───────────────────────────────────────┬──────────────────────────┐
        ▼  Phase 5: aerotick-codegen-llvm       ▼  Phase 6: aerotick-abi   ▼  aerotick-vm
  [LLVM IR / inkwell Native .o]           [aerotick.h C99/C++ FFI]   [Deterministic VM + SHA-256]
```

## Example `.aero` Kernel

```aero
const GRAVITY: m/s^2 = 9.80665 m/s^2;

state Body {
    pos: Vec3<m>;
    vel: Vec3<m/s>;
    thrust: Vec3<N>;
    mass: kg;
}

system Integrate(dt: s) {
    step(prev: Body) -> next: Body {
        let accel_y: m/s^2 = (prev.thrust.y / prev.mass) - GRAVITY;
        let accel: Vec3<m/s^2> = Vec3(prev.thrust.x / prev.mass, accel_y, prev.thrust.z / prev.mass);
        next.vel = prev.vel + accel * dt;
        next.pos = prev.pos + prev.vel * dt;
        next.thrust = prev.thrust;
        next.mass = prev.mass;
    }
}
```

## Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `aerotick-span` | File registry, byte `Span`, and `ariadne` multi-span diagnostic renderer |
| `aerotick-lexer` | `logos`-based zero-copy tokenizer with comment skipping |
| `aerotick-ast` | Spanned AST (`const`, `state`, `system`, `step`, `Vec3<Unit>`) |
| `aerotick-parser` | `chumsky` Pratt/recursive-descent parser with statement & item recovery |
| `aerotick-sema` | 7-component SI algebra, `Vec3` intrinsics, `prev`/`next` & heap/loop checks |
| `aerotick-mir` | Unit-erased SoA MIR lowering & byte-offset/padding calculator |
| `aerotick-codegen-llvm` | Strict IEEE-754 LLVM IR emitter + optional `--features llvm` `inkwell` codegen |
| `aerotick-abi` | C99/C++ `.h` header generator (`AeroVec3Simd4`, `<State>_SoA`, `aerotick_step_*`) |
| `aerotick-vm` | Pure-Rust double-buffered SoA VM (`Float64` / `Q32.32 Fixed64`) + SHA-256 hash |
| `aeroc` | Compiler driver CLI (`check`, `emit-ir`, `emit-header`, `run`, `bench`) |

## Build & CLI Usage

```bash
cd aerotick

# Run formatter, linter, and all workspace unit & compile-fail tests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Check physical dimensions & step safety
cargo run -p aeroc -- check examples/reference_body.aero

# Inspect unit-erased SoA MIR or deterministic LLVM IR
cargo run -p aeroc -- emit-ir examples/reference_body.aero
cargo run -p aeroc -- emit-ir examples/reference_body.aero --llvm

# Generate C-ABI header (aerotick.h)
cargo run -p aeroc -- emit-header examples/reference_body.aero

# Execute deterministic simulation & verify 1,000,000-tick SHA-256 replay hash
cargo run --release -p aeroc -- run examples/reference_body.aero --ticks 1000 --dt 0.01
cargo run --release -p aeroc -- bench examples/reference_body.aero --ticks 1000000

# Optional native LLVM object emission via inkwell (requires LLVM 18–23 dev libs)
cargo run -p aeroc --features llvm -- emit-ir examples/reference_body.aero --llvm --obj build/body.o
```
