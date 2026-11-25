import workspaceCargo from '../../aerotick/Cargo.toml?raw';
import spanLib from '../../aerotick/crates/aerotick-span/src/lib.rs?raw';
import lexerLib from '../../aerotick/crates/aerotick-lexer/src/lib.rs?raw';
import astLib from '../../aerotick/crates/aerotick-ast/src/lib.rs?raw';
import parserLib from '../../aerotick/crates/aerotick-parser/src/lib.rs?raw';
import semaLib from '../../aerotick/crates/aerotick-sema/src/lib.rs?raw';
import mirLib from '../../aerotick/crates/aerotick-mir/src/lib.rs?raw';
import codegenLib from '../../aerotick/crates/aerotick-codegen-llvm/src/lib.rs?raw';
import abiLib from '../../aerotick/crates/aerotick-abi/src/lib.rs?raw';
import vmLib from '../../aerotick/crates/aerotick-vm/src/lib.rs?raw';
import aerocMain from '../../aerotick/crates/aeroc/src/main.rs?raw';
import bench1m from '../../aerotick/crates/aeroc/examples/bench_1m.rs?raw';
import ciYml from '../../aerotick/.github/workflows/ci.yml?raw';
import readmeMd from '../../aerotick/README.md?raw';

export interface WorkspaceFileEntry {
  path: string;
  crate: string;
  milestone: string;
  summary: string;
  content: string;
}

export const RUST_WORKSPACE_FILES: WorkspaceFileEntry[] = [
  {
    path: 'aerotick/Cargo.toml',
    crate: 'workspace',
    milestone: 'M1–M4',
    summary: 'Cargo Edition 2024 workspace root with 10 crates and shared dependencies.',
    content: workspaceCargo,
  },
  {
    path: 'aerotick/crates/aeroc/src/main.rs',
    crate: 'aeroc',
    milestone: 'M1–M4',
    summary: 'Compiler CLI driver: check, emit-ir (--llvm / --obj), emit-header, run, bench.',
    content: aerocMain,
  },
  {
    path: 'aerotick/crates/aerotick-span/src/lib.rs',
    crate: 'aerotick-span',
    milestone: 'M1',
    summary: 'Source registry, byte Span spans, and Ariadne multi-span diagnostic renderer.',
    content: spanLib,
  },
  {
    path: 'aerotick/crates/aerotick-lexer/src/lib.rs',
    crate: 'aerotick-lexer',
    milestone: 'M1–M2',
    summary: 'Zero-copy logos 0.15 tokenizer for units, scientific notation, and kernel keywords.',
    content: lexerLib,
  },
  {
    path: 'aerotick/crates/aerotick-ast/src/lib.rs',
    crate: 'aerotick-ast',
    milestone: 'M1–M2',
    summary: 'Spanned AST types for const, state, system/step, Vec3<Unit>, and expressions.',
    content: astLib,
  },
  {
    path: 'aerotick/crates/aerotick-parser/src/lib.rs',
    crate: 'aerotick-parser',
    milestone: 'M1–M2',
    summary: 'Chumsky 0.10.1 parser with statement/item error recovery and unit disambiguation.',
    content: parserLib,
  },
  {
    path: 'aerotick/crates/aerotick-sema/src/lib.rs',
    crate: 'aerotick-sema',
    milestone: 'M1–M2',
    summary: '7-component SI algebra, Vec3 intrinsics, prev immutability, and zero-heap/loop verifier.',
    content: semaLib,
  },
  {
    path: 'aerotick/crates/aerotick-mir/src/lib.rs',
    crate: 'aerotick-mir',
    milestone: 'M2',
    summary: 'Unit-erased SoA MIR lowering and 32-byte aligned byte-offset layout calculator.',
    content: mirLib,
  },
  {
    path: 'aerotick/crates/aerotick-codegen-llvm/src/lib.rs',
    crate: 'aerotick-codegen-llvm',
    milestone: 'M3',
    summary: 'Strict IEEE-754 LLVM IR emitter (noalias, fp-contract=off) + optional inkwell codegen.',
    content: codegenLib,
  },
  {
    path: 'aerotick/crates/aerotick-abi/src/lib.rs',
    crate: 'aerotick-abi',
    milestone: 'M4',
    summary: 'C99/C++ aerotick.h header generator with 32B aligned AeroVec3Simd4 and FFI kernels.',
    content: abiLib,
  },
  {
    path: 'aerotick/crates/aerotick-vm/src/lib.rs',
    crate: 'aerotick-vm',
    milestone: 'M4',
    summary: 'Double-buffered SoA VM supporting Float64 / Q32.32 Fixed64 and SHA-256 replay hashes.',
    content: vmLib,
  },
  {
    path: 'aerotick/crates/aeroc/examples/bench_1m.rs',
    crate: 'aeroc',
    milestone: 'M4',
    summary: '1,000,000-tick deterministic replay benchmark proving bit-exact SHA-256 hashes.',
    content: bench1m,
  },
  {
    path: 'aerotick/.github/workflows/ci.yml',
    crate: 'ci',
    milestone: 'CI',
    summary: 'GitHub Actions workflow executing rustfmt, clippy -D warnings, cargo test, and bench.',
    content: ciYml,
  },
  {
    path: 'aerotick/README.md',
    crate: 'docs',
    milestone: 'Docs',
    summary: 'Architecture guide, syntax specification, crate map, and Cargo CLI documentation.',
    content: readmeMd,
  },
];
