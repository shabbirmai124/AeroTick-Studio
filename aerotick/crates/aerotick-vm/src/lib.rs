//! Pure-Rust Deterministic Double-Buffered SoA Execution Engine (VM Fallback).
//! Executes compiled `MirProgram` kernels on contiguous SoA buffers with both
//! strict IEEE-754 `f64` and Q32.32 `fixed64` numeric modes, plus SHA-256 bit-exact replay hashes.

use aerotick_ast::BinOp;
use aerotick_mir::{MirInst, MirProgram, MirSystem, MirType, SoAStateLayout};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericMode {
    Float64,
    Fixed64,
}

#[derive(Debug, Error)]
pub enum VmError {
    #[error("MIR program has no runnable system")]
    NoSystem,
    #[error("State layout '{0}' missing")]
    MissingLayout(String),
}

const Q32_SHIFT: u32 = 32;
const Q32_SCALE: f64 = 4_294_967_296.0;

#[inline]
fn to_fixed64(v: f64) -> i64 {
    if !v.is_finite() {
        return 0;
    }
    (v * Q32_SCALE).round() as i64
}

#[inline]
fn from_fixed64(q: i64) -> f64 {
    (q as f64) / Q32_SCALE
}

#[inline]
fn mul_fixed64(a: i64, b: i64) -> i64 {
    ((i128::from(a) * i128::from(b)) >> Q32_SHIFT) as i64
}

#[inline]
fn div_fixed64(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    ((i128::from(a) << Q32_SHIFT) / i128::from(b)) as i64
}

#[derive(Debug, Clone)]
enum RegVal {
    Scalar(f64),
    Vec3([f64; 3]),
}

impl RegVal {
    fn as_scalar(&self) -> f64 {
        match self {
            Self::Scalar(v) => *v,
            Self::Vec3(v) => v[0],
        }
    }

    fn as_vec3(&self) -> [f64; 3] {
        match self {
            Self::Vec3(v) => *v,
            Self::Scalar(v) => [*v, *v, *v],
        }
    }
}

pub struct SoABufferSet {
    pub fields: HashMap<String, Vec<f64>>,
}

pub struct DeterministicVm {
    system: MirSystem,
    layout: SoAStateLayout,
    pub count: usize,
    pub dt: f64,
    pub mode: NumericMode,
    prev_buf: SoABufferSet,
    next_buf: SoABufferSet,
    pub tick: u64,
    pub sim_time: f64,
}

impl DeterministicVm {
    pub fn new(
        mir: &MirProgram,
        count: usize,
        dt: f64,
        mode: NumericMode,
    ) -> Result<Self, VmError> {
        let system = mir.systems.first().ok_or(VmError::NoSystem)?.clone();
        let layout = mir
            .layouts
            .iter()
            .find(|l| l.state_name == system.state_name)
            .ok_or_else(|| VmError::MissingLayout(system.state_name.clone()))?
            .clone();

        let count = count.max(1);
        let prev_buf = alloc_buf(&layout, count);
        let next_buf = alloc_buf(&layout, count);

        let mut vm = Self {
            system,
            layout,
            count,
            dt,
            mode,
            prev_buf,
            next_buf,
            tick: 0,
            sim_time: 0.0,
        };
        vm.seed_initial_state();
        Ok(vm)
    }

    pub fn seed_initial_state(&mut self) {
        self.tick = 0;
        self.sim_time = 0.0;
        let is_orbital = self.layout.state_name.to_lowercase().contains("orbit");
        let is_rocket = self
            .layout
            .fields
            .iter()
            .any(|f| f.name == "fuel_rate" || f.name == "drag_k");

        for i in 0..self.count {
            let spread = if i == 0 { 0.0 } else { ((i % 16) as f64) * 0.5 };
            for f in &self.layout.fields {
                if let Some(arr) = self.prev_buf.fields.get_mut(&f.name) {
                    let base = i * f.lanes;
                    if f.mir_type == MirType::Vec3Simd4 {
                        match f.name.as_str() {
                            "pos" => {
                                arr[base] = if is_orbital { 6_771_000.0 } else { spread };
                                arr[base + 1] = if is_orbital || is_rocket { 0.0 } else { 120.0 + spread };
                                arr[base + 2] = 0.0;
                            }
                            "vel" => {
                                arr[base] = if is_orbital { 0.0 } else { 28.0 + ((i % 5) as f64) * 1.2 };
                                arr[base + 1] = if is_orbital { 7_672.0 } else if is_rocket { 0.0 } else { 42.0 };
                                arr[base + 2] = 0.0;
                            }
                            "thrust" => {
                                arr[base] = 0.0;
                                arr[base + 1] = if is_rocket { 24_000.0 } else { 1_180.0 };
                                arr[base + 2] = 0.0;
                            }
                            _ => {
                                arr[base] = 0.0;
                                arr[base + 1] = 0.0;
                                arr[base + 2] = 0.0;
                            }
                        }
                        arr[base + 3] = 0.0;
                    } else {
                        arr[base] = match f.name.as_str() {
                            "mass" => if is_rocket { 1_200.0 } else { 100.0 },
                            "fuel_rate" => 4.2,
                            "drag_k" => 0.32,
                            "mu" => 3.986_004_418e14,
                            _ => 1.0,
                        };
                    }
                }
            }
        }
    }

    pub fn step_tick(&mut self) {
        let mut regs: HashMap<String, RegVal> = HashMap::new();
        for idx in 0..self.count {
            regs.clear();
            exec_insts(
                &self.system.insts,
                idx,
                self.dt,
                self.mode,
                &self.prev_buf,
                &mut self.next_buf,
                &mut regs,
            );
        }
        std::mem::swap(&mut self.prev_buf, &mut self.next_buf);
        self.tick += 1;
        self.sim_time += self.dt;
    }

    /// Computes a bit-exact SHA-256 hash over all raw IEEE-754 little-endian bytes of `prev_buf`.
    pub fn state_sha256_hex(&self) -> String {
        let mut hasher = Sha256::new();
        for f in &self.layout.fields {
            if let Some(arr) = self.prev_buf.fields.get(&f.name) {
                for &val in arr {
                    let canonical = if val == 0.0 { 0.0 } else { val };
                    hasher.update(canonical.to_le_bytes());
                }
            }
        }
        let digest = hasher.finalize();
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn read_entity0_vec3(&self, field: &str) -> Option<[f64; 3]> {
        self.prev_buf
            .fields
            .get(field)
            .filter(|v| v.len() >= 3)
            .map(|v| [v[0], v[1], v[2]])
    }
}

fn alloc_buf(layout: &SoAStateLayout, count: usize) -> SoABufferSet {
    let mut fields = HashMap::new();
    for f in &layout.fields {
        fields.insert(f.name.clone(), vec![0.0; count * f.lanes]);
    }
    SoABufferSet { fields }
}

fn scalar_binop(a: f64, b: f64, op: BinOp, mode: NumericMode) -> f64 {
    if mode == NumericMode::Fixed64 {
        let qa = to_fixed64(a);
        let qb = to_fixed64(b);
        return match op {
            BinOp::Add => from_fixed64(qa.wrapping_add(qb)),
            BinOp::Sub => from_fixed64(qa.wrapping_sub(qb)),
            BinOp::Mul => from_fixed64(mul_fixed64(qa, qb)),
            BinOp::Div => from_fixed64(div_fixed64(qa, qb)),
            BinOp::Pow => a.powi(b as i32),
            BinOp::Eq => if qa == qb { 1.0 } else { 0.0 },
            BinOp::Ne => if qa != qb { 1.0 } else { 0.0 },
            BinOp::Lt => if qa < qb { 1.0 } else { 0.0 },
            BinOp::Le => if qa <= qb { 1.0 } else { 0.0 },
            BinOp::Gt => if qa > qb { 1.0 } else { 0.0 },
            BinOp::Ge => if qa >= qb { 1.0 } else { 0.0 },
        };
    }

    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => if b == 0.0 { 0.0 } else { a / b },
        BinOp::Pow => a.powi(b as i32),
        BinOp::Eq => if a == b { 1.0 } else { 0.0 },
        BinOp::Ne => if a != b { 1.0 } else { 0.0 },
        BinOp::Lt => if a < b { 1.0 } else { 0.0 },
        BinOp::Le => if a <= b { 1.0 } else { 0.0 },
        BinOp::Gt => if a > b { 1.0 } else { 0.0 },
        BinOp::Ge => if a >= b { 1.0 } else { 0.0 },
    }
}

fn exec_insts(
    insts: &[MirInst],
    idx: usize,
    dt: f64,
    mode: NumericMode,
    prev: &SoABufferSet,
    next: &mut SoABufferSet,
    regs: &mut HashMap<String, RegVal>,
) {
    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => {
                regs.insert(dst.clone(), RegVal::Scalar(*value));
            }
            MirInst::LoadParam { dst, .. } => {
                regs.insert(dst.clone(), RegVal::Scalar(dt));
            }
            MirInst::ConstVec3 { dst, x, y, z } => {
                let vx = regs.get(x).map_or(0.0, RegVal::as_scalar);
                let vy = regs.get(y).map_or(0.0, RegVal::as_scalar);
                let vz = regs.get(z).map_or(0.0, RegVal::as_scalar);
                regs.insert(dst.clone(), RegVal::Vec3([vx, vy, vz]));
            }
            MirInst::LoadSoA { dst, field, is_next, mir_type, .. } => {
                let buf = if *is_next { &*next } else { prev };
                if let Some(arr) = buf.fields.get(field) {
                    if *mir_type == MirType::Vec3Simd4 {
                        let base = idx * 4;
                        regs.insert(dst.clone(), RegVal::Vec3([arr[base], arr[base + 1], arr[base + 2]]));
                    } else {
                        regs.insert(dst.clone(), RegVal::Scalar(arr[idx]));
                    }
                }
            }
            MirInst::StoreSoA { field, src, mir_type, .. } => {
                if let (Some(arr), Some(val)) = (next.fields.get_mut(field), regs.get(src)) {
                    if *mir_type == MirType::Vec3Simd4 {
                        let v = val.as_vec3();
                        let base = idx * 4;
                        arr[base] = v[0];
                        arr[base + 1] = v[1];
                        arr[base + 2] = v[2];
                        arr[base + 3] = 0.0;
                    } else {
                        arr[idx] = val.as_scalar();
                    }
                }
            }
            MirInst::VecExtract { dst, vec, lane } => {
                if let Some(v) = regs.get(vec) {
                    let arr = v.as_vec3();
                    regs.insert(dst.clone(), RegVal::Scalar(arr[*lane]));
                }
            }
            MirInst::UnaryNeg { dst, src, .. } => {
                if let Some(val) = regs.get(src) {
                    let negated = match val {
                        RegVal::Scalar(s) => RegVal::Scalar(-s),
                        RegVal::Vec3(v) => RegVal::Vec3([-v[0], -v[1], -v[2]]),
                    };
                    regs.insert(dst.clone(), negated);
                }
            }
            MirInst::BinOp { dst, op, lhs, rhs, .. } => {
                if let (Some(l), Some(r)) = (regs.get(lhs), regs.get(rhs)) {
                    let out = match (l, r) {
                        (RegVal::Vec3(a), RegVal::Vec3(b)) => RegVal::Vec3([
                            scalar_binop(a[0], b[0], *op, mode),
                            scalar_binop(a[1], b[1], *op, mode),
                            scalar_binop(a[2], b[2], *op, mode),
                        ]),
                        (RegVal::Vec3(a), RegVal::Scalar(b)) => RegVal::Vec3([
                            scalar_binop(a[0], *b, *op, mode),
                            scalar_binop(a[1], *b, *op, mode),
                            scalar_binop(a[2], *b, *op, mode),
                        ]),
                        (RegVal::Scalar(a), RegVal::Vec3(b)) => RegVal::Vec3([
                            scalar_binop(*a, b[0], *op, mode),
                            scalar_binop(*a, b[1], *op, mode),
                            scalar_binop(*a, b[2], *op, mode),
                        ]),
                        (RegVal::Scalar(a), RegVal::Scalar(b)) => {
                            RegVal::Scalar(scalar_binop(*a, *b, *op, mode))
                        }
                    };
                    regs.insert(dst.clone(), out);
                }
            }
            MirInst::Call { dst, intrinsic, args, .. } => {
                let vals: Vec<RegVal> = args.iter().filter_map(|a| regs.get(a).cloned()).collect();
                let res = match (intrinsic.as_str(), vals.as_slice()) {
                    ("norm", [RegVal::Vec3(v)]) => {
                        RegVal::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                    }
                    ("normalize", [RegVal::Vec3(v)]) => {
                        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        if m == 0.0 { RegVal::Vec3([0.0; 3]) } else { RegVal::Vec3([v[0] / m, v[1] / m, v[2] / m]) }
                    }
                    ("dot", [RegVal::Vec3(a), RegVal::Vec3(b)]) => {
                        RegVal::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
                    }
                    ("cross", [RegVal::Vec3(a), RegVal::Vec3(b)]) => RegVal::Vec3([
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]),
                    ("sqrt", [RegVal::Scalar(x)]) => RegVal::Scalar(x.max(0.0).sqrt()),
                    ("sin", [RegVal::Scalar(x)]) => RegVal::Scalar(x.sin()),
                    ("cos", [RegVal::Scalar(x)]) => RegVal::Scalar(x.cos()),
                    ("abs", [RegVal::Scalar(x)]) => RegVal::Scalar(x.abs()),
                    ("min", [RegVal::Scalar(a), RegVal::Scalar(b)]) => RegVal::Scalar(a.min(*b)),
                    ("max", [RegVal::Scalar(a), RegVal::Scalar(b)]) => RegVal::Scalar(a.max(*b)),
                    ("select", [RegVal::Scalar(c), t, f]) => if *c != 0.0 { t.clone() } else { f.clone() },
                    _ => RegVal::Scalar(0.0),
                };
                regs.insert(dst.clone(), res);
            }
            MirInst::Branch { cond, then_insts, else_insts } => {
                let c = regs.get(cond).is_some_and(|v| v.as_scalar() != 0.0);
                if c {
                    exec_insts(then_insts, idx, dt, mode, prev, next, regs);
                } else if let Some(el) = else_insts {
                    exec_insts(el, idx, dt, mode, prev, next, regs);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_mir::lower_to_mir;
    use aerotick_parser::parse;
    use aerotick_sema::analyze;

    #[test]
    fn executes_1000_ticks_deterministically_with_identical_sha256() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, _) = lex(src);
        let (ast, _) = parse(&tokens, src.len());
        let (hir, _) = analyze(&ast.expect("ast"));
        let mir = lower_to_mir(&hir);

        let mut vm1 = DeterministicVm::new(&mir, 1, 0.01, NumericMode::Float64).expect("vm1");
        let mut vm2 = DeterministicVm::new(&mir, 1, 0.01, NumericMode::Float64).expect("vm2");

        for _ in 0..1000 {
            vm1.step_tick();
            vm2.step_tick();
        }
        assert_eq!(vm1.state_sha256_hex(), vm2.state_sha256_hex());
        let vel = vm1.read_entity0_vec3("vel").expect("vel");
        assert!((vel[1] - 61.9335).abs() < 1e-9);
    }
}
