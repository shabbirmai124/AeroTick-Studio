#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Pure-Rust Deterministic Double-Buffered SoA Execution Engine (Swarm VM).
//! Executes compiled `MirProgram` kernels on contiguous SoA buffers with:
//! - Strict IEEE-754 `Float64` and Q32.32 `Fixed64` numeric modes
//! - `libm`-backed deterministic math intrinsics (`sin`, `cos`, `tan`, `atan2`, `sqrt`, `cbrt`)
//! - 6-DoF `Quat` & `Mat3` physics primitives
//! - Deterministic `rayon` multi-threaded chunk execution with bit-exact SHA-256 parity
//! - Binary snapshot & rollback verification (`VmCheckpoint`)
//! - Built-in `test` block runner (`assert_approx`)

use aerotick_ast::BinOp;
use aerotick_mir::{MirInst, MirProgram, MirSystem, MirType, SoAStateLayout};
use rayon::prelude::*;
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
    #[error("Checkpoint state layout mismatch")]
    CheckpointMismatch,
    #[error("Failed to build rayon thread pool: {0}")]
    ThreadPool(String),
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
pub enum RegVal {
    Scalar(f64),
    Vec3([f64; 3]),
    Quat([f64; 4]),
    Mat3([f64; 9]),
}

impl RegVal {
    fn as_scalar(&self) -> f64 {
        match self {
            Self::Scalar(v) => *v,
            Self::Vec3(v) => v[0],
            Self::Quat(q) => q[0],
            Self::Mat3(m) => m[0],
        }
    }

    fn as_vec3(&self) -> [f64; 3] {
        match self {
            Self::Vec3(v) => *v,
            Self::Scalar(v) => [*v, *v, *v],
            Self::Quat(q) => [q[1], q[2], q[3]],
            Self::Mat3(m) => [m[0], m[1], m[2]],
        }
    }

    fn as_quat(&self) -> [f64; 4] {
        match self {
            Self::Quat(q) => *q,
            Self::Vec3(v) => [0.0, v[0], v[1], v[2]],
            Self::Scalar(s) => [*s, 0.0, 0.0, 0.0],
            Self::Mat3(_) => [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn as_mat3(&self) -> [f64; 9] {
        match self {
            Self::Mat3(m) => *m,
            Self::Scalar(s) => [*s, 0.0, 0.0, 0.0, *s, 0.0, 0.0, 0.0, *s],
            _ => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoABufferSet {
    pub fields: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct VmCheckpoint {
    pub tick: u64,
    pub sim_time: f64,
    pub count: usize,
    pub state_name: String,
    pub raw_bytes: Vec<u8>,
    pub sha256_hex: String,
}

const REPLAY_MAGIC: &[u8; 8] = b"AERORPL1";

impl VmCheckpoint {
    /// Serializes this checkpoint into a deterministic little-endian `.aeroreplay` binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.state_name.as_bytes();
        let mut out = Vec::with_capacity(
            8 + 8 + 8 + 8 + 4 + name_bytes.len() + 8 + self.raw_bytes.len() + 64,
        );
        out.extend_from_slice(REPLAY_MAGIC);
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.sim_time.to_le_bytes());
        out.extend_from_slice(&(self.count as u64).to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(self.raw_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.raw_bytes);
        out.extend_from_slice(self.sha256_hex.as_bytes());
        out
    }

    /// Deserializes and SHA-256 verifies a `.aeroreplay` binary blob.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VmError> {
        let mut cursor = 0usize;
        let read_slice = |len: usize, pos: &mut usize| -> Result<&[u8], VmError> {
            let end = pos.checked_add(len).ok_or(VmError::CheckpointMismatch)?;
            if end > bytes.len() {
                return Err(VmError::CheckpointMismatch);
            }
            let slice = &bytes[*pos..end];
            *pos = end;
            Ok(slice)
        };

        if read_slice(8, &mut cursor)? != REPLAY_MAGIC {
            return Err(VmError::CheckpointMismatch);
        }
        let tick = u64::from_le_bytes(
            read_slice(8, &mut cursor)?
                .try_into()
                .map_err(|_| VmError::CheckpointMismatch)?,
        );
        let sim_time = f64::from_le_bytes(
            read_slice(8, &mut cursor)?
                .try_into()
                .map_err(|_| VmError::CheckpointMismatch)?,
        );
        let count = u64::from_le_bytes(
            read_slice(8, &mut cursor)?
                .try_into()
                .map_err(|_| VmError::CheckpointMismatch)?,
        ) as usize;
        let name_len = u32::from_le_bytes(
            read_slice(4, &mut cursor)?
                .try_into()
                .map_err(|_| VmError::CheckpointMismatch)?,
        ) as usize;
        let state_name = String::from_utf8(read_slice(name_len, &mut cursor)?.to_vec())
            .map_err(|_| VmError::CheckpointMismatch)?;
        let raw_len = u64::from_le_bytes(
            read_slice(8, &mut cursor)?
                .try_into()
                .map_err(|_| VmError::CheckpointMismatch)?,
        ) as usize;
        let raw_bytes = read_slice(raw_len, &mut cursor)?.to_vec();
        let sha_bytes = read_slice(64, &mut cursor)?;
        let sha256_hex =
            String::from_utf8(sha_bytes.to_vec()).map_err(|_| VmError::CheckpointMismatch)?;
        if hash_bytes_hex(&raw_bytes) != sha256_hex {
            return Err(VmError::CheckpointMismatch);
        }
        Ok(Self {
            tick,
            sim_time,
            count,
            state_name,
            raw_bytes,
            sha256_hex,
        })
    }
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
        let is_quad = self
            .layout
            .fields
            .iter()
            .any(|f| f.name == "altitude" || f.name == "integral_err");

        for i in 0..self.count {
            let spread = if i == 0 { 0.0 } else { ((i % 16) as f64) * 0.5 };
            for f in &self.layout.fields {
                if let Some(arr) = self.prev_buf.fields.get_mut(&f.name) {
                    let base = i * f.lanes;
                    match f.mir_type {
                        MirType::Vec3Simd4 => {
                            match f.name.as_str() {
                                "pos" => {
                                    arr[base] = if is_orbital { 6_771_000.0 } else { spread };
                                    arr[base + 1] = if is_orbital || is_rocket {
                                        0.0
                                    } else {
                                        120.0 + spread
                                    };
                                    arr[base + 2] = 0.0;
                                }
                                "vel" => {
                                    arr[base] = if is_orbital {
                                        0.0
                                    } else {
                                        28.0 + ((i % 5) as f64) * 1.2
                                    };
                                    arr[base + 1] = if is_orbital {
                                        7_672.0
                                    } else if is_rocket {
                                        0.0
                                    } else {
                                        42.0
                                    };
                                    arr[base + 2] = 0.0;
                                }
                                "thrust" => {
                                    arr[base] = 0.0;
                                    arr[base + 1] = if is_rocket { 24_000.0 } else { 1_180.0 };
                                    arr[base + 2] = 0.0;
                                }
                                "omega" => {
                                    arr[base] = 0.02 * ((i % 3) as f64);
                                    arr[base + 1] = 0.0;
                                    arr[base + 2] = 0.05;
                                }
                                _ => {
                                    arr[base] = 0.0;
                                    arr[base + 1] = 0.0;
                                    arr[base + 2] = 0.0;
                                }
                            }
                            arr[base + 3] = 0.0;
                        }
                        MirType::QuatSimd4 => {
                            arr[base] = 1.0;
                            arr[base + 1] = 0.0;
                            arr[base + 2] = 0.0;
                            arr[base + 3] = 0.0;
                        }
                        MirType::Mat3Simd12 => {
                            // Identity / inertia diagonal stored in row-major 3x4 lanes
                            arr[base] = 120.0;
                            arr[base + 5] = 120.0;
                            arr[base + 10] = 45.0;
                        }
                        _ => {
                            arr[base] = match f.name.as_str() {
                                "mass" => if is_rocket {
                                    1_200.0
                                } else if is_quad {
                                    1.8
                                } else {
                                    100.0
                                },
                                "fuel_rate" => 4.2,
                                "drag_k" => 0.32,
                                "cd" => 0.45,
                                "area" => 1.2,
                                "rho" => 1.225,
                                "mu" => 3.986_004_418e14,
                                "altitude" => 1.0,
                                "target_alt" => 10.0,
                                "vz" | "integral_err" | "prev_err" => 0.0,
                                _ => 1.0,
                            };
                        }
                    }
                }
            }
        }
    }

    pub fn step_tick(&mut self) {
        let mut regs: HashMap<String, RegVal> = HashMap::new();
        for idx in 0..self.count {
            regs.clear();
            exec_insts_into(
                &self.system.insts,
                idx,
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

    /// Deterministic multi-threaded chunk execution using `rayon`.
    /// Each chunk computes disjoint `[start..end)` entity slices from immutable `prev_buf`
    /// and merges back in deterministic index order, yielding the exact same SHA-256 as `step_tick`.
    pub fn step_tick_parallel(&mut self, threads: usize) -> Result<(), VmError> {
        if threads <= 1 || self.count < 64 {
            self.step_tick();
            return Ok(());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| VmError::ThreadPool(e.to_string()))?;

        let chunk_size = self.count.div_ceil(threads).max(1);
        let chunks: Vec<(usize, usize)> = (0..self.count)
            .step_by(chunk_size)
            .map(|start| (start, (start + chunk_size).min(self.count)))
            .collect();

        let layout = &self.layout;
        let system = &self.system;
        let prev = &self.prev_buf;
        let dt = self.dt;
        let mode = self.mode;

        let chunk_results: Vec<(usize, SoABufferSet)> = pool.install(|| {
            chunks
                .into_par_iter()
                .map(|(start, end)| {
                    let local_len = end - start;
                    let mut local_next = alloc_buf(layout, local_len);
                    let mut regs: HashMap<String, RegVal> = HashMap::new();
                    for local_idx in 0..local_len {
                        let global_idx = start + local_idx;
                        regs.clear();
                        exec_insts_into(
                            &system.insts,
                            global_idx,
                            local_idx,
                            dt,
                            mode,
                            prev,
                            &mut local_next,
                            &mut regs,
                        );
                    }
                    (start, local_next)
                })
                .collect()
        });

        for (start, local_next) in chunk_results {
            for f in &self.layout.fields {
                if let (Some(dst), Some(src)) = (
                    self.next_buf.fields.get_mut(&f.name),
                    local_next.fields.get(&f.name),
                ) {
                    let offset = start * f.lanes;
                    dst[offset..offset + src.len()].copy_from_slice(src);
                }
            }
        }

        std::mem::swap(&mut self.prev_buf, &mut self.next_buf);
        self.tick += 1;
        self.sim_time += self.dt;
        Ok(())
    }

    pub fn save_checkpoint(&self) -> VmCheckpoint {
        let mut raw_bytes = Vec::new();
        for f in &self.layout.fields {
            if let Some(arr) = self.prev_buf.fields.get(&f.name) {
                for &val in arr {
                    let canonical = if val == 0.0 { 0.0 } else { val };
                    raw_bytes.extend_from_slice(&canonical.to_le_bytes());
                }
            }
        }
        let sha256_hex = hash_bytes_hex(&raw_bytes);
        VmCheckpoint {
            tick: self.tick,
            sim_time: self.sim_time,
            count: self.count,
            state_name: self.layout.state_name.clone(),
            raw_bytes,
            sha256_hex,
        }
    }

    pub fn restore_checkpoint(&mut self, cp: &VmCheckpoint) -> Result<(), VmError> {
        if cp.state_name != self.layout.state_name || cp.count != self.count {
            return Err(VmError::CheckpointMismatch);
        }
        let mut offset = 0usize;
        for f in &self.layout.fields {
            if let Some(arr) = self.prev_buf.fields.get_mut(&f.name) {
                for slot in arr.iter_mut() {
                    if offset + 8 > cp.raw_bytes.len() {
                        return Err(VmError::CheckpointMismatch);
                    }
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&cp.raw_bytes[offset..offset + 8]);
                    *slot = f64::from_le_bytes(bytes);
                    offset += 8;
                }
            }
        }
        self.tick = cp.tick;
        self.sim_time = cp.sim_time;
        Ok(())
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

    pub fn read_entity0_quat(&self, field: &str) -> Option<[f64; 4]> {
        self.prev_buf
            .fields
            .get(field)
            .filter(|v| v.len() >= 4)
            .map(|v| [v[0], v[1], v[2], v[3]])
    }

    pub fn read_entity0_scalar(&self, field: &str) -> Option<f64> {
        self.prev_buf
            .fields
            .get(field)
            .and_then(|v| v.first().copied())
    }

    /// Returns raw field pointer tables for native Cranelift JIT execution.
    pub fn field_ptrs_mut(&mut self) -> (Vec<*const f64>, Vec<*mut f64>) {
        let mut prev_ptrs = Vec::with_capacity(self.layout.fields.len());
        let mut next_ptrs = Vec::with_capacity(self.layout.fields.len());
        for f in &self.layout.fields {
            if let Some(arr) = self.prev_buf.fields.get(&f.name) {
                prev_ptrs.push(arr.as_ptr());
            }
            if let Some(arr) = self.next_buf.fields.get_mut(&f.name) {
                next_ptrs.push(arr.as_mut_ptr());
            }
        }
        (prev_ptrs, next_ptrs)
    }

    pub fn commit_jit_tick(&mut self) {
        std::mem::swap(&mut self.prev_buf, &mut self.next_buf);
        self.tick += 1;
        self.sim_time += self.dt;
    }
}

fn hash_bytes_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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
            BinOp::Pow => libm::pow(a, b),
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
        BinOp::Pow => libm::pow(a, b),
        BinOp::Eq => if a == b { 1.0 } else { 0.0 },
        BinOp::Ne => if a != b { 1.0 } else { 0.0 },
        BinOp::Lt => if a < b { 1.0 } else { 0.0 },
        BinOp::Le => if a <= b { 1.0 } else { 0.0 },
        BinOp::Gt => if a > b { 1.0 } else { 0.0 },
        BinOp::Ge => if a >= b { 1.0 } else { 0.0 },
    }
}

#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline]
fn quat_norm(q: [f64; 4]) -> [f64; 4] {
    let n = libm::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if n == 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

#[inline]
fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let qn = quat_norm(q);
    let q_v = [0.0, v[0], v[1], v[2]];
    let q_conj = [qn[0], -qn[1], -qn[2], -qn[3]];
    let tmp = quat_mul(qn, q_v);
    let out = quat_mul(tmp, q_conj);
    [out[1], out[2], out[3]]
}

#[inline]
fn quat_integrate(q: [f64; 4], omega: [f64; 3], dt: f64) -> [f64; 4] {
    let omega_q = [0.0, omega[0], omega[1], omega[2]];
    let dq = quat_mul(q, omega_q);
    let half_dt = 0.5 * dt;
    quat_norm([
        q[0] + dq[0] * half_dt,
        q[1] + dq[1] * half_dt,
        q[2] + dq[2] * half_dt,
        q[3] + dq[3] * half_dt,
    ])
}

#[inline]
fn mat3_mul_vec3(m: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

#[inline]
fn mat3_mul_mat3(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] =
                a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    out
}

#[inline]
fn mat3_inv(m: [f64; 9]) -> [f64; 9] {
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = m[5] * m[6] - m[3] * m[8];
    let c02 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if det.abs() < 1e-18 {
        return [0.0; 9];
    }
    let inv_det = 1.0 / det;
    [
        c00 * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        c01 * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        c02 * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

#[allow(clippy::too_many_arguments)]
fn exec_insts_into(
    insts: &[MirInst],
    read_idx: usize,
    write_idx: usize,
    dt: f64,
    mode: NumericMode,
    prev: &SoABufferSet,
    next: &mut SoABufferSet,
    regs: &mut HashMap<String, RegVal>,
) {
    let _ = exec_insts_with_assertions(insts, read_idx, write_idx, dt, mode, prev, next, regs);
}

#[allow(clippy::too_many_arguments)]
fn exec_insts_with_assertions(
    insts: &[MirInst],
    read_idx: usize,
    write_idx: usize,
    dt: f64,
    mode: NumericMode,
    prev: &SoABufferSet,
    next: &mut SoABufferSet,
    regs: &mut HashMap<String, RegVal>,
) -> Option<String> {
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
            MirInst::ConstQuat { dst, w, x, y, z } => {
                let vw = regs.get(w).map_or(1.0, RegVal::as_scalar);
                let vx = regs.get(x).map_or(0.0, RegVal::as_scalar);
                let vy = regs.get(y).map_or(0.0, RegVal::as_scalar);
                let vz = regs.get(z).map_or(0.0, RegVal::as_scalar);
                regs.insert(dst.clone(), RegVal::Quat([vw, vx, vy, vz]));
            }
            MirInst::ConstMat3 { dst, elems } => {
                let mut arr = [0.0; 9];
                for (i, r) in elems.iter().enumerate() {
                    arr[i] = regs.get(r).map_or(0.0, RegVal::as_scalar);
                }
                regs.insert(dst.clone(), RegVal::Mat3(arr));
            }
            MirInst::LoadSoA {
                dst,
                field,
                is_next,
                mir_type,
                ..
            } => {
                let (buf, idx) = if *is_next {
                    (&*next, write_idx)
                } else {
                    (prev, read_idx)
                };
                if let Some(arr) = buf.fields.get(field) {
                    match mir_type {
                        MirType::Vec3Simd4 => {
                            let base = idx * 4;
                            regs.insert(
                                dst.clone(),
                                RegVal::Vec3([arr[base], arr[base + 1], arr[base + 2]]),
                            );
                        }
                        MirType::QuatSimd4 => {
                            let base = idx * 4;
                            regs.insert(
                                dst.clone(),
                                RegVal::Quat([arr[base], arr[base + 1], arr[base + 2], arr[base + 3]]),
                            );
                        }
                        MirType::Mat3Simd12 => {
                            let base = idx * 12;
                            regs.insert(
                                dst.clone(),
                                RegVal::Mat3([
                                    arr[base],
                                    arr[base + 1],
                                    arr[base + 2],
                                    arr[base + 4],
                                    arr[base + 5],
                                    arr[base + 6],
                                    arr[base + 8],
                                    arr[base + 9],
                                    arr[base + 10],
                                ]),
                            );
                        }
                        _ => {
                            regs.insert(dst.clone(), RegVal::Scalar(arr[idx]));
                        }
                    }
                }
            }
            MirInst::StoreSoA {
                field,
                src,
                mir_type,
                ..
            } => {
                if let (Some(arr), Some(val)) = (next.fields.get_mut(field), regs.get(src)) {
                    match mir_type {
                        MirType::Vec3Simd4 => {
                            let v = val.as_vec3();
                            let base = write_idx * 4;
                            arr[base] = v[0];
                            arr[base + 1] = v[1];
                            arr[base + 2] = v[2];
                            arr[base + 3] = 0.0;
                        }
                        MirType::QuatSimd4 => {
                            let q = val.as_quat();
                            let base = write_idx * 4;
                            arr[base] = q[0];
                            arr[base + 1] = q[1];
                            arr[base + 2] = q[2];
                            arr[base + 3] = q[3];
                        }
                        MirType::Mat3Simd12 => {
                            let m = val.as_mat3();
                            let base = write_idx * 12;
                            arr[base] = m[0];
                            arr[base + 1] = m[1];
                            arr[base + 2] = m[2];
                            arr[base + 3] = 0.0;
                            arr[base + 4] = m[3];
                            arr[base + 5] = m[4];
                            arr[base + 6] = m[5];
                            arr[base + 7] = 0.0;
                            arr[base + 8] = m[6];
                            arr[base + 9] = m[7];
                            arr[base + 10] = m[8];
                            arr[base + 11] = 0.0;
                        }
                        _ => {
                            arr[write_idx] = val.as_scalar();
                        }
                    }
                }
            }
            MirInst::VecExtract { dst, vec, lane } => {
                if let Some(v) = regs.get(vec) {
                    let s = match v {
                        RegVal::Vec3(arr) => arr[(*lane).min(2)],
                        RegVal::Quat(q) => q[(*lane).min(3)],
                        _ => v.as_scalar(),
                    };
                    regs.insert(dst.clone(), RegVal::Scalar(s));
                }
            }
            MirInst::Select {
                dst,
                cond,
                then_val,
                else_val,
                ..
            } => {
                let c = regs.get(cond).is_some_and(|v| v.as_scalar() != 0.0);
                if let Some(chosen) = if c {
                    regs.get(then_val).cloned()
                } else {
                    regs.get(else_val).cloned()
                } {
                    regs.insert(dst.clone(), chosen);
                }
            }
            MirInst::UnaryNeg { dst, src, .. } => {
                if let Some(val) = regs.get(src) {
                    let negated = match val {
                        RegVal::Scalar(s) => RegVal::Scalar(-s),
                        RegVal::Vec3(v) => RegVal::Vec3([-v[0], -v[1], -v[2]]),
                        RegVal::Quat(q) => RegVal::Quat([-q[0], -q[1], -q[2], -q[3]]),
                        RegVal::Mat3(m) => RegVal::Mat3(m.map(|e| -e)),
                    };
                    regs.insert(dst.clone(), negated);
                }
            }
            MirInst::BinOp {
                dst, op, lhs, rhs, ..
            } => {
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
                        (RegVal::Quat(a), RegVal::Quat(b)) if *op == BinOp::Mul => {
                            RegVal::Quat(quat_mul(*a, *b))
                        }
                        (RegVal::Mat3(m), RegVal::Vec3(v)) if *op == BinOp::Mul => {
                            RegVal::Vec3(mat3_mul_vec3(*m, *v))
                        }
                        (RegVal::Mat3(a), RegVal::Mat3(b)) if *op == BinOp::Mul => {
                            RegVal::Mat3(mat3_mul_mat3(*a, *b))
                        }
                        (RegVal::Mat3(a), RegVal::Mat3(b)) => {
                            let mut out = [0.0; 9];
                            for i in 0..9 {
                                out[i] = scalar_binop(a[i], b[i], *op, mode);
                            }
                            RegVal::Mat3(out)
                        }
                        (RegVal::Mat3(a), RegVal::Scalar(b)) => {
                            RegVal::Mat3(a.map(|e| scalar_binop(e, *b, *op, mode)))
                        }
                        (RegVal::Scalar(a), RegVal::Mat3(b)) => {
                            RegVal::Mat3(b.map(|e| scalar_binop(*a, e, *op, mode)))
                        }
                        (RegVal::Scalar(a), RegVal::Scalar(b)) => {
                            RegVal::Scalar(scalar_binop(*a, *b, *op, mode))
                        }
                        _ => RegVal::Scalar(scalar_binop(l.as_scalar(), r.as_scalar(), *op, mode)),
                    };
                    regs.insert(dst.clone(), out);
                }
            }
            MirInst::Call {
                dst,
                intrinsic,
                args,
                ..
            } => {
                let vals: Vec<RegVal> = args.iter().filter_map(|a| regs.get(a).cloned()).collect();
                if intrinsic == "assert_approx" && vals.len() == 3 {
                    let ok = match (&vals[0], &vals[1], &vals[2]) {
                        (RegVal::Scalar(a), RegVal::Scalar(b), RegVal::Scalar(tol)) => {
                            (a - b).abs() <= *tol
                        }
                        (RegVal::Vec3(a), RegVal::Vec3(b), RegVal::Vec3(tol)) => {
                            (a[0] - b[0]).abs() <= tol[0]
                                && (a[1] - b[1]).abs() <= tol[1]
                                && (a[2] - b[2]).abs() <= tol[2]
                        }
                        (RegVal::Quat(a), RegVal::Quat(b), RegVal::Quat(tol)) => (0..4)
                            .all(|i| (a[i] - b[i]).abs() <= tol[i]),
                        _ => false,
                    };
                    if !ok {
                        return Some(format!(
                            "assertion failed: {:?} != {:?} (tol {:?})",
                            vals[0], vals[1], vals[2]
                        ));
                    }
                    regs.insert(dst.clone(), RegVal::Scalar(1.0));
                    continue;
                }

                let res = match (intrinsic.as_str(), vals.as_slice()) {
                    ("norm", [RegVal::Vec3(v)]) => {
                        RegVal::Scalar(libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                    }
                    ("norm", [RegVal::Quat(q)]) => RegVal::Scalar(libm::sqrt(
                        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3],
                    )),
                    ("normalize", [RegVal::Vec3(v)]) => {
                        let m = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                        if m == 0.0 {
                            RegVal::Vec3([0.0; 3])
                        } else {
                            RegVal::Vec3([v[0] / m, v[1] / m, v[2] / m])
                        }
                    }
                    ("normalize" | "quat_norm", [RegVal::Quat(q)]) => RegVal::Quat(quat_norm(*q)),
                    ("quat_conj", [RegVal::Quat(q)]) => {
                        RegVal::Quat([q[0], -q[1], -q[2], -q[3]])
                    }
                    ("quat_rotate", [RegVal::Quat(q), RegVal::Vec3(v)]) => {
                        RegVal::Vec3(quat_rotate(*q, *v))
                    }
                    ("quat_integrate", [RegVal::Quat(q), RegVal::Vec3(w), RegVal::Scalar(dt_v)]) => {
                        RegVal::Quat(quat_integrate(*q, *w, *dt_v))
                    }
                    ("mat3_diag", [RegVal::Scalar(x), RegVal::Scalar(y), RegVal::Scalar(z)]) => {
                        RegVal::Mat3([*x, 0.0, 0.0, 0.0, *y, 0.0, 0.0, 0.0, *z])
                    }
                    ("mat3_from_quat", [RegVal::Quat(q)]) => {
                        let [w, x, y, z] = quat_norm(*q);
                        RegVal::Mat3([
                            1.0 - 2.0 * (y * y + z * z),
                            2.0 * (x * y - z * w),
                            2.0 * (x * z + y * w),
                            2.0 * (x * y + z * w),
                            1.0 - 2.0 * (x * x + z * z),
                            2.0 * (y * z - x * w),
                            2.0 * (x * z - y * w),
                            2.0 * (y * z + x * w),
                            1.0 - 2.0 * (x * x + y * y),
                        ])
                    }
                    ("mat3_transpose", [RegVal::Mat3(m)]) => RegVal::Mat3([
                        m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8],
                    ]),
                    ("mat3_inv", [RegVal::Mat3(m)]) => RegVal::Mat3(mat3_inv(*m)),
                    ("dot", [RegVal::Vec3(a), RegVal::Vec3(b)]) => {
                        RegVal::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
                    }
                    ("cross", [RegVal::Vec3(a), RegVal::Vec3(b)]) => RegVal::Vec3([
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]),
                    ("sqrt", [RegVal::Scalar(x)]) => RegVal::Scalar(libm::sqrt(x.max(0.0))),
                    ("cbrt", [RegVal::Scalar(x)]) => RegVal::Scalar(libm::cbrt(*x)),
                    ("sin", [RegVal::Scalar(x)]) => RegVal::Scalar(libm::sin(*x)),
                    ("cos", [RegVal::Scalar(x)]) => RegVal::Scalar(libm::cos(*x)),
                    ("tan", [RegVal::Scalar(x)]) => RegVal::Scalar(libm::tan(*x)),
                    ("atan2", [RegVal::Scalar(y), RegVal::Scalar(x)]) => {
                        RegVal::Scalar(libm::atan2(*y, *x))
                    }
                    ("abs", [RegVal::Scalar(x)]) => RegVal::Scalar(x.abs()),
                    ("abs", [RegVal::Vec3(v)]) => {
                        RegVal::Vec3([v[0].abs(), v[1].abs(), v[2].abs()])
                    }
                    ("min", [RegVal::Scalar(a), RegVal::Scalar(b)]) => RegVal::Scalar(a.min(*b)),
                    ("max", [RegVal::Scalar(a), RegVal::Scalar(b)]) => RegVal::Scalar(a.max(*b)),
                    ("clamp", [RegVal::Scalar(x), RegVal::Scalar(lo), RegVal::Scalar(hi)]) => {
                        RegVal::Scalar(x.max(*lo).min(*hi))
                    }
                    ("select", [RegVal::Scalar(c), t, f]) => {
                        if *c != 0.0 {
                            t.clone()
                        } else {
                            f.clone()
                        }
                    }
                    _ => RegVal::Scalar(0.0),
                };
                regs.insert(dst.clone(), res);
            }
            MirInst::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                for i in *start..*end {
                    regs.insert(format!("%loop_{var}"), RegVal::Scalar(i as f64));
                    if let Some(err) = exec_insts_with_assertions(
                        body, read_idx, write_idx, dt, mode, prev, next, regs,
                    ) {
                        return Some(err);
                    }
                }
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
                let c = regs.get(cond).is_some_and(|v| v.as_scalar() != 0.0);
                if c {
                    if let Some(err) = exec_insts_with_assertions(
                        then_insts, read_idx, write_idx, dt, mode, prev, next, regs,
                    ) {
                        return Some(err);
                    }
                } else if let Some(el) = else_insts
                    && let Some(err) = exec_insts_with_assertions(
                        el, read_idx, write_idx, dt, mode, prev, next, regs,
                    )
                {
                    return Some(err);
                }
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct TestCaseReport {
    pub name: String,
    pub passed: bool,
    pub message: Option<String>,
}

/// Executes all compiled `.aero` `test "name" { ... }` blocks on the deterministic VM.
pub fn run_mir_tests(mir: &MirProgram) -> Vec<TestCaseReport> {
    let empty = SoABufferSet {
        fields: HashMap::new(),
    };
    let mut empty_next = SoABufferSet {
        fields: HashMap::new(),
    };
    mir.tests
        .iter()
        .map(|t| {
            let mut regs = HashMap::new();
            let err = exec_insts_with_assertions(
                &t.insts,
                0,
                0,
                0.0,
                NumericMode::Float64,
                &empty,
                &mut empty_next,
                &mut regs,
            );
            TestCaseReport {
                name: t.name.clone(),
                passed: err.is_none(),
                message: err,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_mir::lower_to_mir;
    use aerotick_parser::parse;
    use aerotick_sema::analyze;

    fn compile_mir(src: &str) -> MirProgram {
        let (tokens, l_errs) = lex(src);
        assert!(l_errs.is_empty(), "{l_errs:?}");
        let (ast_opt, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let Some(ast) = ast_opt else {
            panic!("expected AST");
        };
        let (hir, diags) = analyze(&ast);
        assert!(diags.is_empty(), "{diags:?}");
        lower_to_mir(&hir)
    }

    #[test]
    fn executes_1000_ticks_deterministically_with_identical_sha256() {
        let mir = compile_mir(include_str!("../../../examples/reference_body.aero"));
        let Ok(mut vm1) = DeterministicVm::new(&mir, 1, 0.01, NumericMode::Float64) else {
            panic!("vm1 init failed");
        };
        let Ok(mut vm2) = DeterministicVm::new(&mir, 1, 0.01, NumericMode::Float64) else {
            panic!("vm2 init failed");
        };

        for _ in 0..1000 {
            vm1.step_tick();
            vm2.step_tick();
        }
        assert_eq!(vm1.state_sha256_hex(), vm2.state_sha256_hex());
        let Some(vel) = vm1.read_entity0_vec3("vel") else {
            panic!("expected vel field");
        };
        assert!((vel[1] - 61.9335).abs() < 1e-9);
    }

    #[test]
    fn parallel_swarm_and_checkpoint_rollback_are_bit_exact() {
        let mir = compile_mir(include_str!("../../../examples/reference_body.aero"));
        let Ok(mut seq) = DeterministicVm::new(&mir, 256, 0.01, NumericMode::Float64) else {
            panic!("seq init failed");
        };
        let Ok(mut par) = DeterministicVm::new(&mir, 256, 0.01, NumericMode::Float64) else {
            panic!("par init failed");
        };

        let mut checkpoint_bytes = None;
        for t in 0..100 {
            if t == 50 {
                checkpoint_bytes = Some(seq.save_checkpoint().to_bytes());
            }
            seq.step_tick();
            assert!(par.step_tick_parallel(4).is_ok());
        }
        assert_eq!(seq.state_sha256_hex(), par.state_sha256_hex());

        let Some(raw) = checkpoint_bytes else {
            panic!("expected checkpoint bytes");
        };
        let Ok(cp) = VmCheckpoint::from_bytes(&raw) else {
            panic!("failed to decode .aeroreplay checkpoint");
        };
        assert!(par.restore_checkpoint(&cp).is_ok());
        for _ in 50..100 {
            par.step_tick();
        }
        assert_eq!(seq.state_sha256_hex(), par.state_sha256_hex());
    }

    #[test]
    fn executes_inline_aero_test_blocks() {
        let src = r#"
            test "aerospace_conversions" {
                let speed: m/s = sqrt(100.0 m^2/s^2);
                assert_approx(speed, 10.0 m/s, 1e-9 m/s);
                assert_approx(180.0 deg, 3.141592653589793 rad, 1e-12 rad);
            }
        "#;
        let mir = compile_mir(src);
        let reports = run_mir_tests(&mir);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].passed, "{:?}", reports[0].message);
    }
}
