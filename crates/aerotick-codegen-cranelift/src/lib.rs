#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 5B: Pure-Rust Native JIT Backend powered by Cranelift (`cranelift-jit`).
//! Lowers unit-erased SoA MIR directly into native x86_64/ARM64 machine code
//! with strict IEEE-754 non-contraction semantics and zero C++ dependencies.

use aerotick_ast::BinOp;
use aerotick_mir::{MirInst, MirProgram, MirSystem, MirType, SoAStateLayout};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JitError {
    #[error("MIR program has no runnable system")]
    NoSystem,
    #[error("Missing SoA layout for state '{0}'")]
    MissingLayout(String),
    #[error("Cranelift target error: {0}")]
    Target(String),
    #[error("Cranelift module error: {0}")]
    Module(String),
    #[error("Cranelift verification error: {0}")]
    Codegen(String),
}

type StepKernelFn = unsafe extern "C" fn(*const *const f64, *const *mut f64, usize, f64);

extern "C" fn det_sin(x: f64) -> f64 {
    libm::sin(x)
}
extern "C" fn det_cos(x: f64) -> f64 {
    libm::cos(x)
}
extern "C" fn det_tan(x: f64) -> f64 {
    libm::tan(x)
}
extern "C" fn det_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
extern "C" fn det_sqrt(x: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { libm::sqrt(x) }
}
extern "C" fn det_cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}
extern "C" fn det_pow(a: f64, b: f64) -> f64 {
    libm::pow(a, b)
}

#[derive(Clone)]
enum JitVal {
    Scalar(Value),
    Vec3([Value; 3]),
    Quat([Value; 4]),
    Mat3([Value; 9]),
}

impl JitVal {
    fn as_scalar(&self, _zero: Value) -> Value {
        match self {
            Self::Scalar(v) => *v,
            Self::Vec3(v) => v[0],
            Self::Quat(q) => q[0],
            Self::Mat3(m) => m[0],
        }
    }

    fn as_vec3(&self, _zero: Value) -> [Value; 3] {
        match self {
            Self::Vec3(v) => *v,
            Self::Scalar(s) => [*s, *s, *s],
            Self::Quat(q) => [q[1], q[2], q[3]],
            Self::Mat3(m) => [m[0], m[1], m[2]],
        }
    }

    fn as_quat(&self, zero: Value) -> [Value; 4] {
        match self {
            Self::Quat(q) => *q,
            Self::Vec3(v) => [zero, v[0], v[1], v[2]],
            Self::Scalar(s) => [*s, zero, zero, zero],
            Self::Mat3(_) => [zero, zero, zero, zero],
        }
    }

    fn as_mat3(&self, zero: Value) -> [Value; 9] {
        match self {
            Self::Mat3(m) => *m,
            Self::Scalar(s) => [*s, zero, zero, zero, *s, zero, zero, zero, *s],
            _ => [zero; 9],
        }
    }
}

pub struct CraneliftJitKernel {
    _module: JITModule,
    fn_ptr: *const u8,
    pub system_name: String,
    pub target_isa: String,
    pub ir_text: String,
}

impl CraneliftJitKernel {
    pub fn compile(mir: &MirProgram) -> Result<Self, JitError> {
        let system = mir.systems.first().ok_or(JitError::NoSystem)?;
        let layout = mir
            .layouts
            .iter()
            .find(|l| l.state_name == system.state_name)
            .ok_or_else(|| JitError::MissingLayout(system.state_name.clone()))?;

        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| JitError::Target(e.to_string()))?;
        flag_builder
            .set("is_pic", "true")
            .map_err(|e| JitError::Target(e.to_string()))?;
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JitError::Target(e.to_string()))?;

        let isa_builder =
            cranelift_native::builder().map_err(|e| JitError::Target(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Target(e.to_string()))?;
        let target_isa = isa.triple().to_string();
        let ptr_ty = isa.pointer_type();

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("aerotick_det_sin", det_sin as *const u8);
        jit_builder.symbol("aerotick_det_cos", det_cos as *const u8);
        jit_builder.symbol("aerotick_det_tan", det_tan as *const u8);
        jit_builder.symbol("aerotick_det_atan2", det_atan2 as *const u8);
        jit_builder.symbol("aerotick_det_sqrt", det_sqrt as *const u8);
        jit_builder.symbol("aerotick_det_cbrt", det_cbrt as *const u8);
        jit_builder.symbol("aerotick_det_pow", det_pow as *const u8);

        let mut module = JITModule::new(jit_builder);
        let call_conv = module.isa().default_call_conv();

        let mut sig_unary = Signature::new(call_conv);
        sig_unary.params.push(AbiParam::new(types::F64));
        sig_unary.returns.push(AbiParam::new(types::F64));

        let mut sig_binary = Signature::new(call_conv);
        sig_binary.params.push(AbiParam::new(types::F64));
        sig_binary.params.push(AbiParam::new(types::F64));
        sig_binary.returns.push(AbiParam::new(types::F64));

        let mut helpers = HashMap::new();
        for name in [
            "aerotick_det_sin",
            "aerotick_det_cos",
            "aerotick_det_tan",
            "aerotick_det_sqrt",
            "aerotick_det_cbrt",
        ] {
            let id = module
                .declare_function(name, Linkage::Import, &sig_unary)
                .map_err(|e| JitError::Module(e.to_string()))?;
            helpers.insert(name.to_string(), id);
        }
        for name in ["aerotick_det_atan2", "aerotick_det_pow"] {
            let id = module
                .declare_function(name, Linkage::Import, &sig_binary)
                .map_err(|e| JitError::Module(e.to_string()))?;
            helpers.insert(name.to_string(), id);
        }

        let mut ctx = module.make_context();
        ctx.func.signature = Signature::new(call_conv);
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // prev_ptrs
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // next_ptrs
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // count
        ctx.func.signature.params.push(AbiParam::new(types::F64)); // dt

        let sym_name = format!("aerotick_jit_step_{}", system.name.to_lowercase());
        let fn_id = module
            .declare_function(&sym_name, Linkage::Export, &ctx.func.signature)
            .map_err(|e| JitError::Module(e.to_string()))?;

        let mut fb_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let entry_bb = builder.create_block();
            builder.append_block_params_for_function_params(entry_bb);
            builder.switch_to_block(entry_bb);
            builder.seal_block(entry_bb);

            let prev_ptrs = builder.block_params(entry_bb)[0];
            let next_ptrs = builder.block_params(entry_bb)[1];
            let count = builder.block_params(entry_bb)[2];
            let dt = builder.block_params(entry_bb)[3];

            let mut helper_refs = HashMap::new();
            for (name, id) in &helpers {
                let fref = module.declare_func_in_func(*id, builder.func);
                helper_refs.insert(name.clone(), fref);
            }

            let zero_i = builder.ins().iconst(ptr_ty, 0);
            let loop_bb = builder.create_block();
            builder.append_block_param(loop_bb, ptr_ty);
            let body_bb = builder.create_block();
            let exit_bb = builder.create_block();

            builder.ins().jump(loop_bb, &[zero_i.into()]);
            builder.switch_to_block(loop_bb);
            let idx = builder.block_params(loop_bb)[0];
            let cond = builder.ins().icmp(IntCC::UnsignedLessThan, idx, count);
            builder.ins().brif(cond, body_bb, &[], exit_bb, &[]);

            builder.switch_to_block(body_bb);
            builder.seal_block(body_bb);

            let mut state = LowerState {
                builder,
                ptr_ty,
                prev_ptrs,
                next_ptrs,
                idx,
                dt,
                layout,
                system,
                helpers: helper_refs,
                regs: HashMap::new(),
            };
            state.emit_insts(&system.insts);
            let one_i = state.builder.ins().iconst(ptr_ty, 1);
            let next_idx = state.builder.ins().iadd(idx, one_i);
            state.builder.ins().jump(loop_bb, &[next_idx.into()]);

            state.builder.seal_block(loop_bb);
            state.builder.switch_to_block(exit_bb);
            state.builder.seal_block(exit_bb);
            state.builder.ins().return_(&[]);
            state.builder.finalize();
        }

        let ir_text = ctx.func.display().to_string();
        module
            .define_function(fn_id, &mut ctx)
            .map_err(|e| JitError::Codegen(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| JitError::Module(e.to_string()))?;

        let fn_ptr = module.get_finalized_function(fn_id);
        Ok(Self {
            _module: module,
            fn_ptr,
            system_name: system.name.clone(),
            target_isa,
            ir_text,
        })
    }

    /// Executes one double-buffered tick over `count` SoA entities using native JIT machine code.
    ///
    /// # Safety
    /// Caller must ensure `prev_ptrs` and `next_ptrs` contain valid pointers sized for `count` entities
    /// matching the compiled `MirProgram` `SoAStateLayout` field order.
    pub unsafe fn step(
        &self,
        prev_ptrs: &[*const f64],
        next_ptrs: &[*mut f64],
        count: usize,
        dt: f64,
    ) {
        let func: StepKernelFn = unsafe { std::mem::transmute(self.fn_ptr) };
        unsafe {
            func(prev_ptrs.as_ptr(), next_ptrs.as_ptr(), count, dt);
        }
    }
}

struct LowerState<'a> {
    builder: FunctionBuilder<'a>,
    ptr_ty: types::Type,
    prev_ptrs: Value,
    next_ptrs: Value,
    idx: Value,
    dt: Value,
    layout: &'a SoAStateLayout,
    system: &'a MirSystem,
    helpers: HashMap<String, cranelift_codegen::ir::FuncRef>,
    regs: HashMap<String, JitVal>,
}

impl<'a> LowerState<'a> {
    fn f64_const(&mut self, v: f64) -> Value {
        self.builder.ins().f64const(v)
    }

    fn ptr_bytes(&self) -> i64 {
        i64::from(self.ptr_ty.bytes())
    }

    fn load_field_base(&mut self, table_ptr: Value, field_name: &str) -> Option<(Value, usize)> {
        let pos = self.layout.fields.iter().position(|f| f.name == field_name)?;
        let lanes = self.layout.fields[pos].lanes;
        let offset = (pos as i32) * (self.ptr_bytes() as i32);
        let base = self
            .builder
            .ins()
            .load(self.ptr_ty, MemFlags::trusted(), table_ptr, offset);
        Some((base, lanes))
    }

    fn elem_ptr(&mut self, base: Value, lanes: usize) -> Value {
        let stride_elems = self.builder.ins().imul_imm(self.idx, lanes as i64);
        let byte_off = self.builder.ins().imul_imm(stride_elems, 8);
        self.builder.ins().iadd(base, byte_off)
    }

    fn call_helper1(&mut self, sym: &str, a: Value) -> Value {
        if let Some(&fref) = self.helpers.get(sym) {
            let call = self.builder.ins().call(fref, &[a]);
            self.builder.inst_results(call)[0]
        } else {
            self.f64_const(0.0)
        }
    }

    fn call_helper2(&mut self, sym: &str, a: Value, b: Value) -> Value {
        if let Some(&fref) = self.helpers.get(sym) {
            let call = self.builder.ins().call(fref, &[a, b]);
            self.builder.inst_results(call)[0]
        } else {
            self.f64_const(0.0)
        }
    }

    fn scalar_binop(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let zero = self.f64_const(0.0);
        let one = self.f64_const(1.0);
        match op {
            BinOp::Add => self.builder.ins().fadd(a, b),
            BinOp::Sub => self.builder.ins().fsub(a, b),
            BinOp::Mul => self.builder.ins().fmul(a, b),
            BinOp::Div => {
                let is_z = self.builder.ins().fcmp(FloatCC::Equal, b, zero);
                let q = self.builder.ins().fdiv(a, b);
                self.builder.ins().select(is_z, zero, q)
            }
            BinOp::Pow => self.call_helper2("aerotick_det_pow", a, b),
            BinOp::Eq => {
                let c = self.builder.ins().fcmp(FloatCC::Equal, a, b);
                self.builder.ins().select(c, one, zero)
            }
            BinOp::Ne => {
                let c = self.builder.ins().fcmp(FloatCC::NotEqual, a, b);
                self.builder.ins().select(c, one, zero)
            }
            BinOp::Lt => {
                let c = self.builder.ins().fcmp(FloatCC::LessThan, a, b);
                self.builder.ins().select(c, one, zero)
            }
            BinOp::Le => {
                let c = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
                self.builder.ins().select(c, one, zero)
            }
            BinOp::Gt => {
                let c = self.builder.ins().fcmp(FloatCC::GreaterThan, a, b);
                self.builder.ins().select(c, one, zero)
            }
            BinOp::Ge => {
                let c = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
                self.builder.ins().select(c, one, zero)
            }
        }
    }

    fn quat_mul(&mut self, a: [Value; 4], b: [Value; 4]) -> [Value; 4] {
        let w = {
            let p0 = self.builder.ins().fmul(a[0], b[0]);
            let p1 = self.builder.ins().fmul(a[1], b[1]);
            let p2 = self.builder.ins().fmul(a[2], b[2]);
            let p3 = self.builder.ins().fmul(a[3], b[3]);
            let t0 = self.builder.ins().fsub(p0, p1);
            let t1 = self.builder.ins().fsub(t0, p2);
            self.builder.ins().fsub(t1, p3)
        };
        let x = {
            let p0 = self.builder.ins().fmul(a[0], b[1]);
            let p1 = self.builder.ins().fmul(a[1], b[0]);
            let p2 = self.builder.ins().fmul(a[2], b[3]);
            let p3 = self.builder.ins().fmul(a[3], b[2]);
            let t0 = self.builder.ins().fadd(p0, p1);
            let t1 = self.builder.ins().fadd(t0, p2);
            self.builder.ins().fsub(t1, p3)
        };
        let y = {
            let p0 = self.builder.ins().fmul(a[0], b[2]);
            let p1 = self.builder.ins().fmul(a[1], b[3]);
            let p2 = self.builder.ins().fmul(a[2], b[0]);
            let p3 = self.builder.ins().fmul(a[3], b[1]);
            let t0 = self.builder.ins().fsub(p0, p1);
            let t1 = self.builder.ins().fadd(t0, p2);
            self.builder.ins().fadd(t1, p3)
        };
        let z = {
            let p0 = self.builder.ins().fmul(a[0], b[3]);
            let p1 = self.builder.ins().fmul(a[1], b[2]);
            let p2 = self.builder.ins().fmul(a[2], b[1]);
            let p3 = self.builder.ins().fmul(a[3], b[0]);
            let t0 = self.builder.ins().fadd(p0, p1);
            let t1 = self.builder.ins().fsub(t0, p2);
            self.builder.ins().fadd(t1, p3)
        };
        [w, x, y, z]
    }

    fn quat_norm(&mut self, q: [Value; 4]) -> [Value; 4] {
        let zero = self.f64_const(0.0);
        let one = self.f64_const(1.0);
        let s0 = self.builder.ins().fmul(q[0], q[0]);
        let s1 = self.builder.ins().fmul(q[1], q[1]);
        let s2 = self.builder.ins().fmul(q[2], q[2]);
        let s3 = self.builder.ins().fmul(q[3], q[3]);
        let a01 = self.builder.ins().fadd(s0, s1);
        let a23 = self.builder.ins().fadd(s2, s3);
        let sq = self.builder.ins().fadd(a01, a23);
        let n = self.call_helper1("aerotick_det_sqrt", sq);
        let is_z = self.builder.ins().fcmp(FloatCC::Equal, n, zero);
        let qw = self.builder.ins().fdiv(q[0], n);
        let qx = self.builder.ins().fdiv(q[1], n);
        let qy = self.builder.ins().fdiv(q[2], n);
        let qz = self.builder.ins().fdiv(q[3], n);
        [
            self.builder.ins().select(is_z, one, qw),
            self.builder.ins().select(is_z, zero, qx),
            self.builder.ins().select(is_z, zero, qy),
            self.builder.ins().select(is_z, zero, qz),
        ]
    }

    fn mat3_inv(&mut self, m: [Value; 9]) -> [Value; 9] {
        let zero = self.f64_const(0.0);
        let one = self.f64_const(1.0);
        let eps = self.f64_const(1e-18);

        let sub_prod = |b: &mut FunctionBuilder<'a>, a1: Value, a2: Value, b1: Value, b2: Value| {
            let p1 = b.ins().fmul(a1, a2);
            let p2 = b.ins().fmul(b1, b2);
            b.ins().fsub(p1, p2)
        };

        let c00 = sub_prod(&mut self.builder, m[4], m[8], m[5], m[7]);
        let c01 = sub_prod(&mut self.builder, m[5], m[6], m[3], m[8]);
        let c02 = sub_prod(&mut self.builder, m[3], m[7], m[4], m[6]);

        let t0 = self.builder.ins().fmul(m[0], c00);
        let t1 = self.builder.ins().fmul(m[1], c01);
        let t2 = self.builder.ins().fmul(m[2], c02);
        let s01 = self.builder.ins().fadd(t0, t1);
        let det = self.builder.ins().fadd(s01, t2);
        let abs_det = self.builder.ins().fabs(det);
        let singular = self.builder.ins().fcmp(FloatCC::LessThan, abs_det, eps);
        let inv_det = self.builder.ins().fdiv(one, det);

        let raw = [
            c00,
            sub_prod(&mut self.builder, m[2], m[7], m[1], m[8]),
            sub_prod(&mut self.builder, m[1], m[5], m[2], m[4]),
            c01,
            sub_prod(&mut self.builder, m[0], m[8], m[2], m[6]),
            sub_prod(&mut self.builder, m[2], m[3], m[0], m[5]),
            c02,
            sub_prod(&mut self.builder, m[1], m[6], m[0], m[7]),
            sub_prod(&mut self.builder, m[0], m[4], m[1], m[3]),
        ];

        let mut out = [zero; 9];
        for i in 0..9 {
            let scaled = self.builder.ins().fmul(raw[i], inv_det);
            out[i] = self.builder.ins().select(singular, zero, scaled);
        }
        out
    }

    fn emit_insts(&mut self, insts: &[MirInst]) {
        let zero = self.f64_const(0.0);
        let _sys_name = &self.system.name;
        for inst in insts {
            match inst {
                MirInst::ConstF64 { dst, value } => {
                    let v = self.f64_const(*value);
                    self.regs.insert(dst.clone(), JitVal::Scalar(v));
                }
                MirInst::LoadParam { dst, .. } => {
                    self.regs.insert(dst.clone(), JitVal::Scalar(self.dt));
                }
                MirInst::ConstVec3 { dst, x, y, z } => {
                    let vx = self.regs.get(x).map_or(zero, |v| v.as_scalar(zero));
                    let vy = self.regs.get(y).map_or(zero, |v| v.as_scalar(zero));
                    let vz = self.regs.get(z).map_or(zero, |v| v.as_scalar(zero));
                    self.regs.insert(dst.clone(), JitVal::Vec3([vx, vy, vz]));
                }
                MirInst::ConstQuat { dst, w, x, y, z } => {
                    let vw = self.regs.get(w).map_or(zero, |v| v.as_scalar(zero));
                    let vx = self.regs.get(x).map_or(zero, |v| v.as_scalar(zero));
                    let vy = self.regs.get(y).map_or(zero, |v| v.as_scalar(zero));
                    let vz = self.regs.get(z).map_or(zero, |v| v.as_scalar(zero));
                    self.regs.insert(dst.clone(), JitVal::Quat([vw, vx, vy, vz]));
                }
                MirInst::ConstMat3 { dst, elems } => {
                    let mut arr = [zero; 9];
                    for (i, r) in elems.iter().enumerate() {
                        arr[i] = self.regs.get(r).map_or(zero, |v| v.as_scalar(zero));
                    }
                    self.regs.insert(dst.clone(), JitVal::Mat3(arr));
                }
                MirInst::LoadSoA {
                    dst,
                    field,
                    is_next,
                    mir_type,
                    ..
                } => {
                    let tbl = if *is_next {
                        self.next_ptrs
                    } else {
                        self.prev_ptrs
                    };
                    if let Some((base, lanes)) = self.load_field_base(tbl, field) {
                        let ptr = self.elem_ptr(base, lanes);
                        match mir_type {
                            MirType::Vec3Simd4 => {
                                let x = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 0);
                                let y = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 8);
                                let z = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 16);
                                self.regs.insert(dst.clone(), JitVal::Vec3([x, y, z]));
                            }
                            MirType::QuatSimd4 => {
                                let w = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 0);
                                let x = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 8);
                                let y = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 16);
                                let z = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 24);
                                self.regs.insert(dst.clone(), JitVal::Quat([w, x, y, z]));
                            }
                            MirType::Mat3Simd12 => {
                                let mut m = [zero; 9];
                                let offs = [0, 8, 16, 32, 40, 48, 64, 72, 80];
                                for (i, off) in offs.iter().enumerate() {
                                    m[i] = self.builder.ins().load(
                                        types::F64,
                                        MemFlags::trusted(),
                                        ptr,
                                        *off,
                                    );
                                }
                                self.regs.insert(dst.clone(), JitVal::Mat3(m));
                            }
                            _ => {
                                let v = self.builder.ins().load(types::F64, MemFlags::trusted(), ptr, 0);
                                self.regs.insert(dst.clone(), JitVal::Scalar(v));
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
                    let Some(val) = self.regs.get(src).cloned() else {
                        continue;
                    };
                    if let Some((base, lanes)) = self.load_field_base(self.next_ptrs, field) {
                        let ptr = self.elem_ptr(base, lanes);
                        match mir_type {
                            MirType::Vec3Simd4 => {
                                let v = val.as_vec3(zero);
                                self.builder.ins().store(MemFlags::trusted(), v[0], ptr, 0);
                                self.builder.ins().store(MemFlags::trusted(), v[1], ptr, 8);
                                self.builder.ins().store(MemFlags::trusted(), v[2], ptr, 16);
                                self.builder.ins().store(MemFlags::trusted(), zero, ptr, 24);
                            }
                            MirType::QuatSimd4 => {
                                let q = val.as_quat(zero);
                                self.builder.ins().store(MemFlags::trusted(), q[0], ptr, 0);
                                self.builder.ins().store(MemFlags::trusted(), q[1], ptr, 8);
                                self.builder.ins().store(MemFlags::trusted(), q[2], ptr, 16);
                                self.builder.ins().store(MemFlags::trusted(), q[3], ptr, 24);
                            }
                            MirType::Mat3Simd12 => {
                                let m = val.as_mat3(zero);
                                let offs = [0, 8, 16, 32, 40, 48, 64, 72, 80];
                                for (i, off) in offs.iter().enumerate() {
                                    self.builder.ins().store(MemFlags::trusted(), m[i], ptr, *off);
                                }
                                for pad_off in [24, 56, 88] {
                                    self.builder.ins().store(MemFlags::trusted(), zero, ptr, pad_off);
                                }
                            }
                            _ => {
                                let s = val.as_scalar(zero);
                                self.builder.ins().store(MemFlags::trusted(), s, ptr, 0);
                            }
                        }
                    }
                }
                MirInst::VecExtract { dst, vec, lane } => {
                    if let Some(v) = self.regs.get(vec).cloned() {
                        let out = match v {
                            JitVal::Vec3(arr) => arr[(*lane).min(2)],
                            JitVal::Quat(q) => q[(*lane).min(3)],
                            _ => v.as_scalar(zero),
                        };
                        self.regs.insert(dst.clone(), JitVal::Scalar(out));
                    }
                }
                MirInst::Select {
                    dst,
                    cond,
                    then_val,
                    else_val,
                    ..
                } => {
                    let c_val = self.regs.get(cond).map_or(zero, |v| v.as_scalar(zero));
                    let c_bool = self.builder.ins().fcmp(FloatCC::NotEqual, c_val, zero);
                    if let (Some(t), Some(e)) = (
                        self.regs.get(then_val).cloned(),
                        self.regs.get(else_val).cloned(),
                    ) {
                        let out = match (t, e) {
                            (JitVal::Vec3(tv), JitVal::Vec3(ev)) => JitVal::Vec3([
                                self.builder.ins().select(c_bool, tv[0], ev[0]),
                                self.builder.ins().select(c_bool, tv[1], ev[1]),
                                self.builder.ins().select(c_bool, tv[2], ev[2]),
                            ]),
                            (JitVal::Quat(tq), JitVal::Quat(eq)) => JitVal::Quat([
                                self.builder.ins().select(c_bool, tq[0], eq[0]),
                                self.builder.ins().select(c_bool, tq[1], eq[1]),
                                self.builder.ins().select(c_bool, tq[2], eq[2]),
                                self.builder.ins().select(c_bool, tq[3], eq[3]),
                            ]),
                            (tv, ev) => {
                                let ts = tv.as_scalar(zero);
                                let es = ev.as_scalar(zero);
                                JitVal::Scalar(self.builder.ins().select(c_bool, ts, es))
                            }
                        };
                        self.regs.insert(dst.clone(), out);
                    }
                }
                MirInst::UnaryNeg { dst, src, .. } => {
                    if let Some(v) = self.regs.get(src).cloned() {
                        let out = match v {
                            JitVal::Scalar(s) => JitVal::Scalar(self.builder.ins().fneg(s)),
                            JitVal::Vec3(a) => JitVal::Vec3([
                                self.builder.ins().fneg(a[0]),
                                self.builder.ins().fneg(a[1]),
                                self.builder.ins().fneg(a[2]),
                            ]),
                            JitVal::Quat(q) => JitVal::Quat([
                                self.builder.ins().fneg(q[0]),
                                self.builder.ins().fneg(q[1]),
                                self.builder.ins().fneg(q[2]),
                                self.builder.ins().fneg(q[3]),
                            ]),
                            JitVal::Mat3(m) => JitVal::Mat3(m.map(|e| self.builder.ins().fneg(e))),
                        };
                        self.regs.insert(dst.clone(), out);
                    }
                }
                MirInst::BinOp {
                    dst, op, lhs, rhs, ..
                } => {
                    if let (Some(l), Some(r)) =
                        (self.regs.get(lhs).cloned(), self.regs.get(rhs).cloned())
                    {
                        let out = match (l, r) {
                            (JitVal::Vec3(a), JitVal::Vec3(b)) => JitVal::Vec3([
                                self.scalar_binop(*op, a[0], b[0]),
                                self.scalar_binop(*op, a[1], b[1]),
                                self.scalar_binop(*op, a[2], b[2]),
                            ]),
                            (JitVal::Vec3(a), JitVal::Scalar(b)) => JitVal::Vec3([
                                self.scalar_binop(*op, a[0], b),
                                self.scalar_binop(*op, a[1], b),
                                self.scalar_binop(*op, a[2], b),
                            ]),
                            (JitVal::Scalar(a), JitVal::Vec3(b)) => JitVal::Vec3([
                                self.scalar_binop(*op, a, b[0]),
                                self.scalar_binop(*op, a, b[1]),
                                self.scalar_binop(*op, a, b[2]),
                            ]),
                            (JitVal::Quat(a), JitVal::Quat(b)) if *op == BinOp::Mul => {
                                JitVal::Quat(self.quat_mul(a, b))
                            }
                            (JitVal::Mat3(m), JitVal::Vec3(v)) if *op == BinOp::Mul => {
                                let r0 = self.dot3([m[0], m[1], m[2]], v);
                                let r1 = self.dot3([m[3], m[4], m[5]], v);
                                let r2 = self.dot3([m[6], m[7], m[8]], v);
                                JitVal::Vec3([r0, r1, r2])
                            }
                            (JitVal::Mat3(a), JitVal::Mat3(b)) if *op == BinOp::Mul => {
                                let mut out = [zero; 9];
                                for r in 0..3 {
                                    for c in 0..3 {
                                        out[r * 3 + c] = self.dot3(
                                            [a[r * 3], a[r * 3 + 1], a[r * 3 + 2]],
                                            [b[c], b[3 + c], b[6 + c]],
                                        );
                                    }
                                }
                                JitVal::Mat3(out)
                            }
                            (JitVal::Mat3(a), JitVal::Mat3(b)) => {
                                let mut out = [zero; 9];
                                for i in 0..9 {
                                    out[i] = self.scalar_binop(*op, a[i], b[i]);
                                }
                                JitVal::Mat3(out)
                            }
                            (JitVal::Mat3(a), JitVal::Scalar(b)) => {
                                let mut out = [zero; 9];
                                for i in 0..9 {
                                    out[i] = self.scalar_binop(*op, a[i], b);
                                }
                                JitVal::Mat3(out)
                            }
                            (JitVal::Scalar(a), JitVal::Mat3(b)) => {
                                let mut out = [zero; 9];
                                for i in 0..9 {
                                    out[i] = self.scalar_binop(*op, a, b[i]);
                                }
                                JitVal::Mat3(out)
                            }
                            (lv, rv) => {
                                let ls = lv.as_scalar(zero);
                                let rs = rv.as_scalar(zero);
                                JitVal::Scalar(self.scalar_binop(*op, ls, rs))
                            }
                        };
                        self.regs.insert(dst.clone(), out);
                    }
                }
                MirInst::Call {
                    dst,
                    intrinsic,
                    args,
                    ..
                } => {
                    let vals: Vec<JitVal> = args
                        .iter()
                        .filter_map(|a| self.regs.get(a).cloned())
                        .collect();
                    let res = match (intrinsic.as_str(), vals.as_slice()) {
                        ("sqrt", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper1("aerotick_det_sqrt", *x))
                        }
                        ("cbrt", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper1("aerotick_det_cbrt", *x))
                        }
                        ("sin", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper1("aerotick_det_sin", *x))
                        }
                        ("cos", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper1("aerotick_det_cos", *x))
                        }
                        ("tan", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper1("aerotick_det_tan", *x))
                        }
                        ("atan2", [JitVal::Scalar(y), JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.call_helper2("aerotick_det_atan2", *y, *x))
                        }
                        ("abs", [JitVal::Scalar(x)]) => {
                            JitVal::Scalar(self.builder.ins().fabs(*x))
                        }
                        ("abs", [JitVal::Vec3(v)]) => JitVal::Vec3([
                            self.builder.ins().fabs(v[0]),
                            self.builder.ins().fabs(v[1]),
                            self.builder.ins().fabs(v[2]),
                        ]),
                        ("min", [JitVal::Scalar(a), JitVal::Scalar(b)]) => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThan, *a, *b);
                            JitVal::Scalar(self.builder.ins().select(c, *a, *b))
                        }
                        ("max", [JitVal::Scalar(a), JitVal::Scalar(b)]) => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThan, *a, *b);
                            JitVal::Scalar(self.builder.ins().select(c, *a, *b))
                        }
                        ("clamp", [JitVal::Scalar(x), JitVal::Scalar(lo), JitVal::Scalar(hi)]) => {
                            let c1 = self.builder.ins().fcmp(FloatCC::GreaterThan, *x, *lo);
                            let t = self.builder.ins().select(c1, *x, *lo);
                            let c2 = self.builder.ins().fcmp(FloatCC::LessThan, t, *hi);
                            JitVal::Scalar(self.builder.ins().select(c2, t, *hi))
                        }
                        ("select", [JitVal::Scalar(c), t, e]) => {
                            let cond = self.builder.ins().fcmp(FloatCC::NotEqual, *c, zero);
                            match (t, e) {
                                (JitVal::Vec3(tv), JitVal::Vec3(ev)) => JitVal::Vec3([
                                    self.builder.ins().select(cond, tv[0], ev[0]),
                                    self.builder.ins().select(cond, tv[1], ev[1]),
                                    self.builder.ins().select(cond, tv[2], ev[2]),
                                ]),
                                (JitVal::Quat(tq), JitVal::Quat(eq)) => JitVal::Quat([
                                    self.builder.ins().select(cond, tq[0], eq[0]),
                                    self.builder.ins().select(cond, tq[1], eq[1]),
                                    self.builder.ins().select(cond, tq[2], eq[2]),
                                    self.builder.ins().select(cond, tq[3], eq[3]),
                                ]),
                                (tv, ev) => {
                                    let ts = tv.as_scalar(zero);
                                    let es = ev.as_scalar(zero);
                                    JitVal::Scalar(self.builder.ins().select(cond, ts, es))
                                }
                            }
                        }
                        ("dot", [JitVal::Vec3(a), JitVal::Vec3(b)]) => {
                            JitVal::Scalar(self.dot3(*a, *b))
                        }
                        ("norm", [JitVal::Vec3(v)]) => {
                            let sq = self.dot3(*v, *v);
                            JitVal::Scalar(self.call_helper1("aerotick_det_sqrt", sq))
                        }
                        ("norm", [JitVal::Quat(q)]) => {
                            let s0 = self.builder.ins().fmul(q[0], q[0]);
                            let s1 = self.builder.ins().fmul(q[1], q[1]);
                            let s2 = self.builder.ins().fmul(q[2], q[2]);
                            let s3 = self.builder.ins().fmul(q[3], q[3]);
                            let a01 = self.builder.ins().fadd(s0, s1);
                            let a23 = self.builder.ins().fadd(s2, s3);
                            let sq = self.builder.ins().fadd(a01, a23);
                            JitVal::Scalar(self.call_helper1("aerotick_det_sqrt", sq))
                        }
                        ("normalize", [JitVal::Vec3(v)]) => {
                            let sq = self.dot3(*v, *v);
                            let m = self.call_helper1("aerotick_det_sqrt", sq);
                            let is_z = self.builder.ins().fcmp(FloatCC::Equal, m, zero);
                            let x = self.builder.ins().fdiv(v[0], m);
                            let y = self.builder.ins().fdiv(v[1], m);
                            let z = self.builder.ins().fdiv(v[2], m);
                            JitVal::Vec3([
                                self.builder.ins().select(is_z, zero, x),
                                self.builder.ins().select(is_z, zero, y),
                                self.builder.ins().select(is_z, zero, z),
                            ])
                        }
                        ("cross", [JitVal::Vec3(a), JitVal::Vec3(b)]) => {
                            let x0 = self.builder.ins().fmul(a[1], b[2]);
                            let x1 = self.builder.ins().fmul(a[2], b[1]);
                            let y0 = self.builder.ins().fmul(a[2], b[0]);
                            let y1 = self.builder.ins().fmul(a[0], b[2]);
                            let z0 = self.builder.ins().fmul(a[0], b[1]);
                            let z1 = self.builder.ins().fmul(a[1], b[0]);
                            JitVal::Vec3([
                                self.builder.ins().fsub(x0, x1),
                                self.builder.ins().fsub(y0, y1),
                                self.builder.ins().fsub(z0, z1),
                            ])
                        }
                        ("quat_conj", [JitVal::Quat(q)]) => JitVal::Quat([
                            q[0],
                            self.builder.ins().fneg(q[1]),
                            self.builder.ins().fneg(q[2]),
                            self.builder.ins().fneg(q[3]),
                        ]),
                        ("quat_norm" | "normalize", [JitVal::Quat(q)]) => {
                            JitVal::Quat(self.quat_norm(*q))
                        }
                        ("quat_rotate", [JitVal::Quat(q), JitVal::Vec3(v)]) => {
                            let qn = self.quat_norm(*q);
                            let qv = [zero, v[0], v[1], v[2]];
                            let qc = [
                                qn[0],
                                self.builder.ins().fneg(qn[1]),
                                self.builder.ins().fneg(qn[2]),
                                self.builder.ins().fneg(qn[3]),
                            ];
                            let tmp = self.quat_mul(qn, qv);
                            let out = self.quat_mul(tmp, qc);
                            JitVal::Vec3([out[1], out[2], out[3]])
                        }
                        ("quat_integrate", [JitVal::Quat(q), JitVal::Vec3(w), JitVal::Scalar(dt_v)]) => {
                            let wq = [zero, w[0], w[1], w[2]];
                            let dq = self.quat_mul(*q, wq);
                            let half = self.f64_const(0.5);
                            let hdt = self.builder.ins().fmul(half, *dt_v);
                            let mut next_q = [zero; 4];
                            for i in 0..4 {
                                let step = self.builder.ins().fmul(dq[i], hdt);
                                next_q[i] = self.builder.ins().fadd(q[i], step);
                            }
                            JitVal::Quat(self.quat_norm(next_q))
                        }
                        ("mat3_diag", [JitVal::Scalar(x), JitVal::Scalar(y), JitVal::Scalar(z)]) => {
                            JitVal::Mat3([*x, zero, zero, zero, *y, zero, zero, zero, *z])
                        }
                        ("mat3_from_quat", [JitVal::Quat(q)]) => {
                            let [w, x, y, z] = self.quat_norm(*q);
                            let one = self.f64_const(1.0);
                            let two = self.f64_const(2.0);
                            let xx = self.builder.ins().fmul(x, x);
                            let yy = self.builder.ins().fmul(y, y);
                            let zz = self.builder.ins().fmul(z, z);
                            let xy = self.builder.ins().fmul(x, y);
                            let xz = self.builder.ins().fmul(x, z);
                            let yz = self.builder.ins().fmul(y, z);
                            let xw = self.builder.ins().fmul(x, w);
                            let yw = self.builder.ins().fmul(y, w);
                            let zw = self.builder.ins().fmul(z, w);

                            let y2z2 = self.builder.ins().fadd(yy, zz);
                            let t00 = self.builder.ins().fmul(two, y2z2);
                            let m00 = self.builder.ins().fsub(one, t00);

                            let d01 = self.builder.ins().fsub(xy, zw);
                            let m01 = self.builder.ins().fmul(two, d01);

                            let s02 = self.builder.ins().fadd(xz, yw);
                            let m02 = self.builder.ins().fmul(two, s02);

                            let s10 = self.builder.ins().fadd(xy, zw);
                            let m10 = self.builder.ins().fmul(two, s10);

                            let x2z2 = self.builder.ins().fadd(xx, zz);
                            let t11 = self.builder.ins().fmul(two, x2z2);
                            let m11 = self.builder.ins().fsub(one, t11);

                            let d12 = self.builder.ins().fsub(yz, xw);
                            let m12 = self.builder.ins().fmul(two, d12);

                            let d20 = self.builder.ins().fsub(xz, yw);
                            let m20 = self.builder.ins().fmul(two, d20);

                            let s21 = self.builder.ins().fadd(yz, xw);
                            let m21 = self.builder.ins().fmul(two, s21);

                            let x2y2 = self.builder.ins().fadd(xx, yy);
                            let t22 = self.builder.ins().fmul(two, x2y2);
                            let m22 = self.builder.ins().fsub(one, t22);

                            JitVal::Mat3([m00, m01, m02, m10, m11, m12, m20, m21, m22])
                        }
                        ("mat3_transpose", [JitVal::Mat3(m)]) => JitVal::Mat3([
                            m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8],
                        ]),
                        ("mat3_inv", [JitVal::Mat3(m)]) => JitVal::Mat3(self.mat3_inv(*m)),
                        _ => JitVal::Scalar(zero),
                    };
                    self.regs.insert(dst.clone(), res);
                }
                MirInst::ForLoop {
                    var,
                    start,
                    end,
                    body,
                } => {
                    for i in *start..*end {
                        let iv = self.f64_const(i as f64);
                        self.regs.insert(format!("%loop_{var}"), JitVal::Scalar(iv));
                        self.emit_insts(body);
                    }
                }
                MirInst::Branch {
                    cond,
                    then_insts,
                    else_insts,
                } => {
                    let c_val = self.regs.get(cond).map_or(zero, |v| v.as_scalar(zero));
                    let c_bool = self.builder.ins().fcmp(FloatCC::NotEqual, c_val, zero);
                    let then_bb = self.builder.create_block();
                    let else_bb = self.builder.create_block();
                    let merge_bb = self.builder.create_block();
                    self.builder.ins().brif(c_bool, then_bb, &[], else_bb, &[]);

                    self.builder.switch_to_block(then_bb);
                    self.builder.seal_block(then_bb);
                    self.emit_insts(then_insts);
                    self.builder.ins().jump(merge_bb, &[]);

                    self.builder.switch_to_block(else_bb);
                    self.builder.seal_block(else_bb);
                    if let Some(el) = else_insts {
                        self.emit_insts(el);
                    }
                    self.builder.ins().jump(merge_bb, &[]);

                    self.builder.switch_to_block(merge_bb);
                    self.builder.seal_block(merge_bb);
                }
            }
        }
    }

    fn dot3(&mut self, a: [Value; 3], b: [Value; 3]) -> Value {
        let p0 = self.builder.ins().fmul(a[0], b[0]);
        let p1 = self.builder.ins().fmul(a[1], b[1]);
        let p2 = self.builder.ins().fmul(a[2], b[2]);
        let s01 = self.builder.ins().fadd(p0, p1);
        self.builder.ins().fadd(s01, p2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_mir::lower_to_mir;
    use aerotick_parser::parse;
    use aerotick_sema::analyze;
    use aerotick_vm::{DeterministicVm, NumericMode};

    fn assert_jit_matches_vm(name: &str, src: &str, entities: usize, ticks: usize) {
        let (tokens, l_errs) = lex(src);
        assert!(l_errs.is_empty(), "{name}: {l_errs:?}");
        let (ast_opt, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{name}: {p_errs:?}");
        let Some(ast) = ast_opt else {
            panic!("expected AST for {name}");
        };
        let (hir, s_errs) = analyze(&ast);
        assert!(s_errs.is_empty(), "{name}: {s_errs:?}");
        let mir = lower_to_mir(&hir);

        let Ok(kernel) = CraneliftJitKernel::compile(&mir) else {
            panic!("JIT compile failed for {name}");
        };
        let Ok(mut vm_ref) = DeterministicVm::new(&mir, entities, 0.01, NumericMode::Float64) else {
            panic!("VM init failed for {name}");
        };
        let Ok(mut vm_jit) = DeterministicVm::new(&mir, entities, 0.01, NumericMode::Float64) else {
            panic!("JIT VM init failed for {name}");
        };

        for _ in 0..ticks {
            vm_ref.step_tick();
            let (prev_ptrs, next_ptrs) = vm_jit.field_ptrs_mut();
            unsafe {
                kernel.step(&prev_ptrs, &next_ptrs, vm_jit.count, vm_jit.dt);
            }
            vm_jit.commit_jit_tick();
        }
        assert_eq!(
            vm_ref.state_sha256_hex(),
            vm_jit.state_sha256_hex(),
            "SHA-256 mismatch between VM and Cranelift JIT on {name}"
        );
    }

    #[test]
    fn cranelift_jit_matches_vm_sha256_on_all_flagship_kernels() {
        assert_jit_matches_vm(
            "reference_body.aero",
            include_str!("../../../examples/reference_body.aero"),
            64,
            500,
        );
        assert_jit_matches_vm(
            "rocket_6dof.aero",
            include_str!("../../../examples/rocket_6dof.aero"),
            32,
            250,
        );
        assert_jit_matches_vm(
            "quadcopter_pid.aero",
            include_str!("../../../examples/quadcopter_pid.aero"),
            32,
            250,
        );
    }
}
