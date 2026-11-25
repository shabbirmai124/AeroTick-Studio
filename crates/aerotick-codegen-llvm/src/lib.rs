#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 5A: Deterministic LLVM IR Emitter & Optional Native `inkwell` Code Generator.
//! Supports:
//! - Single-entity & batched SIMD kernel emission (`@aerotick_step_<sys>` & `@aerotick_step_batch_<sys>`)
//! - Strict IEEE-754 semantics (`"fp-contract"="off"`, no fast-math reassociation)
//! - `noalias readonly` (`prev`) and `noalias writeonly` (`next`) with `align 32` guarantees
//! - 6-DoF `QuatSimd4`, `Mat3Simd12`, and branchless `Select` lowering
//! - Self-contained `declare` emission for `libm` & 6-DoF runtime trampolines

use aerotick_ast::BinOp;
use aerotick_mir::{MirInst, MirProgram, MirSystem, MirType};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("LLVM native codegen failed: {0}")]
    LlvmFailure(String),
    #[error("Native LLVM feature is disabled; rebuild with `--features llvm`")]
    FeatureDisabled,
}

fn fmt_llvm_f64(value: f64) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    let s = format!("{value:?}");
    if let Some((mant, exp)) = s.split_once('e') {
        if mant.contains('.') {
            format!("{mant}e{exp}")
        } else {
            format!("{mant}.0e{exp}")
        }
    } else if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

fn llvm_ty(mt: MirType) -> (&'static str, usize) {
    match mt {
        MirType::Vec3Simd4 | MirType::QuatSimd4 => ("<4 x double>", 32),
        MirType::Mat3Simd12 => ("[3 x <4 x double>]", 32),
        MirType::Mat4Simd16 => ("[4 x <4 x double>]", 32),
        MirType::Bool => ("i1", 1),
        MirType::F64 => ("double", 8),
    }
}

fn type_suffix(mt: MirType) -> &'static str {
    match mt {
        MirType::F64 => "f64",
        MirType::Vec3Simd4 => "vec3",
        MirType::QuatSimd4 => "quat",
        MirType::Mat3Simd12 => "mat3",
        MirType::Mat4Simd16 => "mat4",
        MirType::Bool => "i1",
    }
}

/// Emits self-contained, deterministic LLVM IR text (single-step + batched SIMD loop) for `MirProgram`.
pub fn emit_llvm_ir(mir: &MirProgram) -> String {
    let mut header = vec![
        "; ModuleID = \"aerotick_ultra_v2\"".to_string(),
        "; Target: deterministic IEEE-754 (fp-contract=off, no fast-math, align 32 SIMD4)".to_string(),
        "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"".to_string(),
        String::new(),
    ];

    for layout in &mir.layouts {
        let ptrs = vec!["ptr"; layout.fields.len()].join(", ");
        header.push(format!(
            "%struct.{}_SoA = type {{ {ptrs} }}",
            layout.state_name
        ));
    }
    header.push(String::new());

    let mut decls: BTreeSet<String> = BTreeSet::new();
    let mut body_lines: Vec<String> = Vec::new();

    for sys in &mir.systems {
        let sys_lower = sys.name.to_lowercase();
        // Batched multi-entity SIMD entrypoint (zero aliasing across SoA buffers)
        body_lines.push(format!(
            "define void @aerotick_step_batch_{sys_lower}(ptr noalias readonly align 32 %prev, ptr noalias writeonly align 32 %next, i64 %count, double %{}) #0 {{",
            sys.dt_param
        ));
        body_lines.push("entry:".to_string());
        body_lines.push("  %cmp0 = icmp eq i64 %count, 0".to_string());
        body_lines.push("  br i1 %cmp0, label %exit, label %loop".to_string());
        body_lines.push(String::new());
        body_lines.push("loop:".to_string());
        body_lines.push("  %idx = phi i64 [ 0, %entry ], [ %idx.next, %loop.latch ]".to_string());

        let mut counter = 0usize;
        let mut reg_types: HashMap<String, MirType> = HashMap::new();
        let mut rename_map: HashMap<String, String> = HashMap::new();
        emit_insts(
            &sys.insts,
            sys,
            mir,
            &mut counter,
            &mut reg_types,
            &mut rename_map,
            &mut decls,
            &mut body_lines,
        );
        body_lines.push("  br label %loop.latch".to_string());
        body_lines.push(String::new());
        body_lines.push("loop.latch:".to_string());
        body_lines.push("  %idx.next = add nuw nsw i64 %idx, 1".to_string());
        body_lines.push("  %done = icmp eq i64 %idx.next, %count".to_string());
        body_lines.push("  br i1 %done, label %exit, label %loop, !llvm.loop !1".to_string());
        body_lines.push(String::new());
        body_lines.push("exit:".to_string());
        body_lines.push("  ret void".to_string());
        body_lines.push("}".to_string());
        body_lines.push(String::new());

        // Single-entity / count entrypoint forwarding to batched SIMD kernel
        body_lines.push(format!(
            "define void @aerotick_step_{sys_lower}(ptr noalias readonly align 32 %prev, ptr noalias writeonly align 32 %next, i64 %count, double %{}) #0 {{",
            sys.dt_param
        ));
        body_lines.push("entry:".to_string());
        body_lines.push(format!(
            "  call void @aerotick_step_batch_{sys_lower}(ptr noalias readonly align 32 %prev, ptr noalias writeonly align 32 %next, i64 %count, double %{})",
            sys.dt_param
        ));
        body_lines.push("  ret void".to_string());
        body_lines.push("}".to_string());
        body_lines.push(String::new());
    }

    if !decls.is_empty() {
        for d in decls {
            header.push(d);
        }
        header.push(String::new());
    }

    header.extend(body_lines);
    header.push(
        "attributes #0 = { nounwind \"fp-contract\"=\"off\" \"no-infs-fp-math\"=\"false\" \"no-nans-fp-math\"=\"false\" \"approx-func-fp-math\"=\"false\" \"unsafe-fp-math\"=\"false\" }"
            .to_string(),
    );
    header.push("!1 = distinct !{!1, !2}".to_string());
    header.push("!2 = !{!\"llvm.loop.vectorize.enable\", i1 true}".to_string());
    header.join("\n")
}

fn resolve_reg(name: &str, rename_map: &HashMap<String, String>) -> String {
    rename_map
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

fn fresh_dst(dst: &str, suffix: &str, rename_map: &mut HashMap<String, String>) -> String {
    if suffix.is_empty() {
        dst.to_string()
    } else {
        let renamed = format!("{dst}_{suffix}");
        rename_map.insert(dst.to_string(), renamed.clone());
        renamed
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_insts(
    insts: &[MirInst],
    sys: &MirSystem,
    mir: &MirProgram,
    counter: &mut usize,
    reg_types: &mut HashMap<String, MirType>,
    rename_map: &mut HashMap<String, String>,
    decls: &mut BTreeSet<String>,
    out: &mut Vec<String>,
) {
    emit_insts_with_suffix(
        insts, sys, mir, counter, reg_types, rename_map, decls, out, "",
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_insts_with_suffix(
    insts: &[MirInst],
    sys: &MirSystem,
    mir: &MirProgram,
    counter: &mut usize,
    reg_types: &mut HashMap<String, MirType>,
    rename_map: &mut HashMap<String, String>,
    decls: &mut BTreeSet<String>,
    out: &mut Vec<String>,
    suffix: &str,
) {
    let layout = mir.layouts.iter().find(|l| l.state_name == sys.state_name);
    let field_index = |name: &str| {
        layout
            .and_then(|l| l.fields.iter().position(|f| f.name == name))
            .unwrap_or(0)
    };

    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => {
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::F64);
                out.push(format!("  {d} = fadd double 0.0, {}", fmt_llvm_f64(*value)));
            }
            MirInst::ConstVec3 { dst, x, y, z } => {
                let rx = resolve_reg(x, rename_map);
                let ry = resolve_reg(y, rename_map);
                let rz = resolve_reg(z, rename_map);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::Vec3Simd4);
                out.push(format!(
                    "  {d}_0 = insertelement <4 x double> poison, double {rx}, i32 0"
                ));
                out.push(format!(
                    "  {d}_1 = insertelement <4 x double> {d}_0, double {ry}, i32 1"
                ));
                out.push(format!(
                    "  {d}_2 = insertelement <4 x double> {d}_1, double {rz}, i32 2"
                ));
                out.push(format!(
                    "  {d} = insertelement <4 x double> {d}_2, double 0.0, i32 3"
                ));
            }
            MirInst::ConstQuat { dst, w, x, y, z } => {
                let rw = resolve_reg(w, rename_map);
                let rx = resolve_reg(x, rename_map);
                let ry = resolve_reg(y, rename_map);
                let rz = resolve_reg(z, rename_map);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::QuatSimd4);
                out.push(format!(
                    "  {d}_0 = insertelement <4 x double> poison, double {rw}, i32 0"
                ));
                out.push(format!(
                    "  {d}_1 = insertelement <4 x double> {d}_0, double {rx}, i32 1"
                ));
                out.push(format!(
                    "  {d}_2 = insertelement <4 x double> {d}_1, double {ry}, i32 2"
                ));
                out.push(format!(
                    "  {d} = insertelement <4 x double> {d}_2, double {rz}, i32 3"
                ));
            }
            MirInst::ConstMat3 { dst, elems } => {
                let resolved: Vec<String> = elems
                    .iter()
                    .map(|e| format!("double {}", resolve_reg(e, rename_map)))
                    .collect();
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::Mat3Simd12);
                decls.insert(
                    "declare [3 x <4 x double>] @aerotick_mat3_pack(double, double, double, double, double, double, double, double, double)".to_string(),
                );
                out.push(format!(
                    "  {d} = call [3 x <4 x double>] @aerotick_mat3_pack({})",
                    resolved.join(", ")
                ));
            }
            MirInst::LoadParam { dst, name } => {
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::F64);
                out.push(format!("  {d} = fadd double 0.0, %{name}"));
            }
            MirInst::LoadSoA {
                dst,
                state,
                field,
                is_next,
                mir_type,
            } => {
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), *mir_type);
                let f_idx = field_index(field);
                let buf = if *is_next { "%next" } else { "%prev" };
                let (ty, align) = llvm_ty(*mir_type);
                let c = *counter;
                *counter += 1;
                out.push(format!(
                    "  %p_{c} = getelementptr inbounds %struct.{state}_SoA, ptr {buf}, i32 0, i32 {f_idx}"
                ));
                out.push(format!("  %b_{c} = load ptr, ptr %p_{c}, align 8"));
                out.push(format!(
                    "  %g_{c} = getelementptr inbounds {ty}, ptr %b_{c}, i64 %idx"
                ));
                out.push(format!("  {d} = load {ty}, ptr %g_{c}, align {align}"));
            }
            MirInst::StoreSoA {
                state,
                field,
                src,
                mir_type,
            } => {
                let rsrc = resolve_reg(src, rename_map);
                let f_idx = field_index(field);
                let (ty, align) = llvm_ty(*mir_type);
                let c = *counter;
                *counter += 1;
                out.push(format!(
                    "  %p_{c} = getelementptr inbounds %struct.{state}_SoA, ptr %next, i32 0, i32 {f_idx}"
                ));
                out.push(format!("  %b_{c} = load ptr, ptr %p_{c}, align 8"));
                out.push(format!(
                    "  %g_{c} = getelementptr inbounds {ty}, ptr %b_{c}, i64 %idx"
                ));
                out.push(format!("  store {ty} {rsrc}, ptr %g_{c}, align {align}"));
            }
            MirInst::VecExtract { dst, vec, lane } => {
                let rvec = resolve_reg(vec, rename_map);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), MirType::F64);
                out.push(format!(
                    "  {d} = extractelement <4 x double> {rvec}, i32 {lane}"
                ));
            }
            MirInst::Select {
                dst,
                cond,
                then_val,
                else_val,
                mir_type,
            } => {
                let rcond = resolve_reg(cond, rename_map);
                let rthen = resolve_reg(then_val, rename_map);
                let relse = resolve_reg(else_val, rename_map);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), *mir_type);
                let (ty, _) = llvm_ty(*mir_type);
                out.push(format!(
                    "  {d} = select i1 {rcond}, {ty} {rthen}, {ty} {relse}"
                ));
            }
            MirInst::UnaryNeg { dst, src, mir_type } => {
                let rsrc = resolve_reg(src, rename_map);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), *mir_type);
                if *mir_type == MirType::Mat3Simd12 {
                    decls.insert(
                        "declare [3 x <4 x double>] @aerotick_det_mat3_neg([3 x <4 x double>])"
                            .to_string(),
                    );
                    out.push(format!(
                        "  {d} = call [3 x <4 x double>] @aerotick_det_mat3_neg([3 x <4 x double>] {rsrc})"
                    ));
                } else {
                    let (ty, _) = llvm_ty(*mir_type);
                    out.push(format!("  {d} = fneg {ty} {rsrc}"));
                }
            }
            MirInst::BinOp {
                dst,
                op,
                lhs,
                rhs,
                mir_type,
            } => {
                let rlhs = resolve_reg(lhs, rename_map);
                let rrhs = resolve_reg(rhs, rename_map);
                let l_ty = reg_types.get(&rlhs).copied().unwrap_or(MirType::F64);
                let r_ty = reg_types.get(&rrhs).copied().unwrap_or(MirType::F64);
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), *mir_type);

                if matches!(op, BinOp::Pow) {
                    decls.insert("declare double @llvm.pow.f64(double, double)".to_string());
                    out.push(format!(
                        "  {d} = call double @llvm.pow.f64(double {rlhs}, double {rrhs})"
                    ));
                    continue;
                }

                // Special 6-DoF matrix & quaternion products
                if matches!(op, BinOp::Mul)
                    && l_ty == MirType::QuatSimd4
                    && r_ty == MirType::QuatSimd4
                {
                    decls.insert(
                        "declare <4 x double> @aerotick_det_quat_mul(<4 x double>, <4 x double>)"
                            .to_string(),
                    );
                    out.push(format!(
                        "  {d} = call <4 x double> @aerotick_det_quat_mul(<4 x double> {rlhs}, <4 x double> {rrhs})"
                    ));
                    continue;
                }
                if matches!(op, BinOp::Mul)
                    && l_ty == MirType::Mat3Simd12
                    && r_ty == MirType::Vec3Simd4
                {
                    decls.insert(
                        "declare <4 x double> @aerotick_det_mat3_mul_vec3([3 x <4 x double>], <4 x double>)".to_string(),
                    );
                    out.push(format!(
                        "  {d} = call <4 x double> @aerotick_det_mat3_mul_vec3([3 x <4 x double>] {rlhs}, <4 x double> {rrhs})"
                    ));
                    continue;
                }
                if *mir_type == MirType::Mat3Simd12 {
                    let op_name = match op {
                        BinOp::Add => "add",
                        BinOp::Sub => "sub",
                        BinOp::Mul => "mul",
                        BinOp::Div => "div",
                        _ => "add",
                    };
                    let (lt_s, _) = llvm_ty(l_ty);
                    let (rt_s, _) = llvm_ty(r_ty);
                    let sym = format!(
                        "@aerotick_det_mat3_{op_name}_{}_{}",
                        type_suffix(l_ty),
                        type_suffix(r_ty)
                    );
                    decls.insert(format!("declare [3 x <4 x double>] {sym}({lt_s}, {rt_s})"));
                    out.push(format!(
                        "  {d} = call [3 x <4 x double>] {sym}({lt_s} {rlhs}, {rt_s} {rrhs})"
                    ));
                    continue;
                }

                // Comparisons operate on operand type (typically `double`), producing `i1`
                let is_cmp = matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                );
                let op_mir_ty = if is_cmp { l_ty } else { *mir_type };
                let (ty, _) = llvm_ty(op_mir_ty);

                let mut l_op = rlhs;
                let mut r_op = rrhs;
                if matches!(op_mir_ty, MirType::Vec3Simd4 | MirType::QuatSimd4) {
                    if l_ty == MirType::F64 {
                        let c = *counter;
                        *counter += 1;
                        out.push(format!(
                            "  %splat_ins_{c} = insertelement <4 x double> poison, double {l_op}, i32 0"
                        ));
                        out.push(format!(
                            "  %splat_{c} = shufflevector <4 x double> %splat_ins_{c}, <4 x double> poison, <4 x i32> zeroinitializer"
                        ));
                        l_op = format!("%splat_{c}");
                    }
                    if r_ty == MirType::F64 {
                        let c = *counter;
                        *counter += 1;
                        out.push(format!(
                            "  %splat_ins_{c} = insertelement <4 x double> poison, double {r_op}, i32 0"
                        ));
                        out.push(format!(
                            "  %splat_{c} = shufflevector <4 x double> %splat_ins_{c}, <4 x double> poison, <4 x i32> zeroinitializer"
                        ));
                        r_op = format!("%splat_{c}");
                    }
                }
                let llvm_op = match op {
                    BinOp::Add => Some("fadd"),
                    BinOp::Sub => Some("fsub"),
                    BinOp::Mul => Some("fmul"),
                    BinOp::Div => Some("fdiv"),
                    BinOp::Eq => Some("fcmp oeq"),
                    BinOp::Ne => Some("fcmp one"),
                    BinOp::Lt => Some("fcmp olt"),
                    BinOp::Le => Some("fcmp ole"),
                    BinOp::Gt => Some("fcmp ogt"),
                    BinOp::Ge => Some("fcmp oge"),
                    BinOp::Pow => None,
                };
                if let Some(op_str) = llvm_op {
                    out.push(format!(
                        "  {d} = {op_str} {ty} {l_op}, {r_op} ; strict IEEE-754"
                    ));
                }
            }
            MirInst::Call {
                dst,
                intrinsic,
                args,
                mir_type,
            } => {
                let resolved_args: Vec<(String, MirType)> = args
                    .iter()
                    .map(|a| {
                        let ra = resolve_reg(a, rename_map);
                        let aty = reg_types.get(&ra).copied().unwrap_or(*mir_type);
                        (ra, aty)
                    })
                    .collect();
                let d = fresh_dst(dst, suffix, rename_map);
                reg_types.insert(d.clone(), *mir_type);

                let (ty, _) = llvm_ty(*mir_type);
                let sig_suffix = if matches!(
                    intrinsic.as_str(),
                    "abs" | "min" | "max" | "clamp" | "norm" | "select"
                ) {
                    format!(
                        "_{}",
                        resolved_args
                            .iter()
                            .map(|(_, t)| type_suffix(*t))
                            .collect::<Vec<_>>()
                            .join("_")
                    )
                } else {
                    String::new()
                };
                let sym = format!("@aerotick_det_{intrinsic}{sig_suffix}");
                let decl_arg_tys = resolved_args
                    .iter()
                    .map(|(_, t)| llvm_ty(*t).0)
                    .collect::<Vec<_>>()
                    .join(", ");
                decls.insert(format!("declare {ty} {sym}({decl_arg_tys})"));

                let arg_list = resolved_args
                    .iter()
                    .map(|(a, t)| format!("{} {a}", llvm_ty(*t).0))
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!("  {d} = call {ty} {sym}({arg_list})"));
            }
            MirInst::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                out.push(format!("  ; bounded loop {var} in {start}..{end}"));
                let loop_id = *counter;
                *counter += 1;
                for iter_idx in *start..*end {
                    let iter_suffix = if suffix.is_empty() {
                        format!("l{loop_id}_{iter_idx}")
                    } else {
                        format!("{suffix}_l{loop_id}_{iter_idx}")
                    };
                    let mut scoped_rename = rename_map.clone();
                    emit_insts_with_suffix(
                        body,
                        sys,
                        mir,
                        counter,
                        reg_types,
                        &mut scoped_rename,
                        decls,
                        out,
                        &iter_suffix,
                    );
                }
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
                let rcond = resolve_reg(cond, rename_map);
                let c = *counter;
                *counter += 1;
                let else_lbl = if else_insts.is_some() {
                    format!("else_{c}")
                } else {
                    format!("merge_{c}")
                };
                out.push(format!(
                    "  br i1 {rcond}, label %then_{c}, label %{else_lbl}"
                ));
                out.push(format!("then_{c}:"));
                emit_insts_with_suffix(
                    then_insts, sys, mir, counter, reg_types, rename_map, decls, out, suffix,
                );
                out.push(format!("  br label %merge_{c}"));
                if let Some(el) = else_insts {
                    out.push(format!("else_{c}:"));
                    emit_insts_with_suffix(
                        el, sys, mir, counter, reg_types, rename_map, decls, out, suffix,
                    );
                    out.push(format!("  br label %merge_{c}"));
                }
                out.push(format!("merge_{c}:"));
            }
        }
    }
}

/// Compiles MIR to a native object file via `inkwell` when `--features llvm` is enabled.
/// Parses the verified deterministic LLVM IR module and emits a PIC native `.o` file.
#[cfg(feature = "llvm")]
pub fn compile_to_object(
    mir: &MirProgram,
    output_path: &std::path::Path,
) -> Result<(), CodegenError> {
    use inkwell::OptimizationLevel;
    use inkwell::context::Context;
    use inkwell::memory_buffer::MemoryBuffer;
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };

    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;

    let ir_text = emit_llvm_ir(mir);
    let mem_buf =
        MemoryBuffer::create_from_memory_range_copy(ir_text.as_bytes(), "aerotick_ultra_v2");
    let context = Context::create();
    let module = context
        .create_module_from_ir(mem_buf)
        .map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;

    module
        .verify()
        .map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;

    let triple = TargetMachine::get_default_triple();
    let target =
        Target::from_triple(&triple).map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;
    let machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError::LlvmFailure("failed to create target machine".into()))?;

    machine
        .write_to_file(&module, FileType::Object, output_path)
        .map_err(|e| CodegenError::LlvmFailure(e.to_string()))
}

#[cfg(not(feature = "llvm"))]
pub fn compile_to_object(
    _mir: &MirProgram,
    _output_path: &std::path::Path,
) -> Result<(), CodegenError> {
    Err(CodegenError::FeatureDisabled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_mir::lower_to_mir;
    use aerotick_parser::parse;
    use aerotick_sema::analyze;

    #[test]
    fn emits_noalias_strict_ieee754_batched_llvm_ir() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, _) = lex(src);
        let (ast_opt, _) = parse(&tokens, src.len());
        let Some(ast) = ast_opt else {
            panic!("reference_body.aero failed to parse");
        };
        let (hir, _) = analyze(&ast);
        let mir = lower_to_mir(&hir);
        let ir = emit_llvm_ir(&mir);
        assert!(ir.contains("@aerotick_step_batch_integrate(ptr noalias readonly align 32 %prev"));
        assert!(ir.contains("ptr noalias writeonly align 32 %next"));
        assert!(ir.contains("br label %loop.latch"));
        assert!(ir.contains("\"fp-contract\"=\"off\""));
    }

    #[test]
    fn emits_self_contained_6dof_declarations() {
        let src = include_str!("../../../examples/rocket_6dof.aero");
        let (tokens, _) = lex(src);
        let (ast_opt, _) = parse(&tokens, src.len());
        let Some(ast) = ast_opt else {
            panic!("rocket_6dof.aero failed to parse");
        };
        let (hir, _) = analyze(&ast);
        let mir = lower_to_mir(&hir);
        let ir = emit_llvm_ir(&mir);
        assert!(ir.contains("declare <4 x double> @aerotick_det_quat_rotate("));
        assert!(ir.contains("declare <4 x double> @aerotick_det_quat_integrate("));
        assert!(ir.contains("declare [3 x <4 x double>] @aerotick_det_mat3_inv("));
        assert!(ir.contains("declare <4 x double> @aerotick_det_mat3_mul_vec3("));
        assert!(ir.contains("fcmp ogt double"));
    }
}
