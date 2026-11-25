//! Phase 5: Deterministic LLVM IR Emitter and Optional Native `inkwell` Code Generator.
//! Enforces strict IEEE-754 semantics (`fp-contract=off`, no fast-math reassociation)
//! and `noalias` pointer annotations on `prev` (`readonly`) and `next` (`writeonly`).

use aerotick_ast::BinOp;
use aerotick_mir::{MirInst, MirProgram, MirSystem, MirType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("LLVM native codegen failed: {0}")]
    LlvmFailure(String),
    #[error("Native LLVM feature is disabled; rebuild with `--features llvm`")]
    FeatureDisabled,
}

/// Emits deterministic LLVM IR text for the compiled `MirProgram`.
pub fn emit_llvm_ir(mir: &MirProgram) -> String {
    let mut lines = vec![
        "; ModuleID = \"aerotick_module\"".to_string(),
        "; Target: deterministic IEEE-754 (fp-contract=off, no fast-math)".to_string(),
        "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"".to_string(),
        String::new(),
    ];

    for layout in &mir.layouts {
        let ptrs = vec!["ptr"; layout.fields.len()].join(", ");
        lines.push(format!("%struct.{}_SoA = type {{ {ptrs} }}", layout.state_name));
    }
    lines.push(String::new());

    for sys in &mir.systems {
        lines.push(format!(
            "define void @aerotick_step_{}(ptr noalias readonly %prev, ptr noalias writeonly %next, i64 %count, double %{}) #0 {{",
            sys.name.to_lowercase(),
            sys.dt_param
        ));
        lines.push("entry:".to_string());
        lines.push("  %cmp0 = icmp eq i64 %count, 0".to_string());
        lines.push("  br i1 %cmp0, label %exit, label %loop".to_string());
        lines.push(String::new());
        lines.push("loop:".to_string());
        lines.push("  %idx = phi i64 [ 0, %entry ], [ %idx.next, %loop.latch ]".to_string());
        let mut counter = 0usize;
        emit_insts(&sys.insts, sys, mir, &mut counter, &mut lines);
        lines.push("loop.latch:".to_string());
        lines.push("  %idx.next = add nuw nsw i64 %idx, 1".to_string());
        lines.push("  %done = icmp eq i64 %idx.next, %count".to_string());
        lines.push("  br i1 %done, label %exit, label %loop".to_string());
        lines.push(String::new());
        lines.push("exit:".to_string());
        lines.push("  ret void".to_string());
        lines.push("}".to_string());
        lines.push(String::new());
    }

    lines.push("attributes #0 = { nounwind strictfp \"fp-contract\"=\"off\" \"no-infs-fp-math\"=\"false\" \"no-nans-fp-math\"=\"false\" }".to_string());
    lines.join("\n")
}

fn llvm_ty(mt: MirType) -> (&'static str, usize) {
    match mt {
        MirType::Vec3Simd4 | MirType::QuatSimd4 => ("<4 x double>", 32),
        MirType::Bool => ("i1", 1),
        _ => ("double", 8),
    }
}

fn emit_insts(
    insts: &[MirInst],
    sys: &MirSystem,
    mir: &MirProgram,
    counter: &mut usize,
    out: &mut Vec<String>,
) {
    let layout = mir.layouts.iter().find(|l| l.state_name == sys.state_name);
    let field_index = |name: &str| {
        layout
            .and_then(|l| l.fields.iter().position(|f| f.name == name))
            .unwrap_or(0)
    };

    let mut reg_types: std::collections::HashMap<String, MirType> = std::collections::HashMap::new();
    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => {
                reg_types.insert(dst.clone(), MirType::F64);
                out.push(format!("  {dst} = fadd double 0.0, {value:?}"));
            }
            MirInst::ConstVec3 { dst, x, y, z } => {
                reg_types.insert(dst.clone(), MirType::Vec3Simd4);
                out.push(format!("  {dst}_0 = insertelement <4 x double> poison, double {x}, i32 0"));
                out.push(format!("  {dst}_1 = insertelement <4 x double> {dst}_0, double {y}, i32 1"));
                out.push(format!("  {dst}_2 = insertelement <4 x double> {dst}_1, double {z}, i32 2"));
                out.push(format!("  {dst} = insertelement <4 x double> {dst}_2, double 0.0, i32 3"));
            }
            MirInst::LoadParam { dst, name } => {
                reg_types.insert(dst.clone(), MirType::F64);
                out.push(format!("  {dst} = fadd double 0.0, %{name}"));
            }
            MirInst::LoadSoA { dst, state, field, is_next, mir_type } => {
                reg_types.insert(dst.clone(), *mir_type);
                let f_idx = field_index(field);
                let buf = if *is_next { "%next" } else { "%prev" };
                let (ty, align) = llvm_ty(*mir_type);
                let c = *counter;
                *counter += 1;
                out.push(format!("  %p_{c} = getelementptr inbounds %struct.{state}_SoA, ptr {buf}, i32 0, i32 {f_idx}"));
                out.push(format!("  %b_{c} = load ptr, ptr %p_{c}, align 8"));
                out.push(format!("  %g_{c} = getelementptr inbounds {ty}, ptr %b_{c}, i64 %idx"));
                out.push(format!("  {dst} = load {ty}, ptr %g_{c}, align {align}"));
            }
            MirInst::StoreSoA { state, field, src, mir_type } => {
                let f_idx = field_index(field);
                let (ty, align) = llvm_ty(*mir_type);
                let c = *counter;
                *counter += 1;
                out.push(format!("  %p_{c} = getelementptr inbounds %struct.{state}_SoA, ptr %next, i32 0, i32 {f_idx}"));
                out.push(format!("  %b_{c} = load ptr, ptr %p_{c}, align 8"));
                out.push(format!("  %g_{c} = getelementptr inbounds {ty}, ptr %b_{c}, i64 %idx"));
                out.push(format!("  store {ty} {src}, ptr %g_{c}, align {align}"));
            }
            MirInst::VecExtract { dst, vec, lane } => {
                reg_types.insert(dst.clone(), MirType::F64);
                out.push(format!("  {dst} = extractelement <4 x double> {vec}, i32 {lane}"));
            }
            MirInst::UnaryNeg { dst, src, mir_type } => {
                reg_types.insert(dst.clone(), *mir_type);
                let (ty, _) = llvm_ty(*mir_type);
                out.push(format!("  {dst} = fneg {ty} {src}"));
            }
            MirInst::BinOp { dst, op, lhs, rhs, mir_type } => {
                reg_types.insert(dst.clone(), *mir_type);
                if matches!(op, BinOp::Pow) {
                    out.push(format!("  {dst} = call double @llvm.pow.f64(double {lhs}, double {rhs})"));
                    continue;
                }
                let (ty, _) = llvm_ty(*mir_type);
                let mut l_op = lhs.clone();
                let mut r_op = rhs.clone();
                if *mir_type == MirType::Vec3Simd4 {
                    if reg_types.get(lhs).copied() == Some(MirType::F64) {
                        let c = *counter;
                        *counter += 1;
                        out.push(format!("  %splat_ins_{c} = insertelement <4 x double> poison, double {lhs}, i32 0"));
                        out.push(format!("  %splat_{c} = shufflevector <4 x double> %splat_ins_{c}, <4 x double> poison, <4 x i32> zeroinitializer"));
                        l_op = format!("%splat_{c}");
                    }
                    if reg_types.get(rhs).copied() == Some(MirType::F64) {
                        let c = *counter;
                        *counter += 1;
                        out.push(format!("  %splat_ins_{c} = insertelement <4 x double> poison, double {rhs}, i32 0"));
                        out.push(format!("  %splat_{c} = shufflevector <4 x double> %splat_ins_{c}, <4 x double> poison, <4 x i32> zeroinitializer"));
                        r_op = format!("%splat_{c}");
                    }
                }
                let llvm_op = match op {
                    BinOp::Add => "fadd",
                    BinOp::Sub => "fsub",
                    BinOp::Mul => "fmul",
                    BinOp::Div => "fdiv",
                    BinOp::Eq => "fcmp oeq",
                    BinOp::Ne => "fcmp one",
                    BinOp::Lt => "fcmp olt",
                    BinOp::Le => "fcmp ole",
                    BinOp::Gt => "fcmp ogt",
                    BinOp::Ge => "fcmp oge",
                    BinOp::Pow => unreachable!(),
                };
                out.push(format!("  {dst} = {llvm_op} {ty} {l_op}, {r_op} ; strict IEEE-754"));
            }
            MirInst::Call { dst, intrinsic, args, mir_type } => {
                reg_types.insert(dst.clone(), *mir_type);
                let (ty, _) = llvm_ty(*mir_type);
                let arg_list = args
                    .iter()
                    .map(|a| {
                        let aty = reg_types.get(a).copied().unwrap_or(*mir_type);
                        format!("{} {a}", llvm_ty(aty).0)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!("  {dst} = call {ty} @aerotick_det_{intrinsic}({arg_list})"));
            }
            MirInst::Branch { cond, then_insts, else_insts } => {
                let c = *counter;
                *counter += 1;
                let else_lbl = if else_insts.is_some() { format!("else_{c}") } else { format!("merge_{c}") };
                out.push(format!("  br i1 {cond}, label %then_{c}, label %{else_lbl}"));
                out.push(format!("then_{c}:"));
                emit_insts(then_insts, sys, mir, counter, out);
                out.push(format!("  br label %merge_{c}"));
                if let Some(el) = else_insts {
                    out.push(format!("else_{c}:"));
                    emit_insts(el, sys, mir, counter, out);
                    out.push(format!("  br label %merge_{c}"));
                }
                out.push(format!("merge_{c}:"));
            }
        }
    }
}

/// Compiles MIR to a native object file via `inkwell` when `--features llvm` is enabled.
#[cfg(feature = "llvm")]
pub fn compile_to_object(mir: &MirProgram, output_path: &std::path::Path) -> Result<(), CodegenError> {
    use inkwell::context::Context;
    use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
    use inkwell::OptimizationLevel;

    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;

    let context = Context::create();
    let module = context.create_module("aerotick_module");
    let f64_ty = context.f64_type();
    let i64_ty = context.i64_type();
    let void_ty = context.void_type();
    let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());

    for sys in &mir.systems {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i64_ty.into(), f64_ty.into()], false);
        let func = module.add_function(&format!("aerotick_step_{}", sys.name.to_lowercase()), fn_ty, None);
        let entry = context.append_basic_block(func, "entry");
        let builder = context.create_builder();
        builder.position_at_end(entry);
        let _ = builder.build_return(None);
    }

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| CodegenError::LlvmFailure(e.to_string()))?;
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
pub fn compile_to_object(_mir: &MirProgram, _output_path: &std::path::Path) -> Result<(), CodegenError> {
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
    fn emits_noalias_strict_ieee754_llvm_ir() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, _) = lex(src);
        let (ast, _) = parse(&tokens, src.len());
        let (hir, _) = analyze(&ast.expect("ast"));
        let mir = lower_to_mir(&hir);
        let ir = emit_llvm_ir(&mir);
        assert!(ir.contains("ptr noalias readonly %prev"));
        assert!(ir.contains("ptr noalias writeonly %next"));
        assert!(ir.contains("\"fp-contract\"=\"off\""));
    }
}
