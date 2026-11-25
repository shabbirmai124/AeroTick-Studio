//! Phase 4: Lowering Typed HIR into Unit-Erased, Cache-Aligned SoA Mid-Level IR (MIR).
//! Strips all SI unit metadata and computes Structure-of-Arrays (SoA) byte layouts.

use aerotick_ast::BinOp;
use aerotick_sema::{HirExpr, HirProgram, HirStmt, ValueType};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirType {
    F64,
    Vec3Simd4,
    QuatSimd4,
    Mat4Simd16,
    Bool,
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F64 => write!(f, "f64"),
            Self::Vec3Simd4 => write!(f, "vec3_simd4"),
            Self::QuatSimd4 => write!(f, "quat_simd4"),
            Self::Mat4Simd16 => write!(f, "mat4_simd16"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoAFieldLayout {
    pub name: String,
    pub mir_type: MirType,
    pub lanes: usize,
    pub active_lanes: usize,
    pub descriptor_offset_bytes: usize,
    pub bytes_per_entity: usize,
    pub alignment_bytes: usize,
    pub padding_bytes_per_entity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoAStateLayout {
    pub state_name: String,
    pub fields: Vec<SoAFieldLayout>,
    pub total_bytes_per_entity: usize,
    pub total_padding_bytes_per_entity: usize,
}

#[derive(Debug, Clone)]
pub enum MirInst {
    ConstF64 { dst: String, value: f64 },
    ConstVec3 { dst: String, x: String, y: String, z: String },
    LoadParam { dst: String, name: String },
    LoadSoA { dst: String, state: String, field: String, is_next: bool, mir_type: MirType },
    StoreSoA { state: String, field: String, src: String, mir_type: MirType },
    VecExtract { dst: String, vec: String, lane: usize },
    UnaryNeg { dst: String, src: String, mir_type: MirType },
    BinOp { dst: String, op: BinOp, lhs: String, rhs: String, mir_type: MirType },
    Call { dst: String, intrinsic: String, args: Vec<String>, mir_type: MirType },
    Branch { cond: String, then_insts: Vec<MirInst>, else_insts: Option<Vec<MirInst>> },
}

#[derive(Debug, Clone)]
pub struct MirSystem {
    pub name: String,
    pub dt_param: String,
    pub state_name: String,
    pub insts: Vec<MirInst>,
}

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub layouts: Vec<SoAStateLayout>,
    pub systems: Vec<MirSystem>,
}

impl MirProgram {
    pub fn format_ir(&self) -> String {
        let mut lines = vec!["; AeroTick Unit-Erased SoA MIR".to_string(), String::new()];
        for layout in &self.layouts {
            lines.push(format!(
                "soa_layout {} ({} bytes/entity, {} padding bytes) {{",
                layout.state_name, layout.total_bytes_per_entity, layout.total_padding_bytes_per_entity
            ));
            for f in &layout.fields {
                lines.push(format!(
                    "  {:<10} : {:<11} | desc_off={:>2}B | align={:>2}B | stride={:>2}B | pad={}B",
                    f.name,
                    f.mir_type.to_string(),
                    f.descriptor_offset_bytes,
                    f.alignment_bytes,
                    f.bytes_per_entity,
                    f.padding_bytes_per_entity
                ));
            }
            lines.push("}".to_string());
            lines.push(String::new());
        }

        for sys in &self.systems {
            lines.push(format!(
                "kernel @{}(%prev: *const {}_SoA, %next: *mut {}_SoA, %count: usize, %{}: f64) {{",
                sys.name, sys.state_name, sys.state_name, sys.dt_param
            ));
            lines.push("  for %i in 0..%count {".to_string());
            dump_insts(&sys.insts, "    ", &mut lines);
            lines.push("  }".to_string());
            lines.push("}".to_string());
            lines.push(String::new());
        }
        lines.join("\n")
    }
}

fn dump_insts(insts: &[MirInst], indent: &str, out: &mut Vec<String>) {
    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => out.push(format!("{indent}{dst} = const.f64 {value}")),
            MirInst::ConstVec3 { dst, x, y, z } => out.push(format!("{indent}{dst} = simd4.pack <{x}, {y}, {z}, 0.0>")),
            MirInst::LoadParam { dst, name } => out.push(format!("{indent}{dst} = param.f64 %{name}")),
            MirInst::LoadSoA { dst, state, field, is_next, mir_type } => {
                let buf = if *is_next { "next" } else { "prev" };
                out.push(format!("{indent}{dst} = soa.load.{mir_type} @{buf}.{state}.{field}[%i]"));
            }
            MirInst::StoreSoA { state, field, src, mir_type } => {
                out.push(format!("{indent}soa.store.{mir_type} @next.{state}.{field}[%i], {src}"));
            }
            MirInst::VecExtract { dst, vec, lane } => out.push(format!("{indent}{dst} = simd4.extract {vec}, lane {lane}")),
            MirInst::UnaryNeg { dst, src, mir_type } => out.push(format!("{indent}{dst} = neg.{mir_type} {src}")),
            MirInst::BinOp { dst, op, lhs, rhs, mir_type } => out.push(format!("{indent}{dst} = {op:?}.{mir_type} {lhs}, {rhs}")),
            MirInst::Call { dst, intrinsic, args, mir_type } => out.push(format!("{indent}{dst} = call @det_{intrinsic}({}) : {mir_type}", args.join(", "))),
            MirInst::Branch { cond, then_insts, else_insts } => {
                out.push(format!("{indent}if {cond} {{"));
                dump_insts(then_insts, &format!("{indent}  "), out);
                if let Some(el) = else_insts {
                    out.push(format!("{indent}}} else {{"));
                    dump_insts(el, &format!("{indent}  "), out);
                }
                out.push(format!("{indent}}}"));
            }
        }
    }
}

fn infer_mir_type(ty: &ValueType) -> MirType {
    match ty {
        ValueType::Vec3(_) => MirType::Vec3Simd4,
        ValueType::Quat => MirType::QuatSimd4,
        ValueType::Mat4 => MirType::Mat4Simd16,
        ValueType::Bool => MirType::Bool,
        _ => MirType::F64,
    }
}

pub fn lower_to_mir(hir: &HirProgram) -> MirProgram {
    let layouts = hir
        .states
        .iter()
        .map(|st| {
            let mut fields = Vec::with_capacity(st.fields.len());
            let mut desc_off = 0usize;
            let mut total_bytes = 0usize;
            let mut total_pad = 0usize;

            for f in &st.fields {
                let mir_type = infer_mir_type(&f.ty);
                let (lanes, active_lanes, align) = match mir_type {
                    MirType::Vec3Simd4 => (4, 3, 32),
                    MirType::QuatSimd4 => (4, 4, 32),
                    MirType::Mat4Simd16 => (16, 16, 32),
                    _ => (1, 1, 8),
                };
                let bytes_per_entity = lanes * 8;
                let padding = (lanes - active_lanes) * 8;

                fields.push(SoAFieldLayout {
                    name: f.name.clone(),
                    mir_type,
                    lanes,
                    active_lanes,
                    descriptor_offset_bytes: desc_off,
                    bytes_per_entity,
                    alignment_bytes: align,
                    padding_bytes_per_entity: padding,
                });

                desc_off += 8;
                total_bytes += bytes_per_entity;
                total_pad += padding;
            }

            SoAStateLayout {
                state_name: st.name.clone(),
                fields,
                total_bytes_per_entity: total_bytes,
                total_padding_bytes_per_entity: total_pad,
            }
        })
        .collect();

    let const_map: HashMap<String, f64> = hir
        .consts
        .iter()
        .filter_map(|c| c.scalar_value.map(|v| (c.name.clone(), v)))
        .collect();

    let systems = hir
        .systems
        .iter()
        .map(|sys| {
            let mut reg_counter = 0usize;
            let mut locals = HashMap::new();
            let mut insts = Vec::new();
            lower_stmts(&sys.stmts, &const_map, &mut locals, &mut reg_counter, &mut insts);

            MirSystem {
                name: sys.name.clone(),
                dt_param: sys.dt_param.clone(),
                state_name: sys.state_name.clone(),
                insts,
            }
        })
        .collect();

    MirProgram { layouts, systems }
}

fn next_reg(counter: &mut usize) -> String {
    let r = format!("%r{}", *counter);
    *counter += 1;
    r
}

fn lower_expr(
    e: &HirExpr,
    consts: &HashMap<String, f64>,
    locals: &HashMap<String, String>,
    counter: &mut usize,
    out: &mut Vec<MirInst>,
) -> String {
    match e {
        HirExpr::Number { value, .. } => {
            let dst = next_reg(counter);
            out.push(MirInst::ConstF64 { dst: dst.clone(), value: *value });
            dst
        }
        HirExpr::ConstRef { name, .. } => {
            let dst = next_reg(counter);
            let value = consts.get(name).copied().unwrap_or(0.0);
            out.push(MirInst::ConstF64 { dst: dst.clone(), value });
            dst
        }
        HirExpr::LocalRef { name, .. } => locals.get(name).cloned().unwrap_or_else(|| "%undef".into()),
        HirExpr::ParamRef { name, .. } => {
            let dst = next_reg(counter);
            out.push(MirInst::LoadParam { dst: dst.clone(), name: name.clone() });
            dst
        }
        HirExpr::StateFieldRead { is_next, state_name, field, ty, .. } => {
            let dst = next_reg(counter);
            out.push(MirInst::LoadSoA {
                dst: dst.clone(),
                state: state_name.clone(),
                field: field.clone(),
                is_next: *is_next,
                mir_type: infer_mir_type(ty),
            });
            dst
        }
        HirExpr::VecComponent { vec, component, .. } => {
            let v = lower_expr(vec, consts, locals, counter, out);
            let dst = next_reg(counter);
            let lane = match component {
                'x' => 0,
                'y' => 1,
                _ => 2,
            };
            out.push(MirInst::VecExtract { dst: dst.clone(), vec: v, lane });
            dst
        }
        HirExpr::Vec3 { x, y, z, .. } => {
            let rx = lower_expr(x, consts, locals, counter, out);
            let ry = lower_expr(y, consts, locals, counter, out);
            let rz = lower_expr(z, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::ConstVec3 { dst: dst.clone(), x: rx, y: ry, z: rz });
            dst
        }
        HirExpr::UnaryNeg { operand, ty, .. } => {
            let src = lower_expr(operand, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::UnaryNeg { dst: dst.clone(), src, mir_type: infer_mir_type(ty) });
            dst
        }
        HirExpr::Binary { op, left, right, ty, .. } => {
            let lhs = lower_expr(left, consts, locals, counter, out);
            let rhs = lower_expr(right, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::BinOp {
                dst: dst.clone(),
                op: *op,
                lhs,
                rhs,
                mir_type: infer_mir_type(ty),
            });
            dst
        }
        HirExpr::Call { callee, args, ty, .. } => {
            let lowered_args = args
                .iter()
                .map(|a| lower_expr(a, consts, locals, counter, out))
                .collect();
            let dst = next_reg(counter);
            out.push(MirInst::Call {
                dst: dst.clone(),
                intrinsic: callee.clone(),
                args: lowered_args,
                mir_type: infer_mir_type(ty),
            });
            dst
        }
    }
}

fn lower_stmts(
    stmts: &[HirStmt],
    consts: &HashMap<String, f64>,
    locals: &mut HashMap<String, String>,
    counter: &mut usize,
    out: &mut Vec<MirInst>,
) {
    for s in stmts {
        match s {
            HirStmt::Let { name, value, .. } => {
                let r = lower_expr(value, consts, locals, counter, out);
                locals.insert(name.clone(), r);
            }
            HirStmt::AssignField { state_name, field, value, .. } => {
                let r = lower_expr(value, consts, locals, counter, out);
                out.push(MirInst::StoreSoA {
                    state: state_name.clone(),
                    field: field.clone(),
                    src: r,
                    mir_type: infer_mir_type(value.ty()),
                });
            }
            HirStmt::If { cond, then_stmts, else_stmts, .. } => {
                let c = lower_expr(cond, consts, locals, counter, out);
                let mut then_insts = Vec::new();
                lower_stmts(then_stmts, consts, &mut locals.clone(), counter, &mut then_insts);
                let else_insts = else_stmts.as_ref().map(|el| {
                    let mut v = Vec::new();
                    lower_stmts(el, consts, &mut locals.clone(), counter, &mut v);
                    v
                });
                out.push(MirInst::Branch { cond: c, then_insts, else_insts });
            }
            HirStmt::Expr { expr, .. } => {
                let _ = lower_expr(expr, consts, locals, counter, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_parser::parse;
    use aerotick_sema::analyze;

    #[test]
    fn computes_exact_soa_byte_layout_for_reference_body() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, _) = lex(src);
        let (ast, _) = parse(&tokens, src.len());
        let (hir, errs) = analyze(&ast.expect("ast"));
        assert!(errs.is_empty());
        let mir = lower_to_mir(&hir);
        assert_eq!(mir.layouts[0].total_bytes_per_entity, 104);
        assert_eq!(mir.layouts[0].total_padding_bytes_per_entity, 24);
        assert!(mir.format_ir().contains("soa.load.vec3_simd4 @prev.Body.thrust"));
    }
}
