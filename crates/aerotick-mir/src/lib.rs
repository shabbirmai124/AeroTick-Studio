#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 4: Lowering Typed HIR into Unit-Erased, Cache-Aligned SoA Mid-Level IR (MIR)
//! with AeroTick Ultra v2.0 Optimization Pipeline:
//! 1. Pure Function Inlining
//! 2. Bounded Loop Unrolling (`for i in 0..N`)
//! 3. Dimension-Safe IEEE-754 Constant Folding (strictly non-associative via `libm`)
//! 4. Common Subexpression Elimination (CSE) & Dead Code Elimination (DCE)

use aerotick_ast::BinOp;
use aerotick_sema::{HirExpr, HirFn, HirProgram, HirStmt, ValueType};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirType {
    F64,
    Vec3Simd4,
    QuatSimd4,
    Mat3Simd12,
    Mat4Simd16,
    Bool,
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F64 => write!(f, "f64"),
            Self::Vec3Simd4 => write!(f, "vec3_simd4"),
            Self::QuatSimd4 => write!(f, "quat_simd4"),
            Self::Mat3Simd12 => write!(f, "mat3_simd12"),
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

impl SoAStateLayout {
    pub fn cache_lines_64b(&self) -> usize {
        if self.total_bytes_per_entity == 0 {
            0
        } else {
            self.total_bytes_per_entity.div_ceil(64)
        }
    }

    pub fn cache_line_utilization_pct(&self) -> f64 {
        let lines = self.cache_lines_64b();
        if lines == 0 {
            return 100.0;
        }
        let active_bytes: usize = self.fields.iter().map(|f| f.active_lanes * 8).sum();
        (active_bytes as f64 / (lines * 64) as f64) * 100.0
    }
}

#[derive(Debug, Clone)]
pub enum MirInst {
    ConstF64 {
        dst: String,
        value: f64,
    },
    ConstVec3 {
        dst: String,
        x: String,
        y: String,
        z: String,
    },
    ConstQuat {
        dst: String,
        w: String,
        x: String,
        y: String,
        z: String,
    },
    ConstMat3 {
        dst: String,
        elems: [String; 9],
    },
    LoadParam {
        dst: String,
        name: String,
    },
    LoadSoA {
        dst: String,
        state: String,
        field: String,
        is_next: bool,
        mir_type: MirType,
    },
    StoreSoA {
        state: String,
        field: String,
        src: String,
        mir_type: MirType,
    },
    VecExtract {
        dst: String,
        vec: String,
        lane: usize,
    },
    Select {
        dst: String,
        cond: String,
        then_val: String,
        else_val: String,
        mir_type: MirType,
    },
    UnaryNeg {
        dst: String,
        src: String,
        mir_type: MirType,
    },
    BinOp {
        dst: String,
        op: BinOp,
        lhs: String,
        rhs: String,
        mir_type: MirType,
    },
    Call {
        dst: String,
        intrinsic: String,
        args: Vec<String>,
        mir_type: MirType,
    },
    ForLoop {
        var: String,
        start: i64,
        end: i64,
        body: Vec<MirInst>,
    },
    Branch {
        cond: String,
        then_insts: Vec<MirInst>,
        else_insts: Option<Vec<MirInst>>,
    },
}

#[derive(Debug, Clone)]
pub struct MirFn {
    pub name: String,
    pub params: Vec<(String, MirType)>,
    pub return_type: MirType,
    pub insts: Vec<MirInst>,
    pub ret_reg: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MirSystem {
    pub name: String,
    pub dt_param: String,
    pub state_name: String,
    pub insts: Vec<MirInst>,
}

#[derive(Debug, Clone)]
pub struct MirTest {
    pub name: String,
    pub insts: Vec<MirInst>,
}

#[derive(Debug, Clone, Default)]
pub struct MirOptStats {
    pub before_inst_count: usize,
    pub after_inst_count: usize,
    pub folded_constants: usize,
    pub inlined_calls: usize,
    pub unrolled_loops: usize,
    pub cse_eliminated: usize,
    pub dce_eliminated: usize,
}

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub layouts: Vec<SoAStateLayout>,
    pub functions: Vec<MirFn>,
    pub systems: Vec<MirSystem>,
    pub tests: Vec<MirTest>,
    pub opt_stats: MirOptStats,
}

impl MirProgram {
    pub fn format_ir(&self) -> String {
        let mut lines = vec![
            "; AeroTick Ultra v2.0 Unit-Erased SoA MIR".to_string(),
            String::new(),
        ];
        for layout in &self.layouts {
            lines.push(format!(
                "soa_layout {} ({} bytes/entity, {} padding bytes, {}x64B cache lines, {:.1}% util) {{",
                layout.state_name,
                layout.total_bytes_per_entity,
                layout.total_padding_bytes_per_entity,
                layout.cache_lines_64b(),
                layout.cache_line_utilization_pct(),
            ));
            for f in &layout.fields {
                lines.push(format!(
                    "  {:<12} : {:<12} | desc_off={:>2}B | align={:>2}B | stride={:>2}B | pad={}B",
                    f.name,
                    f.mir_type,
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

    pub fn format_stats(&self) -> String {
        let mut out = vec![
            "AeroTick MIR Optimization & SoA Cache Report".to_string(),
            format!(
                "  instructions_before : {}",
                self.opt_stats.before_inst_count
            ),
            format!(
                "  instructions_after  : {}",
                self.opt_stats.after_inst_count
            ),
            format!(
                "  constants_folded    : {}",
                self.opt_stats.folded_constants
            ),
            format!("  pure_fns_inlined    : {}", self.opt_stats.inlined_calls),
            format!("  loops_unrolled      : {}", self.opt_stats.unrolled_loops),
            format!("  cse_eliminated      : {}", self.opt_stats.cse_eliminated),
            format!("  dce_eliminated      : {}", self.opt_stats.dce_eliminated),
        ];
        for l in &self.layouts {
            out.push(format!(
                "  soa[{}] => {} B/entity | {} B pad | {} x 64B cache lines | {:.1}% active lane utilization",
                l.state_name,
                l.total_bytes_per_entity,
                l.total_padding_bytes_per_entity,
                l.cache_lines_64b(),
                l.cache_line_utilization_pct()
            ));
        }
        out.join("\n")
    }
}

fn dump_insts(insts: &[MirInst], indent: &str, out: &mut Vec<String>) {
    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => {
                out.push(format!("{indent}{dst} = const.f64 {value}"))
            }
            MirInst::ConstVec3 { dst, x, y, z } => {
                out.push(format!("{indent}{dst} = simd4.pack <{x}, {y}, {z}, 0.0>"))
            }
            MirInst::ConstQuat { dst, w, x, y, z } => {
                out.push(format!("{indent}{dst} = quat.pack <{w}, {x}, {y}, {z}>"))
            }
            MirInst::ConstMat3 { dst, elems } => {
                out.push(format!("{indent}{dst} = mat3.pack [{}]", elems.join(", ")))
            }
            MirInst::LoadParam { dst, name } => {
                out.push(format!("{indent}{dst} = param.f64 %{name}"))
            }
            MirInst::LoadSoA {
                dst,
                state,
                field,
                is_next,
                mir_type,
            } => {
                let buf = if *is_next { "next" } else { "prev" };
                out.push(format!(
                    "{indent}{dst} = soa.load.{mir_type} @{buf}.{state}.{field}[%i]"
                ));
            }
            MirInst::StoreSoA {
                state,
                field,
                src,
                mir_type,
            } => {
                out.push(format!(
                    "{indent}soa.store.{mir_type} @next.{state}.{field}[%i], {src}"
                ));
            }
            MirInst::VecExtract { dst, vec, lane } => {
                out.push(format!("{indent}{dst} = simd4.extract {vec}, lane {lane}"))
            }
            MirInst::Select {
                dst,
                cond,
                then_val,
                else_val,
                mir_type,
            } => {
                out.push(format!(
                    "{indent}{dst} = select.{mir_type} {cond}, {then_val}, {else_val}"
                ));
            }
            MirInst::UnaryNeg { dst, src, mir_type } => {
                out.push(format!("{indent}{dst} = neg.{mir_type} {src}"))
            }
            MirInst::BinOp {
                dst,
                op,
                lhs,
                rhs,
                mir_type,
            } => {
                out.push(format!("{indent}{dst} = {op:?}.{mir_type} {lhs}, {rhs}"));
            }
            MirInst::Call {
                dst,
                intrinsic,
                args,
                mir_type,
            } => {
                out.push(format!(
                    "{indent}{dst} = call @det_{intrinsic}({}) : {mir_type}",
                    args.join(", ")
                ));
            }
            MirInst::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                out.push(format!("{indent}for {var} in {start}..{end} {{"));
                dump_insts(body, &format!("{indent}  "), out);
                out.push(format!("{indent}}}"));
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
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
        ValueType::Mat3(_) => MirType::Mat3Simd12,
        ValueType::Mat4 => MirType::Mat4Simd16,
        ValueType::Bool => MirType::Bool,
        _ => MirType::F64,
    }
}

pub fn lower_to_mir(hir: &HirProgram) -> MirProgram {
    lower_to_mir_with_opt(hir, true)
}

pub fn lower_to_mir_with_opt(hir: &HirProgram, optimize: bool) -> MirProgram {
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
                    MirType::Mat3Simd12 => (12, 9, 32),
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

    let functions: Vec<MirFn> = hir
        .functions
        .iter()
        .map(|f| lower_fn(f, &const_map))
        .collect();

    let mut systems: Vec<MirSystem> = hir
        .systems
        .iter()
        .map(|sys| {
            let mut reg_counter = 0usize;
            let mut locals = HashMap::new();
            let mut insts = Vec::new();
            lower_stmts(
                &sys.stmts,
                &const_map,
                &mut locals,
                &mut reg_counter,
                &mut insts,
                &mut None,
            );

            MirSystem {
                name: sys.name.clone(),
                dt_param: sys.dt_param.clone(),
                state_name: sys.state_name.clone(),
                insts,
            }
        })
        .collect();

    let mut tests: Vec<MirTest> = hir
        .tests
        .iter()
        .map(|t| {
            let mut reg_counter = 0usize;
            let mut locals = HashMap::new();
            let mut insts = Vec::new();
            lower_stmts(
                &t.body,
                &const_map,
                &mut locals,
                &mut reg_counter,
                &mut insts,
                &mut None,
            );
            MirTest {
                name: t.name.clone(),
                insts,
            }
        })
        .collect();

    let before_inst_count: usize = systems.iter().map(|s| count_insts(&s.insts)).sum::<usize>()
        + tests.iter().map(|t| count_insts(&t.insts)).sum::<usize>();

    let mut stats = MirOptStats {
        before_inst_count,
        ..MirOptStats::default()
    };

    if optimize {
        let fn_map: HashMap<String, MirFn> = functions
            .iter()
            .map(|f| (f.name.clone(), f.clone()))
            .collect();

        for sys in &mut systems {
            sys.insts = optimize_insts(&sys.insts, &fn_map, &mut stats);
        }
        for t in &mut tests {
            t.insts = optimize_insts(&t.insts, &fn_map, &mut stats);
        }
    }

    stats.after_inst_count = systems.iter().map(|s| count_insts(&s.insts)).sum::<usize>()
        + tests.iter().map(|t| count_insts(&t.insts)).sum::<usize>();

    MirProgram {
        layouts,
        functions,
        systems,
        tests,
        opt_stats: stats,
    }
}

fn lower_fn(f: &HirFn, consts: &HashMap<String, f64>) -> MirFn {
    let mut counter = 0usize;
    let mut locals = HashMap::new();
    let mut params = Vec::new();
    for p in &f.params {
        let mt = infer_mir_type(&p.ty);
        let reg = format!("%arg_{}", p.name);
        locals.insert(p.name.clone(), reg);
        params.push((p.name.clone(), mt));
    }
    let mut insts = Vec::new();
    let mut ret_reg = None;
    lower_stmts(
        &f.body,
        consts,
        &mut locals,
        &mut counter,
        &mut insts,
        &mut ret_reg,
    );
    MirFn {
        name: f.name.clone(),
        params,
        return_type: infer_mir_type(&f.return_type),
        insts,
        ret_reg,
    }
}

fn count_insts(insts: &[MirInst]) -> usize {
    let mut total = 0;
    for inst in insts {
        total += 1;
        match inst {
            MirInst::ForLoop { body, .. } => total += count_insts(body),
            MirInst::Branch {
                then_insts,
                else_insts,
                ..
            } => {
                total += count_insts(then_insts);
                if let Some(el) = else_insts {
                    total += count_insts(el);
                }
            }
            _ => {}
        }
    }
    total
}

pub fn optimize_insts(
    insts: &[MirInst],
    fn_map: &HashMap<String, MirFn>,
    stats: &mut MirOptStats,
) -> Vec<MirInst> {
    let mut inline_counter = 0usize;
    let inlined = pass_inline_fns(insts, fn_map, &mut inline_counter, stats);
    let unrolled = pass_unroll_loops(&inlined, &mut inline_counter, stats);
    let folded = pass_const_fold(&unrolled, stats);
    let csed = pass_cse(&folded, stats);
    pass_dce(&csed, stats)
}

fn pass_inline_fns(
    insts: &[MirInst],
    fn_map: &HashMap<String, MirFn>,
    uid: &mut usize,
    stats: &mut MirOptStats,
) -> Vec<MirInst> {
    let mut out = Vec::with_capacity(insts.len());
    let mut out_aliases: HashMap<String, String> = HashMap::new();
    let resolve = |r: &str, alias: &HashMap<String, String>| -> String {
        let mut cur = r.to_string();
        while let Some(next) = alias.get(&cur) {
            if next == &cur {
                break;
            }
            cur = next.clone();
        }
        cur
    };

    for raw_inst in insts {
        let inst = remap_uses(raw_inst, &resolve, &out_aliases);
        match &inst {
            MirInst::Call {
                dst,
                intrinsic,
                args,
                ..
            } if fn_map.contains_key(intrinsic.as_str()) => {
                if let Some(func) = fn_map.get(intrinsic.as_str()) {
                    stats.inlined_calls += 1;
                    let prefix = format!("_inl{}_", *uid);
                    *uid += 1;
                    let mut rename: HashMap<String, String> = HashMap::new();
                    for ((pname, _), arg_reg) in func.params.iter().zip(args.iter()) {
                        rename.insert(format!("%arg_{pname}"), arg_reg.clone());
                    }
                    let inlined_body = pass_inline_fns(&func.insts, fn_map, uid, stats);
                    for bi in &inlined_body {
                        out.push(rename_inst(bi, &prefix, &mut rename));
                    }
                    if let Some(ret) = &func.ret_reg {
                        let mapped_ret = rename
                            .get(ret)
                            .cloned()
                            .unwrap_or_else(|| format!("%{prefix}{}", ret.trim_start_matches('%')));
                        if let Some(last_dst) = out.last().and_then(inst_dst)
                            && last_dst == mapped_ret
                        {
                            if let Some(last) = out.last_mut() {
                                set_inst_dst(last, dst.clone());
                            }
                        } else {
                            out_aliases.insert(dst.clone(), mapped_ret);
                        }
                    }
                    continue;
                }
            }
            MirInst::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                out.push(MirInst::ForLoop {
                    var: var.clone(),
                    start: *start,
                    end: *end,
                    body: pass_inline_fns(body, fn_map, uid, stats),
                });
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
                out.push(MirInst::Branch {
                    cond: cond.clone(),
                    then_insts: pass_inline_fns(then_insts, fn_map, uid, stats),
                    else_insts: else_insts
                        .as_ref()
                        .map(|e| pass_inline_fns(e, fn_map, uid, stats)),
                });
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn pass_unroll_loops(insts: &[MirInst], uid: &mut usize, stats: &mut MirOptStats) -> Vec<MirInst> {
    let mut out = Vec::with_capacity(insts.len());
    for inst in insts {
        match inst {
            MirInst::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                stats.unrolled_loops += 1;
                let expanded_body = pass_unroll_loops(body, uid, stats);
                for iter_val in *start..*end {
                    let prefix = format!("_unr{}_", *uid);
                    *uid += 1;
                    let mut rename = HashMap::new();
                    let idx_reg = format!("%{prefix}{var}");
                    out.push(MirInst::ConstF64 {
                        dst: idx_reg.clone(),
                        value: iter_val as f64,
                    });
                    rename.insert(format!("%loop_{var}"), idx_reg);
                    for bi in &expanded_body {
                        out.push(rename_inst(bi, &prefix, &mut rename));
                    }
                }
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
                out.push(MirInst::Branch {
                    cond: cond.clone(),
                    then_insts: pass_unroll_loops(then_insts, uid, stats),
                    else_insts: else_insts
                        .as_ref()
                        .map(|e| pass_unroll_loops(e, uid, stats)),
                });
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn pass_const_fold(insts: &[MirInst], stats: &mut MirOptStats) -> Vec<MirInst> {
    let mut consts: HashMap<String, f64> = HashMap::new();
    let mut out = Vec::with_capacity(insts.len());

    for inst in insts {
        match inst {
            MirInst::ConstF64 { dst, value } => {
                consts.insert(dst.clone(), *value);
                out.push(inst.clone());
            }
            MirInst::UnaryNeg {
                dst,
                src,
                mir_type: MirType::F64,
            } => {
                if let Some(&v) = consts.get(src) {
                    let folded = -v;
                    consts.insert(dst.clone(), folded);
                    stats.folded_constants += 1;
                    out.push(MirInst::ConstF64 {
                        dst: dst.clone(),
                        value: folded,
                    });
                } else {
                    out.push(inst.clone());
                }
            }
            MirInst::BinOp {
                dst,
                op,
                lhs,
                rhs,
                mir_type,
            } if *mir_type == MirType::F64 || *mir_type == MirType::Bool => {
                if let (Some(&l), Some(&r)) = (consts.get(lhs), consts.get(rhs)) {
                    let res = match op {
                        BinOp::Add => l + r,
                        BinOp::Sub => l - r,
                        BinOp::Mul => l * r,
                        BinOp::Div => {
                            if r == 0.0 {
                                0.0
                            } else {
                                l / r
                            }
                        }
                        BinOp::Pow => libm::pow(l, r),
                        BinOp::Eq => {
                            if l == r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        BinOp::Ne => {
                            if l != r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        BinOp::Lt => {
                            if l < r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        BinOp::Le => {
                            if l <= r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        BinOp::Gt => {
                            if l > r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        BinOp::Ge => {
                            if l >= r {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    consts.insert(dst.clone(), res);
                    stats.folded_constants += 1;
                    out.push(MirInst::ConstF64 {
                        dst: dst.clone(),
                        value: res,
                    });
                } else {
                    out.push(inst.clone());
                }
            }
            MirInst::Call {
                dst,
                intrinsic,
                args,
                mir_type: MirType::F64,
            } => {
                let cargs: Option<Vec<f64>> = args.iter().map(|a| consts.get(a).copied()).collect();
                let folded = cargs.and_then(|vals| match (intrinsic.as_str(), vals.as_slice()) {
                    ("sqrt", [x]) => Some(libm::sqrt((*x).max(0.0))),
                    ("cbrt", [x]) => Some(libm::cbrt(*x)),
                    ("sin", [x]) => Some(libm::sin(*x)),
                    ("cos", [x]) => Some(libm::cos(*x)),
                    ("tan", [x]) => Some(libm::tan(*x)),
                    ("atan2", [y, x]) => Some(libm::atan2(*y, *x)),
                    ("abs", [x]) => Some(x.abs()),
                    ("min", [a, b]) => Some(a.min(*b)),
                    ("max", [a, b]) => Some(a.max(*b)),
                    ("clamp", [x, lo, hi]) => Some(x.max(*lo).min(*hi)),
                    _ => None,
                });
                if let Some(val) = folded {
                    consts.insert(dst.clone(), val);
                    stats.folded_constants += 1;
                    out.push(MirInst::ConstF64 {
                        dst: dst.clone(),
                        value: val,
                    });
                } else {
                    out.push(inst.clone());
                }
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => out.push(MirInst::Branch {
                cond: cond.clone(),
                then_insts: pass_const_fold(then_insts, stats),
                else_insts: else_insts.as_ref().map(|e| pass_const_fold(e, stats)),
            }),
            other => out.push(other.clone()),
        }
    }
    out
}

fn pass_cse(insts: &[MirInst], stats: &mut MirOptStats) -> Vec<MirInst> {
    let mut expr_table: HashMap<String, String> = HashMap::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut out = Vec::with_capacity(insts.len());

    let resolve = |r: &str, alias: &HashMap<String, String>| -> String {
        alias.get(r).cloned().unwrap_or_else(|| r.to_string())
    };

    for inst in insts {
        let rewritten = remap_uses(inst, &resolve, &alias);
        let key_opt = match &rewritten {
            MirInst::ConstF64 { value, .. } => Some(format!("const:{:?}", value.to_bits())),
            MirInst::LoadParam { name, .. } => Some(format!("param:{name}")),
            MirInst::LoadSoA {
                state,
                field,
                is_next: false,
                ..
            } => Some(format!("load_prev:{state}.{field}")),
            MirInst::VecExtract { vec, lane, .. } => Some(format!("extract:{vec}:{lane}")),
            MirInst::UnaryNeg { src, mir_type, .. } => Some(format!("neg:{mir_type}:{src}")),
            MirInst::BinOp {
                op,
                lhs,
                rhs,
                mir_type,
                ..
            } => Some(format!("bin:{op:?}:{mir_type}:{lhs}:{rhs}")),
            _ => None,
        };
        if let (Some(key), Some(dst)) = (key_opt, inst_dst(&rewritten)) {
            if let Some(existing) = expr_table.get(&key) {
                alias.insert(dst, existing.clone());
                stats.cse_eliminated += 1;
                continue;
            }
            expr_table.insert(key, dst);
        }
        out.push(rewritten);
    }
    out
}

fn pass_dce(insts: &[MirInst], stats: &mut MirOptStats) -> Vec<MirInst> {
    let mut used: HashSet<String> = HashSet::new();
    collect_side_effect_roots(insts, &mut used);

    let mut changed = true;
    while changed {
        changed = false;
        for inst in insts {
            let is_side_effecting = matches!(
                inst,
                MirInst::StoreSoA { .. } | MirInst::Branch { .. } | MirInst::ForLoop { .. }
            ) || matches!(inst, MirInst::Call { intrinsic, .. } if intrinsic == "assert_approx");
            let dst_used = inst_dst(inst).is_some_and(|d| used.contains(&d));
            if is_side_effecting || dst_used {
                for u in inst_uses(inst) {
                    if used.insert(u) {
                        changed = true;
                    }
                }
            }
        }
    }

    let mut out = Vec::with_capacity(insts.len());
    for inst in insts {
        let keep = match inst {
            MirInst::StoreSoA { .. } | MirInst::Branch { .. } | MirInst::ForLoop { .. } => true,
            MirInst::Call { intrinsic, dst, .. } => {
                intrinsic == "assert_approx" || used.contains(dst)
            }
            other => inst_dst(other).is_none_or(|d| used.contains(&d)),
        };
        if keep {
            out.push(match inst {
                MirInst::Branch {
                    cond,
                    then_insts,
                    else_insts,
                } => MirInst::Branch {
                    cond: cond.clone(),
                    then_insts: pass_dce(then_insts, stats),
                    else_insts: else_insts.as_ref().map(|e| pass_dce(e, stats)),
                },
                other => other.clone(),
            });
        } else {
            stats.dce_eliminated += 1;
        }
    }
    out
}

fn collect_side_effect_roots(insts: &[MirInst], used: &mut HashSet<String>) {
    for inst in insts {
        match inst {
            MirInst::StoreSoA { src, .. } => {
                used.insert(src.clone());
            }
            MirInst::Call {
                intrinsic, args, ..
            } if intrinsic == "assert_approx" => {
                for a in args {
                    used.insert(a.clone());
                }
            }
            MirInst::Branch {
                cond,
                then_insts,
                else_insts,
            } => {
                used.insert(cond.clone());
                collect_side_effect_roots(then_insts, used);
                if let Some(el) = else_insts {
                    collect_side_effect_roots(el, used);
                }
            }
            MirInst::ForLoop { body, .. } => collect_side_effect_roots(body, used),
            _ => {}
        }
    }
}

fn inst_dst(inst: &MirInst) -> Option<String> {
    match inst {
        MirInst::ConstF64 { dst, .. }
        | MirInst::ConstVec3 { dst, .. }
        | MirInst::ConstQuat { dst, .. }
        | MirInst::ConstMat3 { dst, .. }
        | MirInst::LoadParam { dst, .. }
        | MirInst::LoadSoA { dst, .. }
        | MirInst::VecExtract { dst, .. }
        | MirInst::Select { dst, .. }
        | MirInst::UnaryNeg { dst, .. }
        | MirInst::BinOp { dst, .. }
        | MirInst::Call { dst, .. } => Some(dst.clone()),
        MirInst::StoreSoA { .. } | MirInst::ForLoop { .. } | MirInst::Branch { .. } => None,
    }
}

fn set_inst_dst(inst: &mut MirInst, new_dst: String) {
    match inst {
        MirInst::ConstF64 { dst, .. }
        | MirInst::ConstVec3 { dst, .. }
        | MirInst::ConstQuat { dst, .. }
        | MirInst::ConstMat3 { dst, .. }
        | MirInst::LoadParam { dst, .. }
        | MirInst::LoadSoA { dst, .. }
        | MirInst::VecExtract { dst, .. }
        | MirInst::Select { dst, .. }
        | MirInst::UnaryNeg { dst, .. }
        | MirInst::BinOp { dst, .. }
        | MirInst::Call { dst, .. } => *dst = new_dst,
        MirInst::StoreSoA { .. } | MirInst::ForLoop { .. } | MirInst::Branch { .. } => {}
    }
}

fn inst_uses(inst: &MirInst) -> Vec<String> {
    match inst {
        MirInst::ConstF64 { .. } | MirInst::LoadParam { .. } | MirInst::LoadSoA { .. } => vec![],
        MirInst::ConstVec3 { x, y, z, .. } => vec![x.clone(), y.clone(), z.clone()],
        MirInst::ConstQuat { w, x, y, z, .. } => vec![w.clone(), x.clone(), y.clone(), z.clone()],
        MirInst::ConstMat3 { elems, .. } => elems.to_vec(),
        MirInst::StoreSoA { src, .. } => vec![src.clone()],
        MirInst::VecExtract { vec, .. } => vec![vec.clone()],
        MirInst::Select {
            cond,
            then_val,
            else_val,
            ..
        } => vec![cond.clone(), then_val.clone(), else_val.clone()],
        MirInst::UnaryNeg { src, .. } => vec![src.clone()],
        MirInst::BinOp { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        MirInst::Call { args, .. } => args.clone(),
        MirInst::ForLoop { .. } | MirInst::Branch { .. } => vec![],
    }
}

fn rename_inst(inst: &MirInst, prefix: &str, rename: &mut HashMap<String, String>) -> MirInst {
    let map_use = |r: &str, map: &HashMap<String, String>| -> String {
        map.get(r)
            .cloned()
            .unwrap_or_else(|| format!("%{prefix}{}", r.trim_start_matches('%')))
    };
    let map_dst = |d: &str, map: &mut HashMap<String, String>| -> String {
        let nd = format!("%{prefix}{}", d.trim_start_matches('%'));
        map.insert(d.to_string(), nd.clone());
        nd
    };
    match inst {
        MirInst::ConstF64 { dst, value } => MirInst::ConstF64 {
            dst: map_dst(dst, rename),
            value: *value,
        },
        MirInst::ConstVec3 { dst, x, y, z } => {
            let (nx, ny, nz) = (map_use(x, rename), map_use(y, rename), map_use(z, rename));
            MirInst::ConstVec3 {
                dst: map_dst(dst, rename),
                x: nx,
                y: ny,
                z: nz,
            }
        }
        MirInst::ConstQuat { dst, w, x, y, z } => {
            let (nw, nx, ny, nz) = (
                map_use(w, rename),
                map_use(x, rename),
                map_use(y, rename),
                map_use(z, rename),
            );
            MirInst::ConstQuat {
                dst: map_dst(dst, rename),
                w: nw,
                x: nx,
                y: ny,
                z: nz,
            }
        }
        MirInst::ConstMat3 { dst, elems } => {
            let mapped = elems.each_ref().map(|e| map_use(e, rename));
            MirInst::ConstMat3 {
                dst: map_dst(dst, rename),
                elems: mapped,
            }
        }
        MirInst::LoadParam { dst, name } => MirInst::LoadParam {
            dst: map_dst(dst, rename),
            name: name.clone(),
        },
        MirInst::LoadSoA {
            dst,
            state,
            field,
            is_next,
            mir_type,
        } => MirInst::LoadSoA {
            dst: map_dst(dst, rename),
            state: state.clone(),
            field: field.clone(),
            is_next: *is_next,
            mir_type: *mir_type,
        },
        MirInst::StoreSoA {
            state,
            field,
            src,
            mir_type,
        } => MirInst::StoreSoA {
            state: state.clone(),
            field: field.clone(),
            src: map_use(src, rename),
            mir_type: *mir_type,
        },
        MirInst::VecExtract { dst, vec, lane } => {
            let nv = map_use(vec, rename);
            MirInst::VecExtract {
                dst: map_dst(dst, rename),
                vec: nv,
                lane: *lane,
            }
        }
        MirInst::Select {
            dst,
            cond,
            then_val,
            else_val,
            mir_type,
        } => {
            let (nc, nt, ne) = (
                map_use(cond, rename),
                map_use(then_val, rename),
                map_use(else_val, rename),
            );
            MirInst::Select {
                dst: map_dst(dst, rename),
                cond: nc,
                then_val: nt,
                else_val: ne,
                mir_type: *mir_type,
            }
        }
        MirInst::UnaryNeg { dst, src, mir_type } => {
            let ns = map_use(src, rename);
            MirInst::UnaryNeg {
                dst: map_dst(dst, rename),
                src: ns,
                mir_type: *mir_type,
            }
        }
        MirInst::BinOp {
            dst,
            op,
            lhs,
            rhs,
            mir_type,
        } => {
            let (nl, nr) = (map_use(lhs, rename), map_use(rhs, rename));
            MirInst::BinOp {
                dst: map_dst(dst, rename),
                op: *op,
                lhs: nl,
                rhs: nr,
                mir_type: *mir_type,
            }
        }
        MirInst::Call {
            dst,
            intrinsic,
            args,
            mir_type,
        } => {
            let mapped_args = args.iter().map(|a| map_use(a, rename)).collect();
            MirInst::Call {
                dst: map_dst(dst, rename),
                intrinsic: intrinsic.clone(),
                args: mapped_args,
                mir_type: *mir_type,
            }
        }
        MirInst::ForLoop {
            var,
            start,
            end,
            body,
        } => MirInst::ForLoop {
            var: var.clone(),
            start: *start,
            end: *end,
            body: body
                .iter()
                .map(|b| rename_inst(b, prefix, rename))
                .collect(),
        },
        MirInst::Branch {
            cond,
            then_insts,
            else_insts,
        } => MirInst::Branch {
            cond: map_use(cond, rename),
            then_insts: then_insts
                .iter()
                .map(|b| rename_inst(b, prefix, rename))
                .collect(),
            else_insts: else_insts
                .as_ref()
                .map(|el| el.iter().map(|b| rename_inst(b, prefix, rename)).collect()),
        },
    }
}

fn remap_uses<F>(inst: &MirInst, resolve: &F, alias: &HashMap<String, String>) -> MirInst
where
    F: Fn(&str, &HashMap<String, String>) -> String,
{
    match inst {
        MirInst::ConstVec3 { dst, x, y, z } => MirInst::ConstVec3 {
            dst: dst.clone(),
            x: resolve(x, alias),
            y: resolve(y, alias),
            z: resolve(z, alias),
        },
        MirInst::ConstQuat { dst, w, x, y, z } => MirInst::ConstQuat {
            dst: dst.clone(),
            w: resolve(w, alias),
            x: resolve(x, alias),
            y: resolve(y, alias),
            z: resolve(z, alias),
        },
        MirInst::ConstMat3 { dst, elems } => MirInst::ConstMat3 {
            dst: dst.clone(),
            elems: elems.each_ref().map(|e| resolve(e, alias)),
        },
        MirInst::StoreSoA {
            state,
            field,
            src,
            mir_type,
        } => MirInst::StoreSoA {
            state: state.clone(),
            field: field.clone(),
            src: resolve(src, alias),
            mir_type: *mir_type,
        },
        MirInst::VecExtract { dst, vec, lane } => MirInst::VecExtract {
            dst: dst.clone(),
            vec: resolve(vec, alias),
            lane: *lane,
        },
        MirInst::Select {
            dst,
            cond,
            then_val,
            else_val,
            mir_type,
        } => MirInst::Select {
            dst: dst.clone(),
            cond: resolve(cond, alias),
            then_val: resolve(then_val, alias),
            else_val: resolve(else_val, alias),
            mir_type: *mir_type,
        },
        MirInst::UnaryNeg { dst, src, mir_type } => MirInst::UnaryNeg {
            dst: dst.clone(),
            src: resolve(src, alias),
            mir_type: *mir_type,
        },
        MirInst::BinOp {
            dst,
            op,
            lhs,
            rhs,
            mir_type,
        } => MirInst::BinOp {
            dst: dst.clone(),
            op: *op,
            lhs: resolve(lhs, alias),
            rhs: resolve(rhs, alias),
            mir_type: *mir_type,
        },
        MirInst::Call {
            dst,
            intrinsic,
            args,
            mir_type,
        } => MirInst::Call {
            dst: dst.clone(),
            intrinsic: intrinsic.clone(),
            args: args.iter().map(|a| resolve(a, alias)).collect(),
            mir_type: *mir_type,
        },
        MirInst::Branch {
            cond,
            then_insts,
            else_insts,
        } => MirInst::Branch {
            cond: resolve(cond, alias),
            then_insts: then_insts
                .iter()
                .map(|i| remap_uses(i, resolve, alias))
                .collect(),
            else_insts: else_insts
                .as_ref()
                .map(|el| el.iter().map(|i| remap_uses(i, resolve, alias)).collect()),
        },
        other => other.clone(),
    }
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
            out.push(MirInst::ConstF64 {
                dst: dst.clone(),
                value: *value,
            });
            dst
        }
        HirExpr::ConstRef { name, .. } => {
            let dst = next_reg(counter);
            let value = consts.get(name).copied().unwrap_or(0.0);
            out.push(MirInst::ConstF64 {
                dst: dst.clone(),
                value,
            });
            dst
        }
        HirExpr::LocalRef { name, .. } => {
            locals.get(name).cloned().unwrap_or_else(|| "%undef".into())
        }
        HirExpr::ParamRef { name, .. } => {
            if let Some(existing) = locals.get(name) {
                return existing.clone();
            }
            let dst = next_reg(counter);
            out.push(MirInst::LoadParam {
                dst: dst.clone(),
                name: name.clone(),
            });
            dst
        }
        HirExpr::StateFieldRead {
            is_next,
            state_name,
            field,
            ty,
            ..
        } => {
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
            let actual_lane = if matches!(vec.ty(), ValueType::Quat) {
                match component {
                    'w' => 0,
                    'x' => 1,
                    'y' => 2,
                    _ => 3,
                }
            } else {
                match component {
                    'x' => 0,
                    'y' => 1,
                    _ => 2,
                }
            };
            out.push(MirInst::VecExtract {
                dst: dst.clone(),
                vec: v,
                lane: actual_lane,
            });
            dst
        }
        HirExpr::Vec3 { x, y, z, .. } => {
            let rx = lower_expr(x, consts, locals, counter, out);
            let ry = lower_expr(y, consts, locals, counter, out);
            let rz = lower_expr(z, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::ConstVec3 {
                dst: dst.clone(),
                x: rx,
                y: ry,
                z: rz,
            });
            dst
        }
        HirExpr::Quat { w, x, y, z, .. } => {
            let rw = lower_expr(w, consts, locals, counter, out);
            let rx = lower_expr(x, consts, locals, counter, out);
            let ry = lower_expr(y, consts, locals, counter, out);
            let rz = lower_expr(z, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::ConstQuat {
                dst: dst.clone(),
                w: rw,
                x: rx,
                y: ry,
                z: rz,
            });
            dst
        }
        HirExpr::Mat3 { elems, .. } => {
            let lowered: [String; 9] = elems
                .each_ref()
                .map(|el| lower_expr(el, consts, locals, counter, out));
            let dst = next_reg(counter);
            out.push(MirInst::ConstMat3 {
                dst: dst.clone(),
                elems: lowered,
            });
            dst
        }
        HirExpr::IfExpr {
            cond,
            then_expr,
            else_expr,
            ty,
            ..
        } => {
            let rc = lower_expr(cond, consts, locals, counter, out);
            let rt = lower_expr(then_expr, consts, locals, counter, out);
            let re = lower_expr(else_expr, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::Select {
                dst: dst.clone(),
                cond: rc,
                then_val: rt,
                else_val: re,
                mir_type: infer_mir_type(ty),
            });
            dst
        }
        HirExpr::UnaryNeg { operand, ty, .. } => {
            let src = lower_expr(operand, consts, locals, counter, out);
            let dst = next_reg(counter);
            out.push(MirInst::UnaryNeg {
                dst: dst.clone(),
                src,
                mir_type: infer_mir_type(ty),
            });
            dst
        }
        HirExpr::Binary {
            op,
            left,
            right,
            ty,
            ..
        } => {
            let lhs = lower_expr(left, consts, locals, counter, out);
            let rhs = lower_expr(right, consts, locals, counter, out);
            let dst = next_reg(counter);
            let mir_type = infer_mir_type(ty);
            out.push(MirInst::BinOp {
                dst: dst.clone(),
                op: *op,
                lhs,
                rhs,
                mir_type,
            });
            dst
        }
        HirExpr::Call {
            callee, args, ty, ..
        } => {
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
    ret_reg: &mut Option<String>,
) {
    for s in stmts {
        match s {
            HirStmt::Let { name, value, .. } => {
                let r = lower_expr(value, consts, locals, counter, out);
                locals.insert(name.clone(), r);
            }
            HirStmt::AssignField {
                state_name,
                field,
                value,
                ..
            } => {
                let r = lower_expr(value, consts, locals, counter, out);
                out.push(MirInst::StoreSoA {
                    state: state_name.clone(),
                    field: field.clone(),
                    src: r,
                    mir_type: infer_mir_type(value.ty()),
                });
            }
            HirStmt::If {
                cond,
                then_stmts,
                else_stmts,
                ..
            } => {
                let c = lower_expr(cond, consts, locals, counter, out);
                let mut then_insts = Vec::new();
                lower_stmts(
                    then_stmts,
                    consts,
                    &mut locals.clone(),
                    counter,
                    &mut then_insts,
                    ret_reg,
                );
                let else_insts = else_stmts.as_ref().map(|el| {
                    let mut v = Vec::new();
                    lower_stmts(el, consts, &mut locals.clone(), counter, &mut v, ret_reg);
                    v
                });
                out.push(MirInst::Branch {
                    cond: c,
                    then_insts,
                    else_insts,
                });
            }
            HirStmt::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                let mut loop_locals = locals.clone();
                loop_locals.insert(var.clone(), format!("%loop_{var}"));
                let mut body_insts = Vec::new();
                lower_stmts(
                    body,
                    consts,
                    &mut loop_locals,
                    counter,
                    &mut body_insts,
                    ret_reg,
                );
                out.push(MirInst::ForLoop {
                    var: var.clone(),
                    start: *start,
                    end: *end,
                    body: body_insts,
                });
            }
            HirStmt::Return { value, .. } => {
                let r = lower_expr(value, consts, locals, counter, out);
                *ret_reg = Some(r);
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
        let (ast_opt, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let Some(ast) = ast_opt else {
            panic!("expected AST for reference_body.aero");
        };
        let (hir, errs) = analyze(&ast);
        assert!(errs.is_empty(), "{errs:?}");
        let mir = lower_to_mir(&hir);
        assert_eq!(mir.layouts[0].total_bytes_per_entity, 104);
        assert_eq!(mir.layouts[0].total_padding_bytes_per_entity, 24);
        assert!(
            mir.format_ir()
                .contains("soa.load.vec3_simd4 @prev.Body.thrust")
        );
    }

    #[test]
    fn optimizes_pure_fn_inlining_loop_unrolling_and_const_folding() {
        let src = r#"
            fn gain(x: m) -> m {
                return x * 2.0;
            }
            state S { pos: Vec3<m>; }
            system K(dt: s) {
                step(prev: S) -> next: S {
                    let c: m = 2.0 m + 3.0 m;
                    let g: m = gain(c);
                    for i in 0..3 {
                        let unused: m = 1.0 m;
                    }
                    next.pos = Vec3(g, prev.pos.y, prev.pos.z);
                }
            }
        "#;
        let (tokens, _) = lex(src);
        let (ast_opt, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let Some(ast) = ast_opt else {
            panic!("expected AST");
        };
        let (hir, errs) = analyze(&ast);
        assert!(errs.is_empty(), "{errs:?}");
        let unopt = lower_to_mir_with_opt(&hir, false);
        let mir = lower_to_mir_with_opt(&hir, true);
        assert_eq!(unopt.opt_stats.inlined_calls, 0);
        assert!(mir.opt_stats.inlined_calls >= 1);
        assert!(mir.opt_stats.unrolled_loops >= 1);
        assert!(mir.opt_stats.folded_constants >= 1);
        assert!(mir.opt_stats.dce_eliminated >= 1);
        assert!(mir.format_stats().contains("cache lines"));
    }
}
