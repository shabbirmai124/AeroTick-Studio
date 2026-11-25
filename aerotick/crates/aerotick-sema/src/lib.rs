//! Phase 3: Semantic Analysis, Symbol Resolution, 7-Component SI Dimensional Algebra,
//! `Vec3<Unit>` Propagation, and Double-Buffered Step Safety Verifier.

use aerotick_ast::{BinOp, Expr, Item, Program, Stmt, TypeExpr, UnitExpr};
use aerotick_span::{Diagnostic, Span};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt;

/// 7 base SI exponents `[Length(m), Mass(kg), Time(s), Current(A), Temp(K), Amount(mol), Luminous(cd)]`
/// plus `angle` exponent for `rad`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DimVector {
    pub si: [i8; 7],
    pub angle: i8,
}

impl DimVector {
    pub const DIMENSIONLESS: Self = Self { si: [0; 7], angle: 0 };
    pub const METER: Self = Self { si: [1, 0, 0, 0, 0, 0, 0], angle: 0 };
    pub const KILOGRAM: Self = Self { si: [0, 1, 0, 0, 0, 0, 0], angle: 0 };
    pub const SECOND: Self = Self { si: [0, 0, 1, 0, 0, 0, 0], angle: 0 };
    pub const AMPERE: Self = Self { si: [0, 0, 0, 1, 0, 0, 0], angle: 0 };
    pub const KELVIN: Self = Self { si: [0, 0, 0, 0, 1, 0, 0], angle: 0 };
    pub const MOLE: Self = Self { si: [0, 0, 0, 0, 0, 1, 0], angle: 0 };
    pub const CANDELA: Self = Self { si: [0, 0, 0, 0, 0, 0, 1], angle: 0 };
    pub const RADIAN: Self = Self { si: [0; 7], angle: 1 };
    pub const NEWTON: Self = Self { si: [1, 1, -2, 0, 0, 0, 0], angle: 0 };
    pub const JOULE: Self = Self { si: [2, 1, -2, 0, 0, 0, 0], angle: 0 };
    pub const WATT: Self = Self { si: [2, 1, -3, 0, 0, 0, 0], angle: 0 };
    pub const PASCAL: Self = Self { si: [-1, 1, -2, 0, 0, 0, 0], angle: 0 };
    pub const HERTZ: Self = Self { si: [0, 0, -1, 0, 0, 0, 0], angle: 0 };

    pub fn mul(self, rhs: Self) -> Self {
        let mut si = [0i8; 7];
        for i in 0..7 {
            si[i] = self.si[i].saturating_add(rhs.si[i]);
        }
        Self { si, angle: self.angle.saturating_add(rhs.angle) }
    }

    pub fn div(self, rhs: Self) -> Self {
        let mut si = [0i8; 7];
        for i in 0..7 {
            si[i] = self.si[i].saturating_sub(rhs.si[i]);
        }
        Self { si, angle: self.angle.saturating_sub(rhs.angle) }
    }

    pub fn pow(self, exp: i8) -> Self {
        let mut si = [0i8; 7];
        for i in 0..7 {
            si[i] = self.si[i].saturating_mul(exp);
        }
        Self { si, angle: self.angle.saturating_mul(exp) }
    }

    pub fn sqrt(self) -> Option<Self> {
        let mut si = [0i8; 7];
        for (i, &e) in self.si.iter().enumerate() {
            if e % 2 != 0 {
                return None;
            }
            si[i] = e / 2;
        }
        if self.angle % 2 != 0 {
            return None;
        }
        Some(Self { si, angle: self.angle / 2 })
    }
}

impl fmt::Display for DimVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::DIMENSIONLESS {
            return write!(f, "1");
        }
        if *self == Self::NEWTON {
            return write!(f, "N (kg*m/s^2)");
        }
        if *self == Self::JOULE {
            return write!(f, "J (kg*m^2/s^2)");
        }
        if *self == Self::WATT {
            return write!(f, "W (kg*m^2/s^3)");
        }
        if *self == Self::PASCAL {
            return write!(f, "Pa (kg/(m*s^2))");
        }
        const NAMES: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
        let mut num = Vec::new();
        let mut den = Vec::new();
        for (idx, &exp) in self.si.iter().enumerate() {
            if exp > 0 {
                num.push(if exp == 1 { NAMES[idx].to_string() } else { format!("{}^{}", NAMES[idx], exp) });
            } else if exp < 0 {
                let abs = exp.saturating_abs();
                den.push(if abs == 1 { NAMES[idx].to_string() } else { format!("{}^{}", NAMES[idx], abs) });
            }
        }
        if self.angle > 0 {
            num.push(if self.angle == 1 { "rad".into() } else { format!("rad^{}", self.angle) });
        } else if self.angle < 0 {
            let abs = self.angle.saturating_abs();
            den.push(if abs == 1 { "rad".into() } else { format!("rad^{}", abs) });
        }
        let num_str = if num.is_empty() { "1".into() } else { num.join("*") };
        if den.is_empty() {
            write!(f, "{num_str}")
        } else if den.len() == 1 {
            write!(f, "{num_str}/{}", den[0])
        } else {
            write!(f, "{num_str}/({})", den.join("*"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    Scalar(DimVector),
    Vec3(DimVector),
    Quat,
    Mat4,
    Bool,
    StateRef { state_name: String, is_next: bool },
}

impl ValueType {
    pub fn is_compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a == b,
            (Self::Vec3(a), Self::Vec3(b)) => a == b,
            (Self::Quat, Self::Quat) => true,
            (Self::Mat4, Self::Mat4) => true,
            (Self::Bool, Self::Bool) => true,
            _ => false,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(d) => write!(f, "Scalar<{d}>"),
            Self::Vec3(d) => write!(f, "Vec3<{d}>"),
            Self::Quat => write!(f, "Quat"),
            Self::Mat4 => write!(f, "Mat4"),
            Self::Bool => write!(f, "Bool"),
            Self::StateRef { state_name, is_next } => {
                write!(f, "{}:{state_name}", if *is_next { "next" } else { "prev" })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HirConst {
    pub name: String,
    pub ty: ValueType,
    pub scalar_value: Option<f64>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirStateField {
    pub name: String,
    pub ty: ValueType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirState {
    pub name: String,
    pub fields: Vec<HirStateField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Number { value: f64, ty: ValueType, span: Span },
    ConstRef { name: String, ty: ValueType, span: Span },
    LocalRef { name: String, ty: ValueType, span: Span },
    ParamRef { name: String, ty: ValueType, span: Span },
    StateFieldRead { is_next: bool, state_name: String, field: String, ty: ValueType, span: Span },
    VecComponent { vec: Box<HirExpr>, component: char, ty: ValueType, span: Span },
    Vec3 { x: Box<HirExpr>, y: Box<HirExpr>, z: Box<HirExpr>, ty: ValueType, span: Span },
    Call { callee: String, args: Vec<HirExpr>, ty: ValueType, span: Span },
    UnaryNeg { operand: Box<HirExpr>, ty: ValueType, span: Span },
    Binary { op: BinOp, left: Box<HirExpr>, right: Box<HirExpr>, ty: ValueType, span: Span },
}

impl HirExpr {
    pub fn ty(&self) -> &ValueType {
        match self {
            Self::Number { ty, .. }
            | Self::ConstRef { ty, .. }
            | Self::LocalRef { ty, .. }
            | Self::ParamRef { ty, .. }
            | Self::StateFieldRead { ty, .. }
            | Self::VecComponent { ty, .. }
            | Self::Vec3 { ty, .. }
            | Self::Call { ty, .. }
            | Self::UnaryNeg { ty, .. }
            | Self::Binary { ty, .. } => ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Let { name: String, ty: ValueType, value: HirExpr, span: Span },
    AssignField { state_name: String, field: String, value: HirExpr, span: Span },
    If { cond: HirExpr, then_stmts: Vec<HirStmt>, else_stmts: Option<Vec<HirStmt>>, span: Span },
    Expr { expr: HirExpr, span: Span },
}

#[derive(Debug, Clone)]
pub struct HirSystem {
    pub name: String,
    pub dt_param: String,
    pub prev_name: String,
    pub next_name: String,
    pub state_name: String,
    pub stmts: Vec<HirStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypedBinding {
    pub name: String,
    pub ty: ValueType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirProgram {
    pub consts: Vec<HirConst>,
    pub states: Vec<HirState>,
    pub systems: Vec<HirSystem>,
    pub top_bindings: Vec<TypedBinding>,
}

fn lookup_unit(symbol: &str) -> Option<(DimVector, f64)> {
    match symbol {
        "m" => Some((DimVector::METER, 1.0)),
        "km" => Some((DimVector::METER, 1000.0)),
        "cm" => Some((DimVector::METER, 0.01)),
        "mm" => Some((DimVector::METER, 0.001)),
        "kg" => Some((DimVector::KILOGRAM, 1.0)),
        "g" => Some((DimVector::KILOGRAM, 0.001)),
        "s" => Some((DimVector::SECOND, 1.0)),
        "ms" => Some((DimVector::SECOND, 0.001)),
        "A" => Some((DimVector::AMPERE, 1.0)),
        "K" => Some((DimVector::KELVIN, 1.0)),
        "mol" => Some((DimVector::MOLE, 1.0)),
        "cd" => Some((DimVector::CANDELA, 1.0)),
        "rad" => Some((DimVector::RADIAN, 1.0)),
        "deg" => Some((DimVector::RADIAN, PI / 180.0)),
        "N" => Some((DimVector::NEWTON, 1.0)),
        "kN" => Some((DimVector::NEWTON, 1000.0)),
        "J" => Some((DimVector::JOULE, 1.0)),
        "W" => Some((DimVector::WATT, 1.0)),
        "Pa" => Some((DimVector::PASCAL, 1.0)),
        "MPa" => Some((DimVector::PASCAL, 1_000_000.0)),
        "Hz" => Some((DimVector::HERTZ, 1.0)),
        _ => None,
    }
}

fn is_forbidden_alloc(callee: &str) -> bool {
    matches!(callee, "alloc" | "malloc" | "calloc" | "realloc" | "new" | "Box" | "Vec" | "push" | "spawn")
}

struct Scope<'a> {
    consts: &'a HashMap<String, HirConst>,
    states: &'a HashMap<String, HirState>,
    locals: HashMap<String, ValueType>,
    params: HashMap<String, ValueType>,
    in_step: bool,
    system_name: Option<String>,
    prev_name: Option<String>,
    next_name: Option<String>,
    state_name: Option<String>,
    assigned_next_fields: HashSet<String>,
}

pub fn analyze(program: &Program) -> (HirProgram, Vec<Diagnostic>) {
    let mut diagnostics = Vec::new();
    let mut consts_map: HashMap<String, HirConst> = HashMap::new();
    let mut consts_order: Vec<HirConst> = Vec::new();
    let mut states_map: HashMap<String, HirState> = HashMap::new();
    let mut states_order: Vec<HirState> = Vec::new();
    let mut systems = Vec::new();
    let mut top_bindings = Vec::new();
    let mut top_locals = HashMap::new();

    // Pass 1: const & state definitions
    for item in &program.items {
        match item {
            Item::Const(c) => {
                let mut scope = Scope {
                    consts: &consts_map,
                    states: &states_map,
                    locals: top_locals.clone(),
                    params: HashMap::new(),
                    in_step: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                };
                if let Some(val) = check_expr(&c.value, &mut scope, &mut diagnostics) {
                    let final_ty = match &c.type_ann {
                        Some(ann) => {
                            if let Some(expected) = resolve_type(ann, &mut diagnostics) {
                                if !expected.is_compatible(val.ty()) {
                                    diagnostics.push(
                                        Diagnostic::error(
                                            "E0240",
                                            format!("Unit mismatch in constant '{}'", c.name),
                                            c.value.span(),
                                            format!("deduced {}", val.ty()),
                                        )
                                        .with_secondary(ann.span(), format!("annotated as {expected}")),
                                    );
                                }
                                expected
                            } else {
                                val.ty().clone()
                            }
                        }
                        None => val.ty().clone(),
                    };
                    let scalar_value = match &val {
                        HirExpr::Number { value, .. } => Some(*value),
                        _ => None,
                    };
                    let hc = HirConst {
                        name: c.name.clone(),
                        ty: final_ty,
                        scalar_value,
                        span: c.span,
                    };
                    consts_map.insert(c.name.clone(), hc.clone());
                    consts_order.push(hc);
                }
            }
            Item::State(st) => {
                if states_map.contains_key(&st.name) {
                    diagnostics.push(Diagnostic::error(
                        "E0250",
                        format!("Duplicate state '{}'", st.name),
                        st.name_span,
                        "already defined",
                    ));
                    continue;
                }
                let mut fields = Vec::new();
                let mut seen = HashSet::new();
                for f in &st.fields {
                    if !seen.insert(f.name.clone()) {
                        diagnostics.push(Diagnostic::error(
                            "E0251",
                            format!("Duplicate field '{}' in state '{}'", f.name, st.name),
                            f.name_span,
                            "duplicate field",
                        ));
                        continue;
                    }
                    if let Some(ty) = resolve_type(&f.type_expr, &mut diagnostics) {
                        fields.push(HirStateField {
                            name: f.name.clone(),
                            ty,
                            span: f.span,
                        });
                    }
                }
                let hs = HirState {
                    name: st.name.clone(),
                    fields,
                    span: st.span,
                };
                states_map.insert(st.name.clone(), hs.clone());
                states_order.push(hs);
            }
            _ => {}
        }
    }

    // Pass 2: systems and top-level calculator bindings
    for item in &program.items {
        match item {
            Item::TopLet { name, name_span, type_ann, value, .. } => {
                let mut scope = Scope {
                    consts: &consts_map,
                    states: &states_map,
                    locals: top_locals.clone(),
                    params: HashMap::new(),
                    in_step: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                };
                if let Some(val) = check_expr(value, &mut scope, &mut diagnostics) {
                    let final_ty = match type_ann {
                        Some(ann) => {
                            if let Some(expected) = resolve_type(ann, &mut diagnostics) {
                                if !expected.is_compatible(val.ty()) {
                                    diagnostics.push(
                                        Diagnostic::error(
                                            "E0240",
                                            format!("Unit mismatch in binding '{name}': expected {expected}, found {}", val.ty()),
                                            value.span(),
                                            format!("deduced {}", val.ty()),
                                        )
                                        .with_secondary(ann.span(), format!("annotated as {expected}")),
                                    );
                                }
                                expected
                            } else {
                                val.ty().clone()
                            }
                        }
                        None => val.ty().clone(),
                    };
                    top_locals.insert(name.clone(), final_ty.clone());
                    top_bindings.push(TypedBinding {
                        name: name.clone(),
                        ty: final_ty,
                        span: *name_span,
                    });
                }
            }
            Item::TopExpr { expr, .. } => {
                let mut scope = Scope {
                    consts: &consts_map,
                    states: &states_map,
                    locals: top_locals.clone(),
                    params: HashMap::new(),
                    in_step: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                };
                let _ = check_expr(expr, &mut scope, &mut diagnostics);
            }
            Item::System(sys) => {
                let mut params = HashMap::new();
                let mut dt_param = "dt".to_string();
                for p in &sys.params {
                    if let Some(pty) = resolve_type(&p.type_expr, &mut diagnostics) {
                        params.insert(p.name.clone(), pty);
                        dt_param = p.name.clone();
                    }
                }
                let prev_state = match states_map.get(&sys.step.prev_type) {
                    Some(st) => st.clone(),
                    None => {
                        diagnostics.push(Diagnostic::error(
                            "E0261",
                            format!("Unknown state '{}'", sys.step.prev_type),
                            sys.step.prev_type_span,
                            "unknown state",
                        ));
                        continue;
                    }
                };
                if sys.step.prev_type != sys.step.next_type {
                    diagnostics.push(
                        Diagnostic::error(
                            "E0262",
                            "Double-buffered step states must share the same type",
                            sys.step.next_type_span,
                            format!("expected '{}'", sys.step.prev_type),
                        )
                        .with_secondary(sys.step.prev_type_span, format!("prev is '{}'", sys.step.prev_type)),
                    );
                }

                params.insert(
                    sys.step.prev_name.clone(),
                    ValueType::StateRef {
                        state_name: prev_state.name.clone(),
                        is_next: false,
                    },
                );
                params.insert(
                    sys.step.next_name.clone(),
                    ValueType::StateRef {
                        state_name: prev_state.name.clone(),
                        is_next: true,
                    },
                );

                let mut scope = Scope {
                    consts: &consts_map,
                    states: &states_map,
                    locals: HashMap::new(),
                    params,
                    in_step: true,
                    system_name: Some(sys.name.clone()),
                    prev_name: Some(sys.step.prev_name.clone()),
                    next_name: Some(sys.step.next_name.clone()),
                    state_name: Some(prev_state.name.clone()),
                    assigned_next_fields: HashSet::new(),
                };

                let mut hir_stmts = Vec::new();
                for s in &sys.step.stmts {
                    if let Some(hs) = check_stmt(s, &mut scope, &mut diagnostics) {
                        hir_stmts.push(hs);
                    }
                }

                for f in &prev_state.fields {
                    if !scope.assigned_next_fields.contains(&f.name) {
                        diagnostics.push(
                            Diagnostic::warning(
                                "W0301",
                                format!("Output field '{}.{}' is never written in step", sys.step.next_name, f.name),
                                sys.step.span,
                                format!("missing assignment to `{}.{}`", sys.step.next_name, f.name),
                            )
                            .with_help(format!("Add `{}.{} = {}.{};` to preserve it across ticks.", sys.step.next_name, f.name, sys.step.prev_name, f.name)),
                        );
                    }
                }

                systems.push(HirSystem {
                    name: sys.name.clone(),
                    dt_param,
                    prev_name: sys.step.prev_name.clone(),
                    next_name: sys.step.next_name.clone(),
                    state_name: prev_state.name.clone(),
                    stmts: hir_stmts,
                    span: sys.span,
                });
            }
            _ => {}
        }
    }

    (
        HirProgram {
            consts: consts_order,
            states: states_order,
            systems,
            top_bindings,
        },
        diagnostics,
    )
}

fn resolve_unit(unit: &UnitExpr, diagnostics: &mut Vec<Diagnostic>) -> Option<(DimVector, f64)> {
    match unit {
        UnitExpr::One(_) => Some((DimVector::DIMENSIONLESS, 1.0)),
        UnitExpr::Symbol(sym, span) => match lookup_unit(sym) {
            Some(pair) => Some(pair),
            None => {
                diagnostics.push(Diagnostic::error(
                    "E0201",
                    format!("Unknown physical unit '{sym}'"),
                    *span,
                    "unrecognized SI unit",
                ));
                None
            }
        },
        UnitExpr::Mul(l, r, _) => {
            let (ld, ls) = resolve_unit(l, diagnostics)?;
            let (rd, rs) = resolve_unit(r, diagnostics)?;
            Some((ld.mul(rd), ls * rs))
        }
        UnitExpr::Div(l, r, _) => {
            let (ld, ls) = resolve_unit(l, diagnostics)?;
            let (rd, rs) = resolve_unit(r, diagnostics)?;
            Some((ld.div(rd), ls / rs))
        }
        UnitExpr::Pow(base, exp, _) => {
            let (bd, bs) = resolve_unit(base, diagnostics)?;
            Some((bd.pow(*exp), bs.powi(i32::from(*exp))))
        }
    }
}

fn resolve_type(te: &TypeExpr, diagnostics: &mut Vec<Diagnostic>) -> Option<ValueType> {
    match te {
        TypeExpr::ScalarUnit(u, _) => resolve_unit(u, diagnostics).map(|(d, _)| ValueType::Scalar(d)),
        TypeExpr::Vec3(u, _) => resolve_unit(u, diagnostics).map(|(d, _)| ValueType::Vec3(d)),
        TypeExpr::Quat(_) => Some(ValueType::Quat),
        TypeExpr::Mat4(_) => Some(ValueType::Mat4),
    }
}

fn check_stmt(stmt: &Stmt, scope: &mut Scope<'_>, diagnostics: &mut Vec<Diagnostic>) -> Option<HirStmt> {
    match stmt {
        Stmt::ForbiddenLoop { keyword, span } => {
            diagnostics.push(Diagnostic::error(
                "E0305",
                format!("Unbounded '{keyword}' loop forbidden inside step"),
                *span,
                "unbounded loop violates real-time determinism",
            ));
            None
        }
        Stmt::Let { name, type_ann, value, span, .. } => {
            let val = check_expr(value, scope, diagnostics)?;
            let ty = match type_ann {
                Some(ann) => {
                    let expected = resolve_type(ann, diagnostics)?;
                    if !expected.is_compatible(val.ty()) {
                        diagnostics.push(
                            Diagnostic::error(
                                "E0240",
                                format!("Type mismatch in 'let {name}'"),
                                value.span(),
                                format!("deduced {}", val.ty()),
                            )
                            .with_secondary(ann.span(), format!("expected {expected}")),
                        );
                    }
                    expected
                }
                None => val.ty().clone(),
            };
            scope.locals.insert(name.clone(), ty.clone());
            Some(HirStmt::Let {
                name: name.clone(),
                ty,
                value: val,
                span: *span,
            })
        }
        Stmt::Assign { target, value, span } => {
            if let Expr::MemberAccess { object, field, field_span, .. } = target {
                if let Expr::Ident(root, _) = object.as_ref() {
                    if scope.prev_name.as_deref() == Some(root.as_str()) {
                        diagnostics.push(
                            Diagnostic::error(
                                "E0301",
                                format!("Cannot mutate immutable prev state '{root}.{field}'"),
                                target.span(),
                                "cannot assign to `prev`",
                            )
                            .with_help(format!(
                                "Write result to `{}.{field}` instead.",
                                scope.next_name.as_deref().unwrap_or("next")
                            )),
                        );
                        let _ = check_expr(value, scope, diagnostics);
                        return None;
                    }
                    if scope.next_name.as_deref() == Some(root.as_str())
                        && let Some(st_name) = scope.state_name.clone()
                        && let Some(st) = scope.states.get(&st_name)
                    {
                        let Some(sf) = st.fields.iter().find(|f| f.name == *field) else {
                            diagnostics.push(Diagnostic::error(
                                "E0204",
                                format!("Unknown field '{field}' on state '{st_name}'"),
                                *field_span,
                                "unknown state field",
                            ));
                            return None;
                        };
                        let val = check_expr(value, scope, diagnostics)?;
                        if !sf.ty.is_compatible(val.ty()) {
                            diagnostics.push(
                                Diagnostic::error(
                                    "E0241",
                                    format!("Unit mismatch assigning to '{root}.{field}'"),
                                    value.span(),
                                    format!("deduced {}", val.ty()),
                                )
                                .with_secondary(sf.span, format!("declared as {}", sf.ty)),
                            );
                        }
                        scope.assigned_next_fields.insert(field.clone());
                        return Some(HirStmt::AssignField {
                            state_name: st_name,
                            field: field.clone(),
                            value: val,
                            span: *span,
                        });
                    }
                }
            }
            diagnostics.push(Diagnostic::error(
                "E0306",
                "Invalid assignment target inside step",
                target.span(),
                "expected `next.<field>`",
            ));
            None
        }
        Stmt::If { cond, then_branch, else_branch, span } => {
            let cond_h = check_expr(cond, scope, diagnostics)?;
            if *cond_h.ty() != ValueType::Bool {
                diagnostics.push(Diagnostic::error(
                    "E0242",
                    "If condition must be a boolean comparison",
                    cond.span(),
                    format!("found {}", cond_h.ty()),
                ));
            }
            let mut then_stmts = Vec::new();
            for s in then_branch {
                if let Some(hs) = check_stmt(s, scope, diagnostics) {
                    then_stmts.push(hs);
                }
            }
            let else_stmts = else_branch.as_ref().map(|branch| {
                branch
                    .iter()
                    .filter_map(|s| check_stmt(s, scope, diagnostics))
                    .collect()
            });
            Some(HirStmt::If {
                cond: cond_h,
                then_stmts,
                else_stmts,
                span: *span,
            })
        }
        Stmt::Expr { expr, span } => {
            let e = check_expr(expr, scope, diagnostics)?;
            Some(HirStmt::Expr { expr: e, span: *span })
        }
    }
}

fn check_expr(expr: &Expr, scope: &mut Scope<'_>, diagnostics: &mut Vec<Diagnostic>) -> Option<HirExpr> {
    match expr {
        Expr::Quantity { value, unit, span } => match unit {
            Some(u) => {
                let (dim, scale) = resolve_unit(u, diagnostics)?;
                Some(HirExpr::Number {
                    value: value * scale,
                    ty: ValueType::Scalar(dim),
                    span: *span,
                })
            }
            None => Some(HirExpr::Number {
                value: *value,
                ty: ValueType::Scalar(DimVector::DIMENSIONLESS),
                span: *span,
            }),
        },
        Expr::Ident(name, span) => {
            if let Some(ty) = scope.locals.get(name) {
                return Some(HirExpr::LocalRef { name: name.clone(), ty: ty.clone(), span: *span });
            }
            if let Some(ty) = scope.params.get(name) {
                return Some(HirExpr::ParamRef { name: name.clone(), ty: ty.clone(), span: *span });
            }
            if let Some(c) = scope.consts.get(name) {
                return Some(HirExpr::ConstRef { name: name.clone(), ty: c.ty.clone(), span: *span });
            }
            diagnostics.push(Diagnostic::error(
                "E0203",
                format!("Undefined symbol '{name}'"),
                *span,
                "not found in scope",
            ));
            None
        }
        Expr::MemberAccess { object, field, field_span, span } => {
            let obj = check_expr(object, scope, diagnostics)?;
            match obj.ty().clone() {
                ValueType::StateRef { state_name, is_next } => {
                    let st = scope.states.get(&state_name)?;
                    let Some(sf) = st.fields.iter().find(|f| f.name == *field) else {
                        diagnostics.push(Diagnostic::error(
                            "E0204",
                            format!("Unknown field '{field}' on state '{state_name}'"),
                            *field_span,
                            "unknown field",
                        ));
                        return None;
                    };
                    if is_next && !scope.assigned_next_fields.contains(field) {
                        diagnostics.push(Diagnostic::error(
                            "E0302",
                            format!("Read from uninitialized next state field '{field}'"),
                            *span,
                            "read before assignment in `next`",
                        ));
                    }
                    Some(HirExpr::StateFieldRead {
                        is_next,
                        state_name,
                        field: field.clone(),
                        ty: sf.ty.clone(),
                        span: *span,
                    })
                }
                ValueType::Vec3(d) => {
                    let comp = match field.as_str() {
                        "x" => 'x',
                        "y" => 'y',
                        "z" => 'z',
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0205",
                                format!("Unknown vector component '.{field}'"),
                                *field_span,
                                "expected x, y, or z",
                            ));
                            return None;
                        }
                    };
                    Some(HirExpr::VecComponent {
                        vec: Box::new(obj),
                        component: comp,
                        ty: ValueType::Scalar(d),
                        span: *span,
                    })
                }
                other => {
                    diagnostics.push(Diagnostic::error(
                        "E0206",
                        format!("Cannot access '.{field}' on {other}"),
                        *field_span,
                        "invalid member access",
                    ));
                    None
                }
            }
        }
        Expr::Vec3Lit { x, y, z, span } => {
            let hx = check_expr(x, scope, diagnostics)?;
            let hy = check_expr(y, scope, diagnostics)?;
            let hz = check_expr(z, scope, diagnostics)?;
            let (ValueType::Scalar(dx), ValueType::Scalar(dy), ValueType::Scalar(dz)) =
                (hx.ty().clone(), hy.ty().clone(), hz.ty().clone())
            else {
                diagnostics.push(Diagnostic::error(
                    "E0210",
                    "Vec3 components must be scalar quantities",
                    *span,
                    "non-scalar Vec3 component",
                ));
                return None;
            };
            let is_zero = |e: &Expr| matches!(e, Expr::Quantity { value, unit: None, .. } if *value == 0.0);
            let mut target_dim = dx;
            if is_zero(x) && dy != DimVector::DIMENSIONLESS {
                target_dim = dy;
            } else if is_zero(x) && dz != DimVector::DIMENSIONLESS {
                target_dim = dz;
            }
            for (d, ast_e, zero) in [
                (dx, x.as_ref(), is_zero(x)),
                (dy, y.as_ref(), is_zero(y)),
                (dz, z.as_ref(), is_zero(z)),
            ] {
                if d != target_dim && !zero {
                    diagnostics.push(Diagnostic::error(
                        "E0211",
                        format!("Inconsistent physical units across Vec3 components: expected {target_dim}, found {d}"),
                        ast_e.span(),
                        format!("has unit {d}"),
                    ));
                    return None;
                }
            }
            Some(HirExpr::Vec3 {
                x: Box::new(hx),
                y: Box::new(hy),
                z: Box::new(hz),
                ty: ValueType::Vec3(target_dim),
                span: *span,
            })
        }
        Expr::Call { callee, callee_span, args, span } => {
            if scope.in_step && is_forbidden_alloc(callee) {
                diagnostics.push(Diagnostic::error(
                    "E0303",
                    format!("Heap allocation '{callee}(...)' is forbidden inside step"),
                    *callee_span,
                    "heap allocation prohibited in hot loop",
                ));
                return None;
            }
            if scope.in_step && scope.system_name.as_deref() == Some(callee.as_str()) {
                diagnostics.push(Diagnostic::error(
                    "E0304",
                    format!("Recursion into '{callee}' is forbidden inside step"),
                    *callee_span,
                    "recursion prohibited",
                ));
                return None;
            }

            let mut h_args = Vec::with_capacity(args.len());
            for a in args {
                h_args.push(check_expr(a, scope, diagnostics)?);
            }

            let ret_ty = match callee.as_str() {
                "dot" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => ValueType::Scalar(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error("E0220", "`dot` requires two Vec3 arguments", *span, "invalid args"));
                        return None;
                    }
                },
                "cross" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => ValueType::Vec3(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error("E0220", "`cross` requires two Vec3 arguments", *span, "invalid args"));
                        return None;
                    }
                },
                "norm" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Vec3(a) => ValueType::Scalar(*a),
                    _ => {
                        diagnostics.push(Diagnostic::error("E0220", "`norm` requires one Vec3 argument", *span, "invalid args"));
                        return None;
                    }
                },
                "normalize" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Vec3(_) => ValueType::Vec3(DimVector::DIMENSIONLESS),
                    _ => {
                        diagnostics.push(Diagnostic::error("E0220", "`normalize` requires one Vec3 argument", *span, "invalid args"));
                        return None;
                    }
                },
                "sqrt" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Scalar(a) => match a.sqrt() {
                        Some(d) => ValueType::Scalar(d),
                        None => {
                            diagnostics.push(Diagnostic::error("E0221", format!("Cannot take sqrt of odd unit {a}"), *span, "odd exponent"));
                            return None;
                        }
                    },
                    _ => {
                        diagnostics.push(Diagnostic::error("E0220", "`sqrt` requires one Scalar argument", *span, "invalid args"));
                        return None;
                    }
                },
                "sin" | "cos" | "tan" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Scalar(d) if d.si == [0; 7] && (d.angle == 0 || d.angle == 1) => {
                        ValueType::Scalar(DimVector::DIMENSIONLESS)
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error("E0222", format!("`{callee}` requires rad or dimensionless angle"), *span, "invalid unit"));
                        return None;
                    }
                },
                "abs" if h_args.len() == 1 => h_args[0].ty().clone(),
                "min" | "max" if h_args.len() == 2 && h_args[0].ty().is_compatible(h_args[1].ty()) => {
                    h_args[0].ty().clone()
                }
                "select" if h_args.len() == 3 && *h_args[0].ty() == ValueType::Bool && h_args[1].ty().is_compatible(h_args[2].ty()) => {
                    h_args[1].ty().clone()
                }
                _ => {
                    diagnostics.push(Diagnostic::error("E0223", format!("Invalid call to '{callee}'"), *callee_span, "unknown or ill-typed intrinsic"));
                    return None;
                }
            };

            Some(HirExpr::Call {
                callee: callee.clone(),
                args: h_args,
                ty: ret_ty,
                span: *span,
            })
        }
        Expr::UnaryNeg(inner, span) => {
            let hi = check_expr(inner, scope, diagnostics)?;
            let ty = hi.ty().clone();
            Some(HirExpr::UnaryNeg {
                operand: Box::new(hi),
                ty,
                span: *span,
            })
        }
        Expr::Binary { op, left, right, span } => {
            let hl = check_expr(left, scope, diagnostics)?;
            let hr = check_expr(right, scope, diagnostics)?;
            let lt = hl.ty().clone();
            let rt = hr.ty().clone();

            let ty = match op {
                BinOp::Add | BinOp::Sub => match (&lt, &rt) {
                    (ValueType::Scalar(a), ValueType::Scalar(b)) if a == b => lt,
                    (ValueType::Vec3(a), ValueType::Vec3(b)) if a == b => lt,
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(
                                "E0230",
                                format!("Physical unit mismatch: cannot apply `{op:?}` to `{lt}` and `{rt}`"),
                                *span,
                                format!("incompatible dimensions: {lt} vs {rt}"),
                            )
                            .with_secondary(left.span(), format!("has type {lt}"))
                            .with_secondary(right.span(), format!("has type {rt}")),
                        );
                        return None;
                    }
                },
                BinOp::Mul => match (&lt, &rt) {
                    (ValueType::Scalar(a), ValueType::Scalar(b)) => ValueType::Scalar(a.mul(*b)),
                    (ValueType::Vec3(a), ValueType::Scalar(b))
                    | (ValueType::Scalar(b), ValueType::Vec3(a)) => ValueType::Vec3(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error("E0233", format!("Unsupported `*` between {lt} and {rt}"), *span, "use dot() or cross() for Vec3 * Vec3"));
                        return None;
                    }
                },
                BinOp::Div => {
                    if matches!(right.as_ref(), Expr::Quantity { value, .. } if *value == 0.0) {
                        diagnostics.push(
                            Diagnostic::error(
                                "E0237",
                                "Division by zero physical quantity",
                                right.span(),
                                "divisor is zero",
                            )
                            .with_help("Ensure divisors in deterministic physics kernels are non-zero."),
                        );
                        return None;
                    }
                    match (&lt, &rt) {
                        (ValueType::Scalar(a), ValueType::Scalar(b)) => ValueType::Scalar(a.div(*b)),
                        (ValueType::Vec3(a), ValueType::Scalar(b)) => ValueType::Vec3(a.div(*b)),
                        _ => {
                            diagnostics.push(Diagnostic::error("E0234", format!("Unsupported `/` between {lt} and {rt}"), *span, "invalid division"));
                            return None;
                        }
                    }
                }
                BinOp::Pow => match (&lt, right.as_ref()) {
                    (ValueType::Scalar(a), Expr::Quantity { value, unit: None, .. })
                        if value.fract() == 0.0 && *value >= -127.0 && *value <= 127.0 =>
                    {
                        ValueType::Scalar(a.pow(*value as i8))
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error("E0235", "Power operator requires scalar base and integer exponent", *span, "invalid exponent"));
                        return None;
                    }
                },
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    match (&lt, &rt) {
                        (ValueType::Scalar(a), ValueType::Scalar(b)) if a == b => ValueType::Bool,
                        _ => {
                            diagnostics.push(Diagnostic::error("E0236", format!("Cannot compare {lt} with {rt}"), *span, "unit mismatch in comparison"));
                            return None;
                        }
                    }
                }
            };

            Some(HirExpr::Binary {
                op: *op,
                left: Box::new(hl),
                right: Box::new(hr),
                ty,
                span: *span,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_parser::parse;

    #[test]
    fn type_checks_reference_body_integration() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, _) = lex(src);
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty());
        let (hir, s_errs) = analyze(&ast.expect("ast"));
        assert!(s_errs.is_empty(), "{s_errs:?}");
        assert_eq!(hir.systems.len(), 1);
        assert_eq!(hir.states[0].fields.len(), 4);
    }

    #[test]
    fn rejects_meters_plus_kilograms_undefined_vars_and_div_zero() {
        let src = "let a = 10.0 m + 5.0 kg; let b = missing_var + 1.0 m; let c = 10.0 m / 0.0 s;";
        let (tokens, _) = lex(src);
        let (ast, _) = parse(&tokens, src.len());
        let (_, s_errs) = analyze(&ast.expect("ast"));
        assert_eq!(s_errs.len(), 3);
        assert_eq!(s_errs[0].code, "E0230");
        assert_eq!(s_errs[1].code, "E0203");
        assert_eq!(s_errs[2].code, "E0237");
    }

    #[test]
    fn rejects_prev_mutation_heap_alloc_and_while_loop() {
        let src = r#"
            state Body { pos: Vec3<m>; vel: Vec3<m/s>; }
            system Bad(dt: s) {
                step(prev: Body) -> next: Body {
                    prev.pos = prev.pos + prev.vel * dt;
                    let x = alloc(64);
                    while dt > 0.0 s { next.vel = prev.vel; }
                    next.pos = prev.pos;
                    next.vel = prev.vel;
                }
            }
        "#;
        let (tokens, _) = lex(src);
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let (_, s_errs) = analyze(&ast.expect("ast"));
        let codes: HashSet<&str> = s_errs.iter().map(|d| d.code).collect();
        assert!(codes.contains("E0301"));
        assert!(codes.contains("E0303"));
        assert!(codes.contains("E0305"));
    }
}
