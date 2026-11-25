#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 3: Semantic Analysis, 7-Component SI + Radian Dimensional Algebra,
//! 6-DoF Quat/Mat3 Type Checking, Pure Function & Step-Safety Verification.

use aerotick_ast::{BinOp, Expr, FnDecl, Item, Program, Stmt, TypeExpr, UnitExpr};
use aerotick_span::{Diagnostic, Span};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt;

/// 7 SI base dimensions `[m, kg, s, A, K, mol, cd]` plus explicit `rad` exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DimVector {
    pub si: [i8; 7],
    pub rad: i8,
}

impl DimVector {
    pub const DIMENSIONLESS: Self = Self { si: [0; 7], rad: 0 };
    pub const METER: Self = Self {
        si: [1, 0, 0, 0, 0, 0, 0],
        rad: 0,
    };
    pub const KILOGRAM: Self = Self {
        si: [0, 1, 0, 0, 0, 0, 0],
        rad: 0,
    };
    pub const SECOND: Self = Self {
        si: [0, 0, 1, 0, 0, 0, 0],
        rad: 0,
    };
    pub const AMPERE: Self = Self {
        si: [0, 0, 0, 1, 0, 0, 0],
        rad: 0,
    };
    pub const KELVIN: Self = Self {
        si: [0, 0, 0, 0, 1, 0, 0],
        rad: 0,
    };
    pub const MOLE: Self = Self {
        si: [0, 0, 0, 0, 0, 1, 0],
        rad: 0,
    };
    pub const CANDELA: Self = Self {
        si: [0, 0, 0, 0, 0, 0, 1],
        rad: 0,
    };
    pub const RADIAN: Self = Self { si: [0; 7], rad: 1 };
    pub const NEWTON: Self = Self {
        si: [1, 1, -2, 0, 0, 0, 0],
        rad: 0,
    };
    pub const JOULE: Self = Self {
        si: [2, 1, -2, 0, 0, 0, 0],
        rad: 0,
    };
    pub const WATT: Self = Self {
        si: [2, 1, -3, 0, 0, 0, 0],
        rad: 0,
    };
    pub const PASCAL: Self = Self {
        si: [-1, 1, -2, 0, 0, 0, 0],
        rad: 0,
    };
    pub const HERTZ: Self = Self {
        si: [0, 0, -1, 0, 0, 0, 0],
        rad: 0,
    };
    pub const ACCEL: Self = Self {
        si: [1, 0, -2, 0, 0, 0, 0],
        rad: 0,
    };
    pub const VELOCITY: Self = Self {
        si: [1, 0, -1, 0, 0, 0, 0],
        rad: 0,
    };
    pub const ANGULAR_VEL: Self = Self {
        si: [0, 0, -1, 0, 0, 0, 0],
        rad: 1,
    };

    pub const fn mul(self, rhs: Self) -> Self {
        Self {
            si: [
                self.si[0].saturating_add(rhs.si[0]),
                self.si[1].saturating_add(rhs.si[1]),
                self.si[2].saturating_add(rhs.si[2]),
                self.si[3].saturating_add(rhs.si[3]),
                self.si[4].saturating_add(rhs.si[4]),
                self.si[5].saturating_add(rhs.si[5]),
                self.si[6].saturating_add(rhs.si[6]),
            ],
            rad: self.rad.saturating_add(rhs.rad),
        }
    }

    pub const fn div(self, rhs: Self) -> Self {
        Self {
            si: [
                self.si[0].saturating_sub(rhs.si[0]),
                self.si[1].saturating_sub(rhs.si[1]),
                self.si[2].saturating_sub(rhs.si[2]),
                self.si[3].saturating_sub(rhs.si[3]),
                self.si[4].saturating_sub(rhs.si[4]),
                self.si[5].saturating_sub(rhs.si[5]),
                self.si[6].saturating_sub(rhs.si[6]),
            ],
            rad: self.rad.saturating_sub(rhs.rad),
        }
    }

    pub const fn pow(self, exp: i8) -> Self {
        Self {
            si: [
                self.si[0].saturating_mul(exp),
                self.si[1].saturating_mul(exp),
                self.si[2].saturating_mul(exp),
                self.si[3].saturating_mul(exp),
                self.si[4].saturating_mul(exp),
                self.si[5].saturating_mul(exp),
                self.si[6].saturating_mul(exp),
            ],
            rad: self.rad.saturating_mul(exp),
        }
    }

    pub const fn inv(self) -> Self {
        Self {
            si: [
                self.si[0].saturating_neg(),
                self.si[1].saturating_neg(),
                self.si[2].saturating_neg(),
                self.si[3].saturating_neg(),
                self.si[4].saturating_neg(),
                self.si[5].saturating_neg(),
                self.si[6].saturating_neg(),
            ],
            rad: self.rad.saturating_neg(),
        }
    }

    pub const fn sqrt(self) -> Option<Self> {
        if self.si[0] % 2 != 0
            || self.si[1] % 2 != 0
            || self.si[2] % 2 != 0
            || self.si[3] % 2 != 0
            || self.si[4] % 2 != 0
            || self.si[5] % 2 != 0
            || self.si[6] % 2 != 0
            || self.rad % 2 != 0
        {
            return None;
        }
        Some(Self {
            si: [
                self.si[0] / 2,
                self.si[1] / 2,
                self.si[2] / 2,
                self.si[3] / 2,
                self.si[4] / 2,
                self.si[5] / 2,
                self.si[6] / 2,
            ],
            rad: self.rad / 2,
        })
    }

    pub const fn cbrt(self) -> Option<Self> {
        if self.si[0] % 3 != 0
            || self.si[1] % 3 != 0
            || self.si[2] % 3 != 0
            || self.si[3] % 3 != 0
            || self.si[4] % 3 != 0
            || self.si[5] % 3 != 0
            || self.si[6] % 3 != 0
            || self.rad % 3 != 0
        {
            return None;
        }
        Some(Self {
            si: [
                self.si[0] / 3,
                self.si[1] / 3,
                self.si[2] / 3,
                self.si[3] / 3,
                self.si[4] / 3,
                self.si[5] / 3,
                self.si[6] / 3,
            ],
            rad: self.rad / 3,
        })
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
        const SYMS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
        let mut num = Vec::new();
        let mut den = Vec::new();
        for (i, &e) in self.si.iter().enumerate() {
            if e > 0 {
                if e == 1 {
                    num.push(SYMS[i].to_string());
                } else {
                    num.push(format!("{}^{e}", SYMS[i]));
                }
            } else if e < 0 {
                let mag = e.unsigned_abs();
                if mag == 1 {
                    den.push(SYMS[i].to_string());
                } else {
                    den.push(format!("{}^{mag}", SYMS[i]));
                }
            }
        }
        if self.rad > 0 {
            if self.rad == 1 {
                num.push("rad".to_string());
            } else {
                num.push(format!("rad^{}", self.rad));
            }
        } else if self.rad < 0 {
            let mag = self.rad.unsigned_abs();
            if mag == 1 {
                den.push("rad".to_string());
            } else {
                den.push(format!("rad^{mag}"));
            }
        }
        let num_str = if num.is_empty() {
            "1".to_string()
        } else {
            num.join("*")
        };
        if den.is_empty() {
            write!(f, "{num_str}")
        } else {
            write!(f, "{num_str}/{}", den.join("*"))
        }
    }
}

/// Built-in SI base, derived, and scaled aerospace units with exact SI conversion factors.
pub fn lookup_builtin_unit(sym: &str) -> Option<(DimVector, f64)> {
    match sym {
        // Length
        "m" => Some((DimVector::METER, 1.0)),
        "km" => Some((DimVector::METER, 1_000.0)),
        "cm" => Some((DimVector::METER, 0.01)),
        "mm" => Some((DimVector::METER, 0.001)),
        "ft" => Some((DimVector::METER, 0.3048)),
        "in" => Some((DimVector::METER, 0.0254)),
        // Mass
        "kg" => Some((DimVector::KILOGRAM, 1.0)),
        "g" => Some((DimVector::KILOGRAM, 0.001)),
        // Time
        "s" => Some((DimVector::SECOND, 1.0)),
        "ms" => Some((DimVector::SECOND, 0.001)),
        // Remaining SI base units
        "A" => Some((DimVector::AMPERE, 1.0)),
        "K" => Some((DimVector::KELVIN, 1.0)),
        "mol" => Some((DimVector::MOLE, 1.0)),
        "cd" => Some((DimVector::CANDELA, 1.0)),
        // Angle & rotational speed
        "rad" => Some((DimVector::RADIAN, 1.0)),
        "deg" => Some((DimVector::RADIAN, PI / 180.0)),
        "rpm" => Some((DimVector::ANGULAR_VEL, (2.0 * PI) / 60.0)),
        // Velocity & acceleration
        "kn" => Some((DimVector::VELOCITY, 1_852.0 / 3_600.0)),
        "g0" => Some((DimVector::ACCEL, 9.80665)),
        // Force, energy, power, pressure, frequency
        "N" => Some((DimVector::NEWTON, 1.0)),
        "kN" => Some((DimVector::NEWTON, 1_000.0)),
        "J" => Some((DimVector::JOULE, 1.0)),
        "W" => Some((DimVector::WATT, 1.0)),
        "Pa" => Some((DimVector::PASCAL, 1.0)),
        "MPa" => Some((DimVector::PASCAL, 1_000_000.0)),
        "bar" => Some((DimVector::PASCAL, 100_000.0)),
        "atm" => Some((DimVector::PASCAL, 101_325.0)),
        "Hz" => Some((DimVector::HERTZ, 1.0)),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    Scalar(DimVector),
    Vec3(DimVector),
    Quat,
    Mat3(DimVector),
    Mat4,
    Bool,
    StateRef { state_name: String, is_next: bool },
}

impl ValueType {
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }

    pub fn alignment_label(&self) -> &'static str {
        match self {
            Self::Vec3(_) => "32B SIMD4",
            Self::Quat => "32B SIMD4 Quat",
            Self::Mat3(_) => "32B SIMD12 Mat3",
            Self::Mat4 => "32B SIMD16 Mat4",
            Self::Bool => "1B bool",
            Self::Scalar(_) => "8B f64",
            Self::StateRef { .. } => "SoA ptr",
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(d) => write!(f, "Scalar<{d}>"),
            Self::Vec3(d) => write!(f, "Vec3<{d}>"),
            Self::Quat => write!(f, "Quat"),
            Self::Mat3(d) => write!(f, "Mat3<{d}>"),
            Self::Mat4 => write!(f, "Mat4"),
            Self::Bool => write!(f, "Bool"),
            Self::StateRef {
                state_name,
                is_next,
            } => {
                if *is_next {
                    write!(f, "next:{state_name}")
                } else {
                    write!(f, "prev:{state_name}")
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    UnitAlias,
    Const,
    State,
    StateField,
    Function,
    System,
    Local,
    Param,
    Test,
}

#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub detail: String,
    pub span: Span,
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
pub struct HirParam {
    pub name: String,
    pub ty: ValueType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirFn {
    pub name: String,
    pub params: Vec<HirParam>,
    pub return_type: ValueType,
    pub body: Vec<HirStmt>,
    pub span: Span,
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
pub struct HirTest {
    pub name: String,
    pub body: Vec<HirStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypedBinding {
    pub name: String,
    pub ty: ValueType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Let {
        name: String,
        ty: ValueType,
        value: HirExpr,
        span: Span,
    },
    AssignField {
        state_name: String,
        field: String,
        value: HirExpr,
        span: Span,
    },
    If {
        cond: HirExpr,
        then_stmts: Vec<HirStmt>,
        else_stmts: Option<Vec<HirStmt>>,
        span: Span,
    },
    For {
        var: String,
        start: i64,
        end: i64,
        body: Vec<HirStmt>,
        span: Span,
    },
    Return {
        value: HirExpr,
        span: Span,
    },
    Expr {
        expr: HirExpr,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Number {
        value: f64,
        ty: ValueType,
        span: Span,
    },
    ConstRef {
        name: String,
        ty: ValueType,
        span: Span,
    },
    LocalRef {
        name: String,
        ty: ValueType,
        span: Span,
    },
    ParamRef {
        name: String,
        ty: ValueType,
        span: Span,
    },
    StateFieldRead {
        is_next: bool,
        state_name: String,
        field: String,
        ty: ValueType,
        span: Span,
    },
    VecComponent {
        vec: Box<HirExpr>,
        component: char,
        ty: ValueType,
        span: Span,
    },
    Vec3 {
        x: Box<HirExpr>,
        y: Box<HirExpr>,
        z: Box<HirExpr>,
        ty: ValueType,
        span: Span,
    },
    Quat {
        w: Box<HirExpr>,
        x: Box<HirExpr>,
        y: Box<HirExpr>,
        z: Box<HirExpr>,
        ty: ValueType,
        span: Span,
    },
    Mat3 {
        elems: Box<[HirExpr; 9]>,
        ty: ValueType,
        span: Span,
    },
    IfExpr {
        cond: Box<HirExpr>,
        then_expr: Box<HirExpr>,
        else_expr: Box<HirExpr>,
        ty: ValueType,
        span: Span,
    },
    UnaryNeg {
        operand: Box<HirExpr>,
        ty: ValueType,
        span: Span,
    },
    Binary {
        op: BinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
        ty: ValueType,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<HirExpr>,
        ty: ValueType,
        span: Span,
    },
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
            | Self::Quat { ty, .. }
            | Self::Mat3 { ty, .. }
            | Self::IfExpr { ty, .. }
            | Self::UnaryNeg { ty, .. }
            | Self::Binary { ty, .. }
            | Self::Call { ty, .. } => ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HirProgram {
    pub unit_aliases: Vec<(String, DimVector, f64, Span)>,
    pub consts: Vec<HirConst>,
    pub states: Vec<HirState>,
    pub functions: Vec<HirFn>,
    pub systems: Vec<HirSystem>,
    pub tests: Vec<HirTest>,
    pub top_bindings: Vec<TypedBinding>,
    pub symbols: Vec<SymbolInfo>,
}

#[derive(Clone)]
struct FnSig {
    params: Vec<ValueType>,
    ret: ValueType,
}

struct Scope<'a> {
    unit_aliases: &'a HashMap<String, (DimVector, f64)>,
    consts: &'a HashMap<String, HirConst>,
    states: &'a HashMap<String, HirState>,
    fn_sigs: &'a HashMap<String, FnSig>,
    locals: HashMap<String, ValueType>,
    params: HashMap<String, ValueType>,
    in_step: bool,
    in_fn: bool,
    in_test: bool,
    current_fn: Option<String>,
    expected_return: Option<ValueType>,
    saw_return: bool,
    system_name: Option<String>,
    prev_name: Option<String>,
    next_name: Option<String>,
    state_name: Option<String>,
    assigned_next_fields: HashSet<String>,
    symbols: &'a mut Vec<SymbolInfo>,
}

fn is_forbidden_alloc(name: &str) -> bool {
    matches!(
        name,
        "alloc" | "malloc" | "calloc" | "realloc" | "Box" | "Vec" | "new" | "spawn"
    )
}

/// Runs full semantic analysis, SI dimensional algebra verification, and step-safety checks.
pub fn analyze(program: &Program) -> (HirProgram, Vec<Diagnostic>) {
    let mut diagnostics = Vec::new();
    let mut unit_aliases_map: HashMap<String, (DimVector, f64)> = HashMap::new();
    let mut unit_aliases_order: Vec<(String, DimVector, f64, Span)> = Vec::new();
    let mut consts_map: HashMap<String, HirConst> = HashMap::new();
    let mut consts_order: Vec<HirConst> = Vec::new();
    let mut states_map: HashMap<String, HirState> = HashMap::new();
    let mut states_order: Vec<HirState> = Vec::new();
    let mut fn_sigs: HashMap<String, FnSig> = HashMap::new();
    let mut raw_fns: Vec<&FnDecl> = Vec::new();
    let mut symbols: Vec<SymbolInfo> = Vec::new();

    // Pass 1a: unit aliases, states, constants, and function signatures
    for item in &program.items {
        match item {
            Item::UnitAlias(ua) => {
                if let Some((dim, scale)) =
                    resolve_unit(&ua.unit, &unit_aliases_map, &mut diagnostics)
                {
                    unit_aliases_map.insert(ua.name.clone(), (dim, scale));
                    unit_aliases_order.push((ua.name.clone(), dim, scale, ua.span));
                    symbols.push(SymbolInfo {
                        name: ua.name.clone(),
                        kind: SymbolKind::UnitAlias,
                        detail: format!("unit {} = {} (scale {})", ua.name, dim, scale),
                        span: ua.name_span,
                    });
                }
            }
            Item::State(st) => {
                let mut fields = Vec::with_capacity(st.fields.len());
                for f in &st.fields {
                    if let Some(ty) =
                        resolve_type(&f.type_expr, &unit_aliases_map, &mut diagnostics)
                    {
                        symbols.push(SymbolInfo {
                            name: f.name.clone(),
                            kind: SymbolKind::StateField,
                            detail: format!("{}: {} [{}]", f.name, ty, ty.alignment_label()),
                            span: f.name_span,
                        });
                        fields.push(HirStateField {
                            name: f.name.clone(),
                            ty,
                            span: f.span,
                        });
                    }
                }
                symbols.push(SymbolInfo {
                    name: st.name.clone(),
                    kind: SymbolKind::State,
                    detail: format!("state {} ({} fields)", st.name, fields.len()),
                    span: st.name_span,
                });
                let hir_st = HirState {
                    name: st.name.clone(),
                    fields,
                    span: st.span,
                };
                states_map.insert(st.name.clone(), hir_st.clone());
                states_order.push(hir_st);
            }
            Item::Const(c) => {
                let mut scope = Scope {
                    unit_aliases: &unit_aliases_map,
                    consts: &consts_map,
                    states: &states_map,
                    fn_sigs: &fn_sigs,
                    locals: HashMap::new(),
                    params: HashMap::new(),
                    in_step: false,
                    in_fn: false,
                    in_test: false,
                    current_fn: None,
                    expected_return: None,
                    saw_return: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                    symbols: &mut symbols,
                };
                if let Some(val) = check_expr(&c.value, &mut scope, &mut diagnostics) {
                    let final_ty = match &c.type_ann {
                        Some(ann) => {
                            if let Some(expected) =
                                resolve_type(ann, &unit_aliases_map, &mut diagnostics)
                            {
                                if !expected.is_compatible(val.ty()) {
                                    let code = mismatch_code(&expected, val.ty());
                                    diagnostics.push(
                                        Diagnostic::error(
                                            code,
                                            format!(
                                                "Mismatch in const '{}': expected {expected}, found {}",
                                                c.name,
                                                val.ty()
                                            ),
                                            c.value.span(),
                                            format!("deduced {}", val.ty()),
                                        )
                                        .with_secondary(ann.span(), format!("expected {expected}")),
                                    );
                                }
                                expected
                            } else {
                                val.ty().clone()
                            }
                        }
                        None => val.ty().clone(),
                    };
                    let scalar_value = eval_const_scalar(&val, &consts_map);
                    symbols.push(SymbolInfo {
                        name: c.name.clone(),
                        kind: SymbolKind::Const,
                        detail: format!(
                            "const {}: {} [{}]",
                            c.name,
                            final_ty,
                            final_ty.alignment_label()
                        ),
                        span: c.name_span,
                    });
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
            Item::Fn(f) => {
                let mut param_tys = Vec::with_capacity(f.params.len());
                let mut ok = true;
                for p in &f.params {
                    if let Some(pty) =
                        resolve_type(&p.type_expr, &unit_aliases_map, &mut diagnostics)
                    {
                        param_tys.push(pty);
                    } else {
                        ok = false;
                    }
                }
                if let Some(ret_ty) =
                    resolve_type(&f.return_type, &unit_aliases_map, &mut diagnostics)
                    && ok
                {
                    fn_sigs.insert(
                        f.name.clone(),
                        FnSig {
                            params: param_tys,
                            ret: ret_ty.clone(),
                        },
                    );
                    symbols.push(SymbolInfo {
                        name: f.name.clone(),
                        kind: SymbolKind::Function,
                        detail: format!("pure fn {} -> {ret_ty}", f.name),
                        span: f.name_span,
                    });
                    raw_fns.push(f);
                }
            }
            _ => {}
        }
    }

    // Pass 1b: type-check pure function bodies
    let mut functions = Vec::new();
    for f in raw_fns {
        let Some(sig) = fn_sigs.get(&f.name).cloned() else {
            continue;
        };
        let mut params = HashMap::new();
        let mut hir_params = Vec::with_capacity(f.params.len());
        for (p, pty) in f.params.iter().zip(sig.params.iter()) {
            params.insert(p.name.clone(), pty.clone());
            symbols.push(SymbolInfo {
                name: p.name.clone(),
                kind: SymbolKind::Param,
                detail: format!("{}: {} [{}]", p.name, pty, pty.alignment_label()),
                span: p.name_span,
            });
            hir_params.push(HirParam {
                name: p.name.clone(),
                ty: pty.clone(),
                span: p.span,
            });
        }
        let mut scope = Scope {
            unit_aliases: &unit_aliases_map,
            consts: &consts_map,
            states: &states_map,
            fn_sigs: &fn_sigs,
            locals: HashMap::new(),
            params,
            in_step: false,
            in_fn: true,
            in_test: false,
            current_fn: Some(f.name.clone()),
            expected_return: Some(sig.ret.clone()),
            saw_return: false,
            system_name: None,
            prev_name: None,
            next_name: None,
            state_name: None,
            assigned_next_fields: HashSet::new(),
            symbols: &mut symbols,
        };
        let mut body = Vec::new();
        for s in &f.body {
            if let Some(hs) = check_stmt(s, &mut scope, &mut diagnostics) {
                body.push(hs);
            }
        }
        if !scope.saw_return {
            diagnostics.push(Diagnostic::error(
                "E0203",
                format!(
                    "Pure function '{}' must end with a `return` statement",
                    f.name
                ),
                f.span,
                "missing return statement",
            ));
        }
        functions.push(HirFn {
            name: f.name.clone(),
            params: hir_params,
            return_type: sig.ret,
            body,
            span: f.span,
        });
    }

    // Pass 1c: Detect mutual recursion across pure functions (`E0304`)
    check_mutual_recursion(&functions, &mut diagnostics);

    // Pass 2: systems, tests, and top-level bindings
    let mut systems = Vec::new();
    let mut tests = Vec::new();
    let mut top_bindings = Vec::new();
    let mut top_locals: HashMap<String, ValueType> = HashMap::new();

    for item in &program.items {
        match item {
            Item::TopLet {
                name,
                name_span,
                type_ann,
                value,
                ..
            } => {
                let mut scope = Scope {
                    unit_aliases: &unit_aliases_map,
                    consts: &consts_map,
                    states: &states_map,
                    fn_sigs: &fn_sigs,
                    locals: top_locals.clone(),
                    params: HashMap::new(),
                    in_step: false,
                    in_fn: false,
                    in_test: false,
                    current_fn: None,
                    expected_return: None,
                    saw_return: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                    symbols: &mut symbols,
                };
                if let Some(val) = check_expr(value, &mut scope, &mut diagnostics) {
                    let final_ty = match type_ann {
                        Some(ann) => {
                            if let Some(expected) =
                                resolve_type(ann, &unit_aliases_map, &mut diagnostics)
                            {
                                if !expected.is_compatible(val.ty()) {
                                    let code = mismatch_code(&expected, val.ty());
                                    diagnostics.push(
                                        Diagnostic::error(
                                            code,
                                            format!(
                                                "Mismatch in binding '{name}': expected {expected}, found {}",
                                                val.ty()
                                            ),
                                            value.span(),
                                            format!("deduced {}", val.ty()),
                                        )
                                        .with_secondary(
                                            ann.span(),
                                            format!("annotated as {expected}"),
                                        ),
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
                        ty: final_ty.clone(),
                        span: *name_span,
                    });
                    symbols.push(SymbolInfo {
                        name: name.clone(),
                        kind: SymbolKind::Local,
                        detail: format!("{}: {} [{}]", name, final_ty, final_ty.alignment_label()),
                        span: *name_span,
                    });
                }
            }
            Item::TopExpr { expr, .. } => {
                let mut scope = Scope {
                    unit_aliases: &unit_aliases_map,
                    consts: &consts_map,
                    states: &states_map,
                    fn_sigs: &fn_sigs,
                    locals: top_locals.clone(),
                    params: HashMap::new(),
                    in_step: false,
                    in_fn: false,
                    in_test: false,
                    current_fn: None,
                    expected_return: None,
                    saw_return: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                    symbols: &mut symbols,
                };
                let _ = check_expr(expr, &mut scope, &mut diagnostics);
            }
            Item::System(sys) => {
                let mut params = HashMap::new();
                let mut dt_param = "dt".to_string();
                for p in &sys.params {
                    if let Some(pty) =
                        resolve_type(&p.type_expr, &unit_aliases_map, &mut diagnostics)
                    {
                        params.insert(p.name.clone(), pty.clone());
                        symbols.push(SymbolInfo {
                            name: p.name.clone(),
                            kind: SymbolKind::Param,
                            detail: format!("{}: {} [{}]", p.name, pty, pty.alignment_label()),
                            span: p.name_span,
                        });
                        dt_param = p.name.clone();
                    }
                }
                let prev_state = match states_map.get(&sys.step.prev_type) {
                    Some(st) => st.clone(),
                    None => {
                        diagnostics.push(Diagnostic::error(
                            "E0201",
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
                            "E0203",
                            "Double-buffered step states must share the same type",
                            sys.step.next_type_span,
                            format!("expected '{}'", sys.step.prev_type),
                        )
                        .with_secondary(
                            sys.step.prev_type_span,
                            format!("prev is '{}'", sys.step.prev_type),
                        ),
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
                    unit_aliases: &unit_aliases_map,
                    consts: &consts_map,
                    states: &states_map,
                    fn_sigs: &fn_sigs,
                    locals: HashMap::new(),
                    params,
                    in_step: true,
                    in_fn: false,
                    in_test: false,
                    current_fn: None,
                    expected_return: None,
                    saw_return: false,
                    system_name: Some(sys.name.clone()),
                    prev_name: Some(sys.step.prev_name.clone()),
                    next_name: Some(sys.step.next_name.clone()),
                    state_name: Some(prev_state.name.clone()),
                    assigned_next_fields: HashSet::new(),
                    symbols: &mut symbols,
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
                                format!(
                                    "Output field '{}.{}' is never written in step",
                                    sys.step.next_name, f.name
                                ),
                                sys.step.span,
                                format!(
                                    "missing assignment to `{}.{}`",
                                    sys.step.next_name, f.name
                                ),
                            )
                            .with_help(format!(
                                "Add `{}.{} = {}.{};` to preserve it across ticks.",
                                sys.step.next_name, f.name, sys.step.prev_name, f.name
                            )),
                        );
                    }
                }

                symbols.push(SymbolInfo {
                    name: sys.name.clone(),
                    kind: SymbolKind::System,
                    detail: format!(
                        "system {}({}: s) -> {}",
                        sys.name, dt_param, prev_state.name
                    ),
                    span: sys.name_span,
                });

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
            Item::Test(t) => {
                let mut scope = Scope {
                    unit_aliases: &unit_aliases_map,
                    consts: &consts_map,
                    states: &states_map,
                    fn_sigs: &fn_sigs,
                    locals: HashMap::new(),
                    params: HashMap::new(),
                    in_step: false,
                    in_fn: false,
                    in_test: true,
                    current_fn: None,
                    expected_return: None,
                    saw_return: false,
                    system_name: None,
                    prev_name: None,
                    next_name: None,
                    state_name: None,
                    assigned_next_fields: HashSet::new(),
                    symbols: &mut symbols,
                };
                let mut body = Vec::new();
                for s in &t.body {
                    if let Some(hs) = check_stmt(s, &mut scope, &mut diagnostics) {
                        body.push(hs);
                    }
                }
                symbols.push(SymbolInfo {
                    name: t.name.clone(),
                    kind: SymbolKind::Test,
                    detail: format!("test \"{}\"", t.name),
                    span: t.name_span,
                });
                tests.push(HirTest {
                    name: t.name.clone(),
                    body,
                    span: t.span,
                });
            }
            _ => {}
        }
    }

    (
        HirProgram {
            unit_aliases: unit_aliases_order,
            consts: consts_order,
            states: states_order,
            functions,
            systems,
            tests,
            top_bindings,
            symbols,
        },
        diagnostics,
    )
}

fn eval_const_scalar(expr: &HirExpr, consts: &HashMap<String, HirConst>) -> Option<f64> {
    match expr {
        HirExpr::Number { value, .. } => Some(*value),
        HirExpr::ConstRef { name, .. } => consts.get(name).and_then(|c| c.scalar_value),
        HirExpr::UnaryNeg { operand, .. } => eval_const_scalar(operand, consts).map(|v| -v),
        HirExpr::Binary {
            op, left, right, ..
        } => {
            let l = eval_const_scalar(left, consts)?;
            let r = eval_const_scalar(right, consts)?;
            Some(match op {
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
            })
        }
        _ => None,
    }
}

fn eval_static_int(expr: &HirExpr, consts: &HashMap<String, HirConst>) -> Option<i64> {
    if *expr.ty() != ValueType::Scalar(DimVector::DIMENSIONLESS) {
        return None;
    }
    let val = eval_const_scalar(expr, consts)?;
    if val.is_finite() && val.fract() == 0.0 && (-1_000_000.0..=1_000_000.0).contains(&val) {
        Some(val as i64)
    } else {
        None
    }
}

fn resolve_unit(
    unit: &UnitExpr,
    aliases: &HashMap<String, (DimVector, f64)>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<(DimVector, f64)> {
    match unit {
        UnitExpr::One(_) => Some((DimVector::DIMENSIONLESS, 1.0)),
        UnitExpr::Symbol(sym, span) => {
            if let Some(pair) = aliases.get(sym.as_str()) {
                return Some(*pair);
            }
            if let Some(pair) = lookup_builtin_unit(sym) {
                return Some(pair);
            }
            diagnostics.push(Diagnostic::error(
                "E0201",
                format!("Unknown physical unit '{sym}'"),
                *span,
                "unrecognized SI unit",
            ));
            None
        }
        UnitExpr::Mul(l, r, _) => {
            let (ld, ls) = resolve_unit(l, aliases, diagnostics)?;
            let (rd, rs) = resolve_unit(r, aliases, diagnostics)?;
            Some((ld.mul(rd), ls * rs))
        }
        UnitExpr::Div(l, r, _) => {
            let (ld, ls) = resolve_unit(l, aliases, diagnostics)?;
            let (rd, rs) = resolve_unit(r, aliases, diagnostics)?;
            Some((ld.div(rd), ls / rs))
        }
        UnitExpr::Pow(base, exp, _) => {
            let (bd, bs) = resolve_unit(base, aliases, diagnostics)?;
            Some((bd.pow(*exp), bs.powi(i32::from(*exp))))
        }
    }
}

fn resolve_type(
    te: &TypeExpr,
    aliases: &HashMap<String, (DimVector, f64)>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ValueType> {
    match te {
        TypeExpr::ScalarUnit(u, _) => {
            resolve_unit(u, aliases, diagnostics).map(|(d, _)| ValueType::Scalar(d))
        }
        TypeExpr::Vec3(u, _) => {
            resolve_unit(u, aliases, diagnostics).map(|(d, _)| ValueType::Vec3(d))
        }
        TypeExpr::Quat(_) => Some(ValueType::Quat),
        TypeExpr::Mat3(u, _) => {
            resolve_unit(u, aliases, diagnostics).map(|(d, _)| ValueType::Mat3(d))
        }
        TypeExpr::Mat4(_) => Some(ValueType::Mat4),
    }
}

fn mismatch_code(expected: &ValueType, found: &ValueType) -> &'static str {
    match (expected, found) {
        (ValueType::Scalar(_), ValueType::Scalar(_))
        | (ValueType::Vec3(_), ValueType::Vec3(_))
        | (ValueType::Mat3(_), ValueType::Mat3(_)) => "E0202",
        _ => "E0203",
    }
}

fn check_mutual_recursion(functions: &[HirFn], diagnostics: &mut Vec<Diagnostic>) {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let fn_spans: HashMap<&str, Span> = functions
        .iter()
        .map(|f| (f.name.as_str(), f.span))
        .collect();
    for f in functions {
        let mut calls = Vec::new();
        collect_calls_in_stmts(&f.body, &mut calls);
        adj.insert(f.name.as_str(), calls);
    }

    for f in functions {
        let start = f.name.as_str();
        let mut stack: Vec<&str> = adj.get(start).cloned().unwrap_or_default();
        let mut visited: HashSet<&str> = HashSet::new();
        while let Some(callee) = stack.pop() {
            if callee == start {
                let directly_self =
                    adj.get(start).is_some_and(|v| v.contains(&start)) && visited.is_empty();
                if !directly_self {
                    let span = fn_spans.get(start).copied().unwrap_or(f.span);
                    diagnostics.push(Diagnostic::error(
                        "E0304",
                        format!("Mutual recursion involving pure function '{start}' is forbidden"),
                        span,
                        "cyclic call graph violates static inlining guarantee",
                    ));
                }
                break;
            }
            if visited.insert(callee)
                && let Some(nexts) = adj.get(callee)
            {
                for n in nexts {
                    stack.push(n);
                }
            }
        }
    }
}

fn collect_calls_in_stmts<'a>(stmts: &'a [HirStmt], out: &mut Vec<&'a str>) {
    for s in stmts {
        match s {
            HirStmt::Let { value, .. }
            | HirStmt::AssignField { value, .. }
            | HirStmt::Return { value, .. } => collect_calls_in_expr(value, out),
            HirStmt::If {
                cond,
                then_stmts,
                else_stmts,
                ..
            } => {
                collect_calls_in_expr(cond, out);
                collect_calls_in_stmts(then_stmts, out);
                if let Some(el) = else_stmts {
                    collect_calls_in_stmts(el, out);
                }
            }
            HirStmt::For { body, .. } => collect_calls_in_stmts(body, out),
            HirStmt::Expr { expr, .. } => collect_calls_in_expr(expr, out),
        }
    }
}

fn collect_calls_in_expr<'a>(expr: &'a HirExpr, out: &mut Vec<&'a str>) {
    match expr {
        HirExpr::VecComponent { vec, .. } => collect_calls_in_expr(vec, out),
        HirExpr::Vec3 { x, y, z, .. } => {
            collect_calls_in_expr(x, out);
            collect_calls_in_expr(y, out);
            collect_calls_in_expr(z, out);
        }
        HirExpr::Quat { w, x, y, z, .. } => {
            collect_calls_in_expr(w, out);
            collect_calls_in_expr(x, out);
            collect_calls_in_expr(y, out);
            collect_calls_in_expr(z, out);
        }
        HirExpr::Mat3 { elems, .. } => {
            for el in elems.iter() {
                collect_calls_in_expr(el, out);
            }
        }
        HirExpr::IfExpr {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_calls_in_expr(cond, out);
            collect_calls_in_expr(then_expr, out);
            collect_calls_in_expr(else_expr, out);
        }
        HirExpr::UnaryNeg { operand, .. } => collect_calls_in_expr(operand, out),
        HirExpr::Binary { left, right, .. } => {
            collect_calls_in_expr(left, out);
            collect_calls_in_expr(right, out);
        }
        HirExpr::Call { callee, args, .. } => {
            out.push(callee.as_str());
            for a in args {
                collect_calls_in_expr(a, out);
            }
        }
        HirExpr::Number { .. }
        | HirExpr::ConstRef { .. }
        | HirExpr::LocalRef { .. }
        | HirExpr::ParamRef { .. }
        | HirExpr::StateFieldRead { .. } => {}
    }
}

fn check_stmt(
    stmt: &Stmt,
    scope: &mut Scope<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<HirStmt> {
    match stmt {
        Stmt::ForbiddenLoop { keyword, span } => {
            diagnostics.push(Diagnostic::error(
                "E0305",
                format!("Unbounded '{keyword}' loop forbidden inside deterministic kernel"),
                *span,
                "unbounded loop violates real-time determinism",
            ));
            None
        }
        Stmt::Let {
            name,
            name_span,
            type_ann,
            value,
            span,
        } => {
            let val = check_expr(value, scope, diagnostics)?;
            let ty = match type_ann {
                Some(ann) => {
                    let expected = resolve_type(ann, scope.unit_aliases, diagnostics)?;
                    if !expected.is_compatible(val.ty()) {
                        let code = mismatch_code(&expected, val.ty());
                        diagnostics.push(
                            Diagnostic::error(
                                code,
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
            scope.symbols.push(SymbolInfo {
                name: name.clone(),
                kind: SymbolKind::Local,
                detail: format!("{}: {} [{}]", name, ty, ty.alignment_label()),
                span: *name_span,
            });
            Some(HirStmt::Let {
                name: name.clone(),
                ty,
                value: val,
                span: *span,
            })
        }
        Stmt::Assign {
            target,
            value,
            span,
        } => {
            if let Expr::MemberAccess {
                object,
                field,
                field_span,
                ..
            } = target
                && let Expr::Ident(root, _) = object.as_ref()
            {
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
                            "E0201",
                            format!("Unknown field '{field}' on state '{st_name}'"),
                            *field_span,
                            "unknown state field",
                        ));
                        return None;
                    };
                    let val = check_expr(value, scope, diagnostics)?;
                    if !sf.ty.is_compatible(val.ty()) {
                        let code = mismatch_code(&sf.ty, val.ty());
                        diagnostics.push(
                            Diagnostic::error(
                                code,
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
            diagnostics.push(Diagnostic::error(
                "E0301",
                "Invalid assignment target inside step",
                target.span(),
                "expected `next.<field>`",
            ));
            None
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            span,
        } => {
            let cond_h = check_expr(cond, scope, diagnostics)?;
            if *cond_h.ty() != ValueType::Bool {
                diagnostics.push(Diagnostic::error(
                    "E0203",
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
        Stmt::For {
            var,
            var_span,
            start,
            end,
            body,
            span,
        } => {
            let hs = check_expr(start, scope, diagnostics)?;
            let he = check_expr(end, scope, diagnostics)?;
            let Some(start_i) = eval_static_int(&hs, scope.consts) else {
                diagnostics.push(Diagnostic::error(
                    "E0305",
                    "For-loop lower bound must be a compile-time integer constant",
                    start.span(),
                    "non-constant loop bound",
                ));
                return None;
            };
            let Some(end_i) = eval_static_int(&he, scope.consts) else {
                diagnostics.push(Diagnostic::error(
                    "E0305",
                    "For-loop upper bound must be a compile-time integer constant",
                    end.span(),
                    "non-constant loop bound",
                ));
                return None;
            };
            if end_i < start_i || (end_i - start_i) > 4096 {
                diagnostics.push(Diagnostic::error(
                    "E0305",
                    format!("For-loop range {start_i}..{end_i} exceeds deterministic static bound [0..4096]"),
                    *span,
                    "invalid static loop range",
                ));
                return None;
            }
            let prev_local = scope
                .locals
                .insert(var.clone(), ValueType::Scalar(DimVector::DIMENSIONLESS));
            scope.symbols.push(SymbolInfo {
                name: var.clone(),
                kind: SymbolKind::Local,
                detail: format!("{var}: Scalar<1> [8B f64] (loop index)"),
                span: *var_span,
            });
            let mut body_stmts = Vec::new();
            for s in body {
                if let Some(st) = check_stmt(s, scope, diagnostics) {
                    body_stmts.push(st);
                }
            }
            if let Some(old) = prev_local {
                scope.locals.insert(var.clone(), old);
            } else {
                scope.locals.remove(var);
            }
            Some(HirStmt::For {
                var: var.clone(),
                start: start_i,
                end: end_i,
                body: body_stmts,
                span: *span,
            })
        }
        Stmt::Return { value, span } => {
            let val = check_expr(value, scope, diagnostics)?;
            if !scope.in_fn {
                diagnostics.push(Diagnostic::error(
                    "E0203",
                    "`return` is only valid inside pure `fn` declarations",
                    *span,
                    "unexpected return",
                ));
                return None;
            }
            if let Some(expected) = &scope.expected_return
                && !expected.is_compatible(val.ty())
            {
                let code = mismatch_code(expected, val.ty());
                diagnostics.push(Diagnostic::error(
                    code,
                    format!(
                        "Return type mismatch: expected {expected}, found {}",
                        val.ty()
                    ),
                    value.span(),
                    format!("deduced {}", val.ty()),
                ));
            }
            scope.saw_return = true;
            Some(HirStmt::Return {
                value: val,
                span: *span,
            })
        }
        Stmt::Expr { expr, span } => {
            let e = check_expr(expr, scope, diagnostics)?;
            Some(HirStmt::Expr {
                expr: e,
                span: *span,
            })
        }
    }
}

fn check_expr(
    expr: &Expr,
    scope: &mut Scope<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<HirExpr> {
    match expr {
        Expr::Quantity { value, unit, span } => match unit {
            Some(u) => {
                let (dim, scale) = resolve_unit(u, scope.unit_aliases, diagnostics)?;
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
                return Some(HirExpr::LocalRef {
                    name: name.clone(),
                    ty: ty.clone(),
                    span: *span,
                });
            }
            if let Some(ty) = scope.params.get(name) {
                return Some(HirExpr::ParamRef {
                    name: name.clone(),
                    ty: ty.clone(),
                    span: *span,
                });
            }
            if let Some(c) = scope.consts.get(name) {
                return Some(HirExpr::ConstRef {
                    name: name.clone(),
                    ty: c.ty.clone(),
                    span: *span,
                });
            }
            diagnostics.push(Diagnostic::error(
                "E0201",
                format!("Undefined symbol '{name}'"),
                *span,
                "not found in scope",
            ));
            None
        }
        Expr::MemberAccess {
            object,
            field,
            field_span,
            span,
        } => {
            let obj = check_expr(object, scope, diagnostics)?;
            match obj.ty().clone() {
                ValueType::StateRef {
                    state_name,
                    is_next,
                } => {
                    let st = scope.states.get(&state_name)?;
                    let Some(sf) = st.fields.iter().find(|f| f.name == *field) else {
                        diagnostics.push(Diagnostic::error(
                            "E0201",
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
                                "E0201",
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
                ValueType::Quat => {
                    let comp = match field.as_str() {
                        "w" => 'w',
                        "x" => 'x',
                        "y" => 'y',
                        "z" => 'z',
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0201",
                                format!("Unknown quaternion component '.{field}'"),
                                *field_span,
                                "expected w, x, y, or z",
                            ));
                            return None;
                        }
                    };
                    Some(HirExpr::VecComponent {
                        vec: Box::new(obj),
                        component: comp,
                        ty: ValueType::Scalar(DimVector::DIMENSIONLESS),
                        span: *span,
                    })
                }
                other => {
                    diagnostics.push(Diagnostic::error(
                        "E0203",
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
                    "E0203",
                    "Vec3 components must be scalar quantities",
                    *span,
                    "non-scalar Vec3 component",
                ));
                return None;
            };
            let is_zero =
                |e: &Expr| matches!(e, Expr::Quantity { value, unit: None, .. } if *value == 0.0);
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
                        "E0202",
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
        Expr::QuatLit { w, x, y, z, span } => {
            let hw = check_expr(w, scope, diagnostics)?;
            let hx = check_expr(x, scope, diagnostics)?;
            let hy = check_expr(y, scope, diagnostics)?;
            let hz = check_expr(z, scope, diagnostics)?;
            for (comp, ast_e) in [
                (&hw, w.as_ref()),
                (&hx, x.as_ref()),
                (&hy, y.as_ref()),
                (&hz, z.as_ref()),
            ] {
                if *comp.ty() != ValueType::Scalar(DimVector::DIMENSIONLESS) {
                    let code = if matches!(comp.ty(), ValueType::Scalar(_)) {
                        "E0202"
                    } else {
                        "E0203"
                    };
                    diagnostics.push(Diagnostic::error(
                        code,
                        format!("Quat components must be dimensionless, found {}", comp.ty()),
                        ast_e.span(),
                        "expected dimensionless scalar",
                    ));
                    return None;
                }
            }
            Some(HirExpr::Quat {
                w: Box::new(hw),
                x: Box::new(hx),
                y: Box::new(hy),
                z: Box::new(hz),
                ty: ValueType::Quat,
                span: *span,
            })
        }
        Expr::Mat3Lit { elems, span } => {
            let mut hir_elems = Vec::with_capacity(9);
            let mut target_dim: Option<DimVector> = None;
            for ast_e in elems.iter() {
                let he = check_expr(ast_e, scope, diagnostics)?;
                let ValueType::Scalar(d) = *he.ty() else {
                    diagnostics.push(Diagnostic::error(
                        "E0203",
                        "Mat3 elements must be scalar quantities",
                        ast_e.span(),
                        "expected scalar",
                    ));
                    return None;
                };
                let is_zero = matches!(
                    ast_e,
                    Expr::Quantity {
                        value,
                        unit: None,
                        ..
                    } if *value == 0.0
                );
                if !is_zero {
                    if let Some(td) = target_dim {
                        if td != d {
                            diagnostics.push(Diagnostic::error(
                                "E0202",
                                format!("Inconsistent physical units across Mat3 elements: expected {td}, found {d}"),
                                ast_e.span(),
                                format!("has unit {d}"),
                            ));
                            return None;
                        }
                    } else {
                        target_dim = Some(d);
                    }
                }
                hir_elems.push(he);
            }
            let arr: [HirExpr; 9] = hir_elems.try_into().ok()?;
            Some(HirExpr::Mat3 {
                elems: Box::new(arr),
                ty: ValueType::Mat3(target_dim.unwrap_or(DimVector::DIMENSIONLESS)),
                span: *span,
            })
        }
        Expr::IfExpr {
            cond,
            then_expr,
            else_expr,
            span,
        } => {
            let hc = check_expr(cond, scope, diagnostics)?;
            if *hc.ty() != ValueType::Bool {
                diagnostics.push(Diagnostic::error(
                    "E0203",
                    "Conditional expression condition must be a boolean comparison",
                    cond.span(),
                    format!("found {}", hc.ty()),
                ));
            }
            let ht = check_expr(then_expr, scope, diagnostics)?;
            let he = check_expr(else_expr, scope, diagnostics)?;
            if !ht.ty().is_compatible(he.ty()) {
                let code = mismatch_code(ht.ty(), he.ty());
                diagnostics.push(
                    Diagnostic::error(
                        code,
                        format!(
                            "Incompatible branch types in `if/else` expression: {} vs {}",
                            ht.ty(),
                            he.ty()
                        ),
                        *span,
                        "branch types must match",
                    )
                    .with_secondary(then_expr.span(), format!("then branch has {}", ht.ty()))
                    .with_secondary(else_expr.span(), format!("else branch has {}", he.ty())),
                );
                return None;
            }
            let ty = ht.ty().clone();
            Some(HirExpr::IfExpr {
                cond: Box::new(hc),
                then_expr: Box::new(ht),
                else_expr: Box::new(he),
                ty,
                span: *span,
            })
        }
        Expr::Call {
            callee,
            callee_span,
            args,
            span,
        } => {
            if (scope.in_step || scope.in_fn) && is_forbidden_alloc(callee) {
                diagnostics.push(Diagnostic::error(
                    "E0303",
                    format!(
                        "Heap allocation '{callee}(...)' is forbidden inside deterministic kernel"
                    ),
                    *callee_span,
                    "heap allocation prohibited in hot loop",
                ));
                return None;
            }
            if (scope.in_step && scope.system_name.as_deref() == Some(callee.as_str()))
                || (scope.in_fn && scope.current_fn.as_deref() == Some(callee.as_str()))
            {
                diagnostics.push(Diagnostic::error(
                    "E0304",
                    format!("Recursion into '{callee}' is forbidden"),
                    *callee_span,
                    "recursion prohibited",
                ));
                return None;
            }

            let mut h_args = Vec::with_capacity(args.len());
            for a in args {
                h_args.push(check_expr(a, scope, diagnostics)?);
            }

            if let Some(sig) = scope.fn_sigs.get(callee.as_str()) {
                if sig.params.len() != h_args.len() {
                    diagnostics.push(Diagnostic::error(
                        "E0203",
                        format!(
                            "Function '{callee}' expects {} argument(s), found {}",
                            sig.params.len(),
                            h_args.len()
                        ),
                        *callee_span,
                        "argument count mismatch",
                    ));
                    return None;
                }
                for (idx, (expected, actual)) in sig.params.iter().zip(h_args.iter()).enumerate() {
                    if !expected.is_compatible(actual.ty()) {
                        let code = mismatch_code(expected, actual.ty());
                        diagnostics.push(Diagnostic::error(
                            code,
                            format!(
                                "Argument {} to '{callee}' mismatch: expected {expected}, found {}",
                                idx + 1,
                                actual.ty()
                            ),
                            args[idx].span(),
                            format!("deduced {}", actual.ty()),
                        ));
                        return None;
                    }
                }
                return Some(HirExpr::Call {
                    callee: callee.clone(),
                    args: h_args,
                    ty: sig.ret.clone(),
                    span: *span,
                });
            }

            let ret_ty = match callee.as_str() {
                "dot" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => ValueType::Scalar(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`dot` requires two Vec3 arguments",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "cross" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => ValueType::Vec3(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`cross` requires two Vec3 arguments",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "norm" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Vec3(a) => ValueType::Scalar(*a),
                    ValueType::Quat => ValueType::Scalar(DimVector::DIMENSIONLESS),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`norm` requires one Vec3 or Quat argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "normalize" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Vec3(_) => ValueType::Vec3(DimVector::DIMENSIONLESS),
                    ValueType::Quat => ValueType::Quat,
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`normalize` requires one Vec3 or Quat argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "quat_conj" | "quat_norm" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Quat => ValueType::Quat,
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            format!("`{callee}` requires a Quat argument"),
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "quat_rotate" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Quat, ValueType::Vec3(u)) => ValueType::Vec3(*u),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`quat_rotate` requires (Quat, Vec3<U>)",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "quat_integrate" if h_args.len() == 3 => {
                    match (h_args[0].ty(), h_args[1].ty(), h_args[2].ty()) {
                        (
                            ValueType::Quat,
                            ValueType::Vec3(omega_dim),
                            ValueType::Scalar(dt_dim),
                        ) if *omega_dim == DimVector::ANGULAR_VEL
                            && *dt_dim == DimVector::SECOND =>
                        {
                            ValueType::Quat
                        }
                        (ValueType::Quat, ValueType::Vec3(_), ValueType::Scalar(_)) => {
                            diagnostics.push(Diagnostic::error(
                                "E0202",
                                "`quat_integrate` requires omega: Vec3<rad/s> and dt: s",
                                *span,
                                "unit mismatch in quat_integrate",
                            ));
                            return None;
                        }
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0203",
                                "`quat_integrate` requires (Quat, Vec3<rad/s>, s)",
                                *span,
                                "type mismatch in quat_integrate",
                            ));
                            return None;
                        }
                    }
                }
                "mat3_diag" if h_args.len() == 3 => {
                    match (h_args[0].ty(), h_args[1].ty(), h_args[2].ty()) {
                        (ValueType::Scalar(a), ValueType::Scalar(b), ValueType::Scalar(c))
                            if a == b && b == c =>
                        {
                            ValueType::Mat3(*a)
                        }
                        (ValueType::Scalar(_), ValueType::Scalar(_), ValueType::Scalar(_)) => {
                            diagnostics.push(Diagnostic::error(
                                "E0202",
                                "`mat3_diag` requires three Scalar<U> arguments with identical units",
                                *span,
                                "incompatible diagonal units",
                            ));
                            return None;
                        }
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0203",
                                "`mat3_diag` requires three Scalar<U> arguments",
                                *span,
                                "invalid args",
                            ));
                            return None;
                        }
                    }
                }
                "mat3_from_quat" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Quat => ValueType::Mat3(DimVector::DIMENSIONLESS),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`mat3_from_quat` requires a Quat argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "mat3_transpose" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Mat3(u) => ValueType::Mat3(*u),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`mat3_transpose` requires a Mat3<U> argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "mat3_inv" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Mat3(u) => ValueType::Mat3(u.inv()),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`mat3_inv` requires a Mat3<U> argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "sqrt" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Scalar(a) => match a.sqrt() {
                        Some(d) => ValueType::Scalar(d),
                        None => {
                            diagnostics.push(Diagnostic::error(
                                "E0210",
                                format!("Non-integer resulting dimension: cannot take sqrt of odd unit {a}"),
                                *span,
                                "non-integer resulting dimension",
                            ));
                            return None;
                        }
                    },
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`sqrt` requires one Scalar argument",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "cbrt" if h_args.len() == 1 => {
                    match h_args[0].ty() {
                        ValueType::Scalar(a) => match a.cbrt() {
                            Some(d) => ValueType::Scalar(d),
                            None => {
                                diagnostics.push(Diagnostic::error(
                                "E0210",
                                format!("Non-integer resulting dimension: cannot take cbrt of unit {a}"),
                                *span,
                                "non-integer resulting dimension",
                            ));
                                return None;
                            }
                        },
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0203",
                                "`cbrt` requires one Scalar argument",
                                *span,
                                "invalid args",
                            ));
                            return None;
                        }
                    }
                }
                "sin" | "cos" | "tan" if h_args.len() == 1 => match h_args[0].ty() {
                    ValueType::Scalar(d) if d.si == [0; 7] && (d.rad == 0 || d.rad == 1) => {
                        ValueType::Scalar(DimVector::DIMENSIONLESS)
                    }
                    ValueType::Scalar(_) => {
                        diagnostics.push(Diagnostic::error(
                            "E0202",
                            format!("`{callee}` requires rad or dimensionless angle"),
                            *span,
                            "invalid unit for trig function",
                        ));
                        return None;
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            format!("`{callee}` requires a scalar angle argument"),
                            *span,
                            "invalid type for trig function",
                        ));
                        return None;
                    }
                },
                "atan2" if h_args.len() == 2 => match (h_args[0].ty(), h_args[1].ty()) {
                    (ValueType::Scalar(a), ValueType::Scalar(b)) if a == b => {
                        ValueType::Scalar(DimVector::RADIAN)
                    }
                    (ValueType::Scalar(_), ValueType::Scalar(_)) => {
                        diagnostics.push(Diagnostic::error(
                            "E0202",
                            "`atan2(y, x)` requires two scalars of identical dimension",
                            *span,
                            "unit mismatch in atan2",
                        ));
                        return None;
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`atan2(y, x)` requires two Scalar arguments",
                            *span,
                            "invalid args",
                        ));
                        return None;
                    }
                },
                "abs" if h_args.len() == 1 => h_args[0].ty().clone(),
                "min" | "max"
                    if h_args.len() == 2 && h_args[0].ty().is_compatible(h_args[1].ty()) =>
                {
                    h_args[0].ty().clone()
                }
                "clamp"
                    if h_args.len() == 3
                        && h_args[0].ty().is_compatible(h_args[1].ty())
                        && h_args[0].ty().is_compatible(h_args[2].ty()) =>
                {
                    h_args[0].ty().clone()
                }
                "select"
                    if h_args.len() == 3
                        && *h_args[0].ty() == ValueType::Bool
                        && h_args[1].ty().is_compatible(h_args[2].ty()) =>
                {
                    h_args[1].ty().clone()
                }
                "assert_approx" if h_args.len() == 3 => {
                    if !scope.in_test {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "`assert_approx` can only be used inside `test` blocks",
                            *callee_span,
                            "only valid inside test blocks",
                        ));
                        return None;
                    }
                    if !h_args[0].ty().is_compatible(h_args[1].ty())
                        || !h_args[0].ty().is_compatible(h_args[2].ty())
                    {
                        let code = mismatch_code(h_args[0].ty(), h_args[1].ty());
                        diagnostics.push(Diagnostic::error(
                            code,
                            format!(
                                "Unit mismatch in `assert_approx`: {}, {}, {}",
                                h_args[0].ty(),
                                h_args[1].ty(),
                                h_args[2].ty()
                            ),
                            *span,
                            "all assert_approx arguments must have identical dimensions",
                        ));
                        return None;
                    }
                    ValueType::Scalar(DimVector::DIMENSIONLESS)
                }
                _ => {
                    diagnostics.push(Diagnostic::error(
                        "E0203",
                        format!("Invalid call to '{callee}'"),
                        *callee_span,
                        "unknown or ill-typed intrinsic",
                    ));
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
        Expr::Neg(inner, span) => {
            let hi = check_expr(inner, scope, diagnostics)?;
            let ty = hi.ty().clone();
            Some(HirExpr::UnaryNeg {
                operand: Box::new(hi),
                ty,
                span: *span,
            })
        }
        Expr::Binary {
            op,
            left,
            right,
            span,
        } => {
            let hl = check_expr(left, scope, diagnostics)?;
            let hr = check_expr(right, scope, diagnostics)?;
            let lt = hl.ty().clone();
            let rt = hr.ty().clone();

            let ty = match op {
                BinOp::Add | BinOp::Sub => match (&lt, &rt) {
                    (ValueType::Scalar(a), ValueType::Scalar(b)) if a == b => lt,
                    (ValueType::Vec3(a), ValueType::Vec3(b)) if a == b => lt,
                    (ValueType::Mat3(a), ValueType::Mat3(b)) if a == b => lt,
                    _ => {
                        let code = mismatch_code(&lt, &rt);
                        diagnostics.push(
                            Diagnostic::error(
                                code,
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
                    (ValueType::Quat, ValueType::Quat) => ValueType::Quat,
                    (ValueType::Mat3(a), ValueType::Vec3(b)) => ValueType::Vec3(a.mul(*b)),
                    (ValueType::Mat3(a), ValueType::Mat3(b)) => ValueType::Mat3(a.mul(*b)),
                    (ValueType::Mat3(a), ValueType::Scalar(b))
                    | (ValueType::Scalar(b), ValueType::Mat3(a)) => ValueType::Mat3(a.mul(*b)),
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            format!("Unsupported `*` between {lt} and {rt}"),
                            *span,
                            "use dot() or cross() for Vec3 * Vec3",
                        ));
                        return None;
                    }
                },
                BinOp::Div => {
                    if matches!(right.as_ref(), Expr::Quantity { value, .. } if *value == 0.0) {
                        diagnostics.push(
                            Diagnostic::error(
                                "E0202",
                                "Division by zero physical quantity",
                                right.span(),
                                "divisor is zero",
                            )
                            .with_help(
                                "Ensure divisors in deterministic physics kernels are non-zero.",
                            ),
                        );
                        return None;
                    }
                    match (&lt, &rt) {
                        (ValueType::Scalar(a), ValueType::Scalar(b)) => {
                            ValueType::Scalar(a.div(*b))
                        }
                        (ValueType::Vec3(a), ValueType::Scalar(b)) => ValueType::Vec3(a.div(*b)),
                        (ValueType::Mat3(a), ValueType::Scalar(b)) => ValueType::Mat3(a.div(*b)),
                        _ => {
                            diagnostics.push(Diagnostic::error(
                                "E0203",
                                format!("Unsupported `/` between {lt} and {rt}"),
                                *span,
                                "invalid division",
                            ));
                            return None;
                        }
                    }
                }
                BinOp::Pow => match (&lt, right.as_ref()) {
                    (
                        ValueType::Scalar(a),
                        Expr::Quantity {
                            value, unit: None, ..
                        },
                    ) if value.fract() == 0.0 && (-127.0..=127.0).contains(value) => {
                        ValueType::Scalar(a.pow(*value as i8))
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "E0203",
                            "Power operator requires scalar base and integer exponent",
                            *span,
                            "invalid exponent",
                        ));
                        return None;
                    }
                },
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    if !lt.is_compatible(&rt) {
                        let code = mismatch_code(&lt, &rt);
                        diagnostics.push(Diagnostic::error(
                            code,
                            format!("Cannot compare incompatible types {lt} and {rt}"),
                            *span,
                            "incompatible comparison",
                        ));
                        return None;
                    }
                    ValueType::Bool
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;
    use aerotick_parser::parse;
    use proptest::prelude::*;

    fn compile_check(src: &str) -> (HirProgram, Vec<Diagnostic>) {
        let (tokens, mut diags) = lex(src);
        let (ast, p_errs) = parse(&tokens, src.len());
        diags.extend(p_errs);
        if let Some(p) = ast {
            let (hir, s_errs) = analyze(&p);
            diags.extend(s_errs);
            return (hir, diags);
        }
        (
            HirProgram {
                unit_aliases: vec![],
                consts: vec![],
                states: vec![],
                functions: vec![],
                systems: vec![],
                tests: vec![],
                top_bindings: vec![],
                symbols: vec![],
            },
            diags,
        )
    }

    #[test]
    fn checks_6dof_quat_mat3_pure_fn_and_units() {
        let src = r#"
            unit Thrust = N;
            unit Inertia = kg*m^2;
            fn compute_torque(inertia: Mat3<Inertia>, alpha: Vec3<rad/s^2>) -> Vec3<N*m> {
                let raw: Vec3<kg*m^2*rad/s^2> = inertia * alpha;
                return Vec3(raw.x / 1.0 rad, raw.y / 1.0 rad, raw.z / 1.0 rad);
            }
            let alt: m = 1.0 ft + 12.0 in;
            let spd: m/s = 100.0 kn;
            let spin: rad/s = 600.0 rpm;
            let p: Pa = 1.0 atm;
            let root: m/s = sqrt(100.0 m^2/s^2);
            let side: m = cbrt(27.0 m^3);
            let q: Quat = quat_integrate(Quat(1.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0 rad/s), 0.1 s);
            let v_rot: Vec3<m> = quat_rotate(q, Vec3(1.0 m, 0.0, 0.0));
        "#;
        let (_hir, diags) = compile_check(src);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn emits_exact_section_3_3_error_and_warning_codes() {
        let src = r#"
            let bad_unit: foo = 1.0 m;
            let bad_dim: m = 1.0 m + 2.0 kg;
            let bad_type: Vec3<m> = 1.0 m;
            let bad_root = sqrt(10.0 m);
            let bad_cube = cbrt(10.0 m^2);

            fn ping(x: m) -> m { return pong(x); }
            fn pong(x: m) -> m { return ping(x); }

            state S { x: m; y: m; }
            system Bad(dt: s) {
                step(prev: S) -> next: S {
                    let early = next.x;
                    prev.x = prev.x + 1.0 m;
                    let heap = alloc(128);
                    while dt > 0.0 s {
                        let z = 1.0;
                    }
                    next.x = prev.x;
                }
            }
        "#;
        let (_hir, diags) = compile_check(src);
        let codes: Vec<&str> = diags.iter().map(|d| d.code).collect();
        for expected in [
            "E0201", "E0202", "E0203", "E0210", "E0301", "E0302", "E0303", "E0304", "E0305",
            "W0301",
        ] {
            assert!(
                codes.contains(&expected),
                "missing expected code {expected} in {codes:?}"
            );
        }
    }

    fn arb_dim_vector() -> impl Strategy<Value = DimVector> {
        (prop::array::uniform7(-12i8..=12i8), -12i8..=12i8)
            .prop_map(|(si, rad)| DimVector { si, rad })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        #[test]
        fn si_dim_vector_forms_abelian_group(
            a in arb_dim_vector(),
            b in arb_dim_vector(),
            c in arb_dim_vector(),
        ) {
            prop_assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
            prop_assert_eq!(a.mul(b), b.mul(a));
            prop_assert_eq!(a.mul(DimVector::DIMENSIONLESS), a);
            prop_assert_eq!(DimVector::DIMENSIONLESS.mul(a), a);
            prop_assert_eq!(a.div(a), DimVector::DIMENSIONLESS);
            prop_assert_eq!(a.mul(a.inv()), DimVector::DIMENSIONLESS);
        }

        #[test]
        fn si_sqrt_and_cbrt_roundtrip(a in arb_dim_vector()) {
            let sq = a.pow(2);
            prop_assert_eq!(sq.sqrt(), Some(a));
            let cb = a.pow(3);
            prop_assert_eq!(cb.cbrt(), Some(a));
        }
    }
}
