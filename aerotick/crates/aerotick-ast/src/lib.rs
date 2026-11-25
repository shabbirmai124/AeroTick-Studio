//! Phase 2: Untyped Abstract Syntax Tree definitions with attached `Span` metadata
//! covering `const`, `state`, `system`/`step`, `Vec3<Unit>`, statements, and expressions.

use aerotick_span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Const(ConstDecl),
    State(StateDecl),
    System(SystemDecl),
    TopLet {
        name: String,
        name_span: Span,
        type_ann: Option<TypeExpr>,
        value: Expr,
        span: Span,
    },
    TopExpr {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDecl {
    pub name: String,
    pub name_span: Span,
    pub type_ann: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateField {
    pub name: String,
    pub name_span: Span,
    pub type_expr: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateDecl {
    pub name: String,
    pub name_span: Span,
    pub fields: Vec<StateField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub name_span: Span,
    pub type_expr: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepBlock {
    pub prev_name: String,
    pub prev_type: String,
    pub prev_type_span: Span,
    pub next_name: String,
    pub next_type: String,
    pub next_type_span: Span,
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemDecl {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub step: StepBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    ScalarUnit(UnitExpr, Span),
    Vec3(UnitExpr, Span),
    Quat(Span),
    Mat4(Span),
}

impl TypeExpr {
    pub fn span(&self) -> Span {
        match self {
            Self::ScalarUnit(_, s) | Self::Vec3(_, s) | Self::Quat(s) | Self::Mat4(s) => *s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnitExpr {
    Symbol(String, Span),
    One(Span),
    Mul(Box<UnitExpr>, Box<UnitExpr>, Span),
    Div(Box<UnitExpr>, Box<UnitExpr>, Span),
    Pow(Box<UnitExpr>, i8, Span),
}

impl UnitExpr {
    pub fn span(&self) -> Span {
        match self {
            Self::Symbol(_, s)
            | Self::One(s)
            | Self::Mul(_, _, s)
            | Self::Div(_, _, s)
            | Self::Pow(_, _, s) => *s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        name_span: Span,
        type_ann: Option<TypeExpr>,
        value: Expr,
        span: Span,
    },
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    If {
        cond: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
    ForbiddenLoop {
        keyword: String,
        span: Span,
    },
    Expr {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Quantity {
        value: f64,
        unit: Option<UnitExpr>,
        span: Span,
    },
    Ident(String, Span),
    MemberAccess {
        object: Box<Expr>,
        field: String,
        field_span: Span,
        span: Span,
    },
    Vec3Lit {
        x: Box<Expr>,
        y: Box<Expr>,
        z: Box<Expr>,
        span: Span,
    },
    Call {
        callee: String,
        callee_span: Span,
        args: Vec<Expr>,
        span: Span,
    },
    Neg(Box<Expr>, Span),
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Self::Quantity { span, .. }
            | Self::Ident(_, span)
            | Self::MemberAccess { span, .. }
            | Self::Vec3Lit { span, .. }
            | Self::Call { span, .. }
            | Self::Neg(_, span)
            | Self::Binary { span, .. } => *span,
        }
    }
}
