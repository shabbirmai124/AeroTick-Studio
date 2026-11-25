#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 2: Parser with `chumsky` error recovery and multi-error diagnostics
//! supporting AeroTick Ultra v2.0 (`unit`, `const`, `state`, `fn`, `system`/`step`, `test`,
//! `Vec3<U>`, `Quat`, `Mat3<U>`, `if`/`else` expressions, bounded `for i in 0..N`, `return`).

use aerotick_ast::{
    BinOp, ConstDecl, Expr, FnDecl, Item, Param, Program, StateDecl, StateField, StepBlock, Stmt,
    SystemDecl, TestDecl, TypeExpr, UnitAliasDecl, UnitExpr,
};
use aerotick_lexer::{SpannedToken, TokenKind};
use aerotick_span::{Diagnostic, Span};
use chumsky::input::{Stream, ValueInput};
use chumsky::prelude::*;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PTok {
    Tok(TokenKind),
    TightStar,
    TightSlash,
}

impl fmt::Display for PTok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tok(k) => write!(f, "{k}"),
            Self::TightStar => write!(f, "*"),
            Self::TightSlash => write!(f, "/"),
        }
    }
}

fn to_span(s: SimpleSpan) -> Span {
    Span::new(s.start, s.end)
}

fn unit_parser<'a, I>(
    allow_one_and_spaced: bool,
) -> impl Parser<'a, I, UnitExpr, extra::Err<Rich<'a, PTok, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = PTok, Span = SimpleSpan>,
{
    recursive(move |unit_expr| {
        let sym = select! {
            PTok::Tok(TokenKind::Ident(name)) => name,
            PTok::Tok(TokenKind::KwIn) => "in".to_string(),
        }
        .map_with(|name, e| UnitExpr::Symbol(name, to_span(e.span())));

        let atom = if allow_one_and_spaced {
            let one = select! {
                PTok::Tok(TokenKind::Number(n)) if n == "1" => (),
            }
            .map_with(|(), e| UnitExpr::One(to_span(e.span())));

            choice((
                sym,
                one,
                unit_expr.delimited_by(
                    just(PTok::Tok(TokenKind::LParen)),
                    just(PTok::Tok(TokenKind::RParen)),
                ),
            ))
            .boxed()
        } else {
            sym.boxed()
        };

        let pow = atom
            .then(
                just(PTok::Tok(TokenKind::Caret))
                    .ignore_then(just(PTok::Tok(TokenKind::Minus)).or_not())
                    .then(select! { PTok::Tok(TokenKind::Number(n)) => n }.try_map(
                        |num_str, span| {
                            num_str.parse::<i8>().map_err(|_| {
                                Rich::custom(
                                    span,
                                    format!("unit exponent '{num_str}' must be an integer"),
                                )
                            })
                        },
                    ))
                    .or_not(),
            )
            .map_with(|(base, exp_opt), e| match exp_opt {
                Some((neg, mag)) => {
                    let exp = if neg.is_some() {
                        mag.saturating_neg()
                    } else {
                        mag
                    };
                    UnitExpr::Pow(Box::new(base), exp, to_span(e.span()))
                }
                None => base,
            });

        let mul_op = if allow_one_and_spaced {
            choice((
                just(PTok::TightStar).to(true),
                just(PTok::Tok(TokenKind::Star)).to(true),
                just(PTok::TightSlash).to(false),
                just(PTok::Tok(TokenKind::Slash)).to(false),
            ))
            .boxed()
        } else {
            choice((
                just(PTok::TightStar).to(true),
                just(PTok::TightSlash).to(false),
            ))
            .boxed()
        };

        pow.clone()
            .foldl(mul_op.then(pow).repeated(), |lhs, (is_mul, rhs)| {
                let sp = lhs.span().merge(rhs.span());
                if is_mul {
                    UnitExpr::Mul(Box::new(lhs), Box::new(rhs), sp)
                } else {
                    UnitExpr::Div(Box::new(lhs), Box::new(rhs), sp)
                }
            })
    })
}

fn type_parser<'a, I>()
-> impl Parser<'a, I, TypeExpr, extra::Err<Rich<'a, PTok, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = PTok, Span = SimpleSpan>,
{
    let ann_unit = unit_parser(true);
    let vec3_ty = just(PTok::Tok(TokenKind::KwVec3))
        .ignore_then(ann_unit.clone().delimited_by(
            just(PTok::Tok(TokenKind::Lt)),
            just(PTok::Tok(TokenKind::Gt)),
        ))
        .map_with(|u, e| TypeExpr::Vec3(u, to_span(e.span())));

    let mat3_ty = just(PTok::Tok(TokenKind::KwMat3))
        .ignore_then(ann_unit.clone().delimited_by(
            just(PTok::Tok(TokenKind::Lt)),
            just(PTok::Tok(TokenKind::Gt)),
        ))
        .map_with(|u, e| TypeExpr::Mat3(u, to_span(e.span())));

    let quat_ty =
        just(PTok::Tok(TokenKind::KwQuat)).map_with(|_, e| TypeExpr::Quat(to_span(e.span())));
    let mat4_ty =
        just(PTok::Tok(TokenKind::KwMat4)).map_with(|_, e| TypeExpr::Mat4(to_span(e.span())));
    let scalar_ty = ann_unit.map_with(|u, e| TypeExpr::ScalarUnit(u, to_span(e.span())));

    choice((vec3_ty, mat3_ty, quat_ty, mat4_ty, scalar_ty))
}

fn expr_parser<'a, I>() -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, PTok, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = PTok, Span = SimpleSpan>,
{
    let literal_unit_suffix = unit_parser(false);

    recursive(|expr| {
        let quantity = select! {
            PTok::Tok(TokenKind::Number(n)) => n,
        }
        .try_map(|num_str, span| {
            num_str
                .parse::<f64>()
                .map_err(|_| Rich::custom(span, format!("invalid numeric literal '{num_str}'")))
        })
        .then(literal_unit_suffix.clone().or_not())
        .map_with(|(value, unit), e| Expr::Quantity {
            value,
            unit,
            span: to_span(e.span()),
        });

        let vec3_lit = just(PTok::Tok(TokenKind::KwVec3))
            .ignore_then(
                expr.clone()
                    .then_ignore(just(PTok::Tok(TokenKind::Comma)))
                    .then(expr.clone())
                    .then_ignore(just(PTok::Tok(TokenKind::Comma)))
                    .then(expr.clone())
                    .delimited_by(
                        just(PTok::Tok(TokenKind::LParen)),
                        just(PTok::Tok(TokenKind::RParen)),
                    ),
            )
            .map_with(|((x, y), z), e| Expr::Vec3Lit {
                x: Box::new(x),
                y: Box::new(y),
                z: Box::new(z),
                span: to_span(e.span()),
            });

        let quat_lit = just(PTok::Tok(TokenKind::KwQuat))
            .ignore_then(
                expr.clone()
                    .then_ignore(just(PTok::Tok(TokenKind::Comma)))
                    .then(expr.clone())
                    .then_ignore(just(PTok::Tok(TokenKind::Comma)))
                    .then(expr.clone())
                    .then_ignore(just(PTok::Tok(TokenKind::Comma)))
                    .then(expr.clone())
                    .delimited_by(
                        just(PTok::Tok(TokenKind::LParen)),
                        just(PTok::Tok(TokenKind::RParen)),
                    ),
            )
            .map_with(|(((w, x), y), z), e| Expr::QuatLit {
                w: Box::new(w),
                x: Box::new(x),
                y: Box::new(y),
                z: Box::new(z),
                span: to_span(e.span()),
            });

        let mat3_lit = just(PTok::Tok(TokenKind::KwMat3))
            .ignore_then(
                expr.clone()
                    .separated_by(just(PTok::Tok(TokenKind::Comma)))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(
                        just(PTok::Tok(TokenKind::LParen)),
                        just(PTok::Tok(TokenKind::RParen)),
                    )
                    .try_map(|elems, span| {
                        let arr: [Expr; 9] = elems.try_into().map_err(|v: Vec<Expr>| {
                            Rich::custom(
                                span,
                                format!("Mat3(...) expects 9 elements, found {}", v.len()),
                            )
                        })?;
                        Ok(arr)
                    }),
            )
            .map_with(|elems, e| Expr::Mat3Lit {
                elems: Box::new(elems),
                span: to_span(e.span()),
            });

        let if_expr = just(PTok::Tok(TokenKind::KwIf))
            .ignore_then(expr.clone())
            .then(expr.clone().delimited_by(
                just(PTok::Tok(TokenKind::LBrace)),
                just(PTok::Tok(TokenKind::RBrace)),
            ))
            .then_ignore(just(PTok::Tok(TokenKind::KwElse)))
            .then(expr.clone().delimited_by(
                just(PTok::Tok(TokenKind::LBrace)),
                just(PTok::Tok(TokenKind::RBrace)),
            ))
            .map_with(|((cond, then_expr), else_expr), e| Expr::IfExpr {
                cond: Box::new(cond),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                span: to_span(e.span()),
            });

        let call = select! {
            PTok::Tok(TokenKind::Ident(name)) => name,
        }
        .map_with(|name, e| (name, to_span(e.span())))
        .then(
            expr.clone()
                .separated_by(just(PTok::Tok(TokenKind::Comma)))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(
                    just(PTok::Tok(TokenKind::LParen)),
                    just(PTok::Tok(TokenKind::RParen)),
                ),
        )
        .map_with(|((callee, callee_span), args), e| Expr::Call {
            callee,
            callee_span,
            args,
            span: to_span(e.span()),
        });

        let ident = select! {
            PTok::Tok(TokenKind::Ident(name)) => name,
        }
        .map_with(|name, e| Expr::Ident(name, to_span(e.span())));

        let parens = expr.clone().delimited_by(
            just(PTok::Tok(TokenKind::LParen)),
            just(PTok::Tok(TokenKind::RParen)),
        );

        let atom = choice((
            vec3_lit, quat_lit, mat3_lit, if_expr, call, quantity, ident, parens,
        ));

        let postfix = atom.foldl(
            just(PTok::Tok(TokenKind::Dot))
                .ignore_then(
                    select! { PTok::Tok(TokenKind::Ident(f)) => f }
                        .map_with(|f, e| (f, to_span(e.span()))),
                )
                .repeated(),
            |object, (field, field_span)| {
                let sp = object.span().merge(field_span);
                Expr::MemberAccess {
                    object: Box::new(object),
                    field,
                    field_span,
                    span: sp,
                }
            },
        );

        let power = postfix.clone().foldl(
            just(PTok::Tok(TokenKind::Caret))
                .ignore_then(postfix)
                .repeated(),
            |lhs, rhs| {
                let sp = lhs.span().merge(rhs.span());
                Expr::Binary {
                    op: BinOp::Pow,
                    left: Box::new(lhs),
                    right: Box::new(rhs),
                    span: sp,
                }
            },
        );

        let unary = just(PTok::Tok(TokenKind::Minus))
            .repeated()
            .foldr(power, |_op, rhs| {
                let sp = rhs.span();
                Expr::Neg(Box::new(rhs), sp)
            });

        let product = unary.clone().foldl(
            choice((
                just(PTok::Tok(TokenKind::Star)).to(BinOp::Mul),
                just(PTok::TightStar).to(BinOp::Mul),
                just(PTok::Tok(TokenKind::Slash)).to(BinOp::Div),
                just(PTok::TightSlash).to(BinOp::Div),
            ))
            .then(unary)
            .repeated(),
            |lhs, (op, rhs)| {
                let sp = lhs.span().merge(rhs.span());
                Expr::Binary {
                    op,
                    left: Box::new(lhs),
                    right: Box::new(rhs),
                    span: sp,
                }
            },
        );

        let sum = product.clone().foldl(
            choice((
                just(PTok::Tok(TokenKind::Plus)).to(BinOp::Add),
                just(PTok::Tok(TokenKind::Minus)).to(BinOp::Sub),
            ))
            .then(product)
            .repeated(),
            |lhs, (op, rhs)| {
                let sp = lhs.span().merge(rhs.span());
                Expr::Binary {
                    op,
                    left: Box::new(lhs),
                    right: Box::new(rhs),
                    span: sp,
                }
            },
        );

        sum.clone().foldl(
            choice((
                just(PTok::Tok(TokenKind::EqEq)).to(BinOp::Eq),
                just(PTok::Tok(TokenKind::BangEq)).to(BinOp::Ne),
                just(PTok::Tok(TokenKind::Lt)).to(BinOp::Lt),
                just(PTok::Tok(TokenKind::LtEq)).to(BinOp::Le),
                just(PTok::Tok(TokenKind::Gt)).to(BinOp::Gt),
                just(PTok::Tok(TokenKind::GtEq)).to(BinOp::Ge),
            ))
            .then(sum)
            .repeated(),
            |lhs, (op, rhs)| {
                let sp = lhs.span().merge(rhs.span());
                Expr::Binary {
                    op,
                    left: Box::new(lhs),
                    right: Box::new(rhs),
                    span: sp,
                }
            },
        )
    })
}

fn stmt_parser<'a, I>() -> impl Parser<'a, I, Stmt, extra::Err<Rich<'a, PTok, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = PTok, Span = SimpleSpan>,
{
    let expr = expr_parser();
    let ty = type_parser();

    recursive(move |stmt| {
        let block = stmt.clone().repeated().collect::<Vec<_>>().delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        );

        let let_stmt = just(PTok::Tok(TokenKind::KwLet))
            .ignore_then(
                select! { PTok::Tok(TokenKind::Ident(name)) => name }
                    .map_with(|n, e| (n, to_span(e.span()))),
            )
            .then(
                just(PTok::Tok(TokenKind::Colon))
                    .ignore_then(ty.clone())
                    .or_not(),
            )
            .then_ignore(just(PTok::Tok(TokenKind::Eq)))
            .then(expr.clone())
            .then_ignore(just(PTok::Tok(TokenKind::Semi)))
            .map_with(|(((name, name_span), type_ann), value), e| Stmt::Let {
                name,
                name_span,
                type_ann,
                value,
                span: to_span(e.span()),
            });

        let return_stmt = just(PTok::Tok(TokenKind::KwReturn))
            .ignore_then(expr.clone())
            .then_ignore(just(PTok::Tok(TokenKind::Semi)))
            .map_with(|value, e| Stmt::Return {
                value,
                span: to_span(e.span()),
            });

        let assign_stmt = expr
            .clone()
            .then_ignore(just(PTok::Tok(TokenKind::Eq)))
            .then(expr.clone())
            .then_ignore(just(PTok::Tok(TokenKind::Semi)))
            .map_with(|(target, value), e| Stmt::Assign {
                target,
                value,
                span: to_span(e.span()),
            });

        let if_stmt = just(PTok::Tok(TokenKind::KwIf))
            .ignore_then(expr.clone())
            .then(block.clone())
            .then(
                just(PTok::Tok(TokenKind::KwElse))
                    .ignore_then(block.clone())
                    .or_not(),
            )
            .map_with(|((cond, then_branch), else_branch), e| Stmt::If {
                cond,
                then_branch,
                else_branch,
                span: to_span(e.span()),
            });

        let for_stmt = just(PTok::Tok(TokenKind::KwFor))
            .ignore_then(
                select! { PTok::Tok(TokenKind::Ident(name)) => name }
                    .map_with(|n, e| (n, to_span(e.span()))),
            )
            .then_ignore(just(PTok::Tok(TokenKind::KwIn)))
            .then(expr.clone())
            .then_ignore(just(PTok::Tok(TokenKind::DotDot)))
            .then(expr.clone())
            .then(block.clone())
            .map_with(|((((var, var_span), start), end), body), e| Stmt::For {
                var,
                var_span,
                start,
                end,
                body,
                span: to_span(e.span()),
            });

        let loop_stmt = choice((
            just(PTok::Tok(TokenKind::KwWhile)).to("while".to_string()),
            just(PTok::Tok(TokenKind::KwLoop)).to("loop".to_string()),
        ))
        .then_ignore(none_of([PTok::Tok(TokenKind::LBrace)]).repeated())
        .then_ignore(block)
        .map_with(|keyword, e| Stmt::ForbiddenLoop {
            keyword,
            span: to_span(e.span()),
        });

        let expr_stmt = expr
            .clone()
            .then_ignore(just(PTok::Tok(TokenKind::Semi)))
            .map_with(|expr, e| Stmt::Expr {
                expr,
                span: to_span(e.span()),
            });

        choice((
            let_stmt,
            return_stmt,
            if_stmt,
            for_stmt,
            loop_stmt,
            assign_stmt,
            expr_stmt,
        ))
        .recover_with(via_parser(
            none_of([PTok::Tok(TokenKind::Semi), PTok::Tok(TokenKind::RBrace)])
                .repeated()
                .then_ignore(just(PTok::Tok(TokenKind::Semi)))
                .map_with(|_, e| Stmt::Expr {
                    expr: Expr::Quantity {
                        value: 0.0,
                        unit: None,
                        span: to_span(e.span()),
                    },
                    span: to_span(e.span()),
                }),
        ))
    })
}

fn program_parser<'a, I>() -> impl Parser<'a, I, Program, extra::Err<Rich<'a, PTok, SimpleSpan>>>
where
    I: ValueInput<'a, Token = PTok, Span = SimpleSpan>,
{
    let expr = expr_parser();
    let ty = type_parser();
    let stmt = stmt_parser();
    let ann_unit = unit_parser(true);

    let ident_spanned = select! { PTok::Tok(TokenKind::Ident(name)) => name }
        .map_with(|n, e| (n, to_span(e.span())));

    let unit_alias_item = just(PTok::Tok(TokenKind::KwUnit))
        .ignore_then(ident_spanned)
        .then_ignore(just(PTok::Tok(TokenKind::Eq)))
        .then(ann_unit)
        .then_ignore(just(PTok::Tok(TokenKind::Semi)))
        .map_with(|((name, name_span), unit), e| {
            Item::UnitAlias(UnitAliasDecl {
                name,
                name_span,
                unit,
                span: to_span(e.span()),
            })
        });

    let const_item = just(PTok::Tok(TokenKind::KwConst))
        .ignore_then(ident_spanned)
        .then(
            just(PTok::Tok(TokenKind::Colon))
                .ignore_then(ty.clone())
                .or_not(),
        )
        .then_ignore(just(PTok::Tok(TokenKind::Eq)))
        .then(expr.clone())
        .then_ignore(just(PTok::Tok(TokenKind::Semi)))
        .map_with(|(((name, name_span), type_ann), value), e| {
            Item::Const(ConstDecl {
                name,
                name_span,
                type_ann,
                value,
                span: to_span(e.span()),
            })
        });

    let state_field = ident_spanned
        .then_ignore(just(PTok::Tok(TokenKind::Colon)))
        .then(ty.clone())
        .then_ignore(just(PTok::Tok(TokenKind::Semi)))
        .map_with(|((name, name_span), type_expr), e| StateField {
            name,
            name_span,
            type_expr,
            span: to_span(e.span()),
        });

    let state_item = just(PTok::Tok(TokenKind::KwState))
        .ignore_then(ident_spanned)
        .then(state_field.repeated().collect::<Vec<_>>().delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        ))
        .map_with(|((name, name_span), fields), e| {
            Item::State(StateDecl {
                name,
                name_span,
                fields,
                span: to_span(e.span()),
            })
        });

    let param = ident_spanned
        .then_ignore(just(PTok::Tok(TokenKind::Colon)))
        .then(ty.clone())
        .map_with(|((name, name_span), type_expr), e| Param {
            name,
            name_span,
            type_expr,
            span: to_span(e.span()),
        });

    let fn_item = just(PTok::Tok(TokenKind::KwFn))
        .ignore_then(ident_spanned)
        .then(
            param
                .clone()
                .separated_by(just(PTok::Tok(TokenKind::Comma)))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(
                    just(PTok::Tok(TokenKind::LParen)),
                    just(PTok::Tok(TokenKind::RParen)),
                ),
        )
        .then_ignore(just(PTok::Tok(TokenKind::Arrow)))
        .then(ty.clone())
        .then(stmt.clone().repeated().collect::<Vec<_>>().delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        ))
        .map_with(|((((name, name_span), params), return_type), body), e| {
            Item::Fn(FnDecl {
                name,
                name_span,
                params,
                return_type,
                body,
                span: to_span(e.span()),
            })
        });

    let step_block = just(PTok::Tok(TokenKind::KwStep))
        .ignore_then(
            ident_spanned
                .then_ignore(just(PTok::Tok(TokenKind::Colon)))
                .then(ident_spanned)
                .delimited_by(
                    just(PTok::Tok(TokenKind::LParen)),
                    just(PTok::Tok(TokenKind::RParen)),
                ),
        )
        .then_ignore(just(PTok::Tok(TokenKind::Arrow)))
        .then(ident_spanned)
        .then_ignore(just(PTok::Tok(TokenKind::Colon)))
        .then(ident_spanned)
        .then(stmt.clone().repeated().collect::<Vec<_>>().delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        ))
        .map_with(
            |(
                (
                    (((prev_name, _), (prev_type, prev_type_span)), (next_name, _)),
                    (next_type, next_type_span),
                ),
                stmts,
            ),
             e| {
                StepBlock {
                    prev_name,
                    prev_type,
                    prev_type_span,
                    next_name,
                    next_type,
                    next_type_span,
                    stmts,
                    span: to_span(e.span()),
                }
            },
        );

    let system_item = just(PTok::Tok(TokenKind::KwSystem))
        .ignore_then(ident_spanned)
        .then(
            param
                .separated_by(just(PTok::Tok(TokenKind::Comma)))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(
                    just(PTok::Tok(TokenKind::LParen)),
                    just(PTok::Tok(TokenKind::RParen)),
                ),
        )
        .then(step_block.delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        ))
        .map_with(|(((name, name_span), params), step), e| {
            Item::System(SystemDecl {
                name,
                name_span,
                params,
                step,
                span: to_span(e.span()),
            })
        });

    let test_item = just(PTok::Tok(TokenKind::KwTest))
        .ignore_then(
            select! { PTok::Tok(TokenKind::StringLit(s)) => s }
                .map_with(|n, e| (n, to_span(e.span()))),
        )
        .then(stmt.repeated().collect::<Vec<_>>().delimited_by(
            just(PTok::Tok(TokenKind::LBrace)),
            just(PTok::Tok(TokenKind::RBrace)),
        ))
        .map_with(|((name, name_span), body), e| {
            Item::Test(TestDecl {
                name,
                name_span,
                body,
                span: to_span(e.span()),
            })
        });

    let top_let = just(PTok::Tok(TokenKind::KwLet))
        .ignore_then(ident_spanned)
        .then(just(PTok::Tok(TokenKind::Colon)).ignore_then(ty).or_not())
        .then_ignore(just(PTok::Tok(TokenKind::Eq)))
        .then(expr.clone())
        .then_ignore(just(PTok::Tok(TokenKind::Semi)))
        .map_with(|(((name, name_span), type_ann), value), e| Item::TopLet {
            name,
            name_span,
            type_ann,
            value,
            span: to_span(e.span()),
        });

    let top_expr = expr
        .then_ignore(just(PTok::Tok(TokenKind::Semi)))
        .map_with(|expr, e| Item::TopExpr {
            expr,
            span: to_span(e.span()),
        });

    let item = choice((
        unit_alias_item,
        const_item,
        state_item,
        fn_item,
        system_item,
        test_item,
        top_let,
        top_expr,
    ))
    .recover_with(via_parser(
        none_of([PTok::Tok(TokenKind::Semi), PTok::Tok(TokenKind::RBrace)])
            .repeated()
            .then_ignore(choice((
                just(PTok::Tok(TokenKind::Semi)),
                just(PTok::Tok(TokenKind::RBrace)),
            )))
            .map_with(|_, e| Item::TopExpr {
                expr: Expr::Quantity {
                    value: 0.0,
                    unit: None,
                    span: to_span(e.span()),
                },
                span: to_span(e.span()),
            }),
    ));

    item.repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
        .map(|items| Program { items })
}

fn prepare_tokens(tokens: &[SpannedToken]) -> Vec<(PTok, SimpleSpan)> {
    let mut out = Vec::with_capacity(tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        let is_tight_unit_op = matches!(tok.kind, TokenKind::Star | TokenKind::Slash)
            && i > 0
            && i + 1 < tokens.len()
            && tokens[i - 1].span.end == tok.span.start
            && tok.span.end == tokens[i + 1].span.start
            && matches!(tokens[i + 1].kind, TokenKind::Ident(_) | TokenKind::KwIn)
            && tokens
                .get(i + 2)
                .is_none_or(|t| !matches!(t.kind, TokenKind::Dot | TokenKind::LParen));

        let wrapped = match (&tok.kind, is_tight_unit_op) {
            (TokenKind::Star, true) => PTok::TightStar,
            (TokenKind::Slash, true) => PTok::TightSlash,
            _ => PTok::Tok(tok.kind.clone()),
        };
        out.push((wrapped, SimpleSpan::from(tok.span.start..tok.span.end)));
    }
    out
}

/// Parses a token stream into an untyped AST and collects all recovered syntax diagnostics.
pub fn parse(tokens: &[SpannedToken], eof_offset: usize) -> (Option<Program>, Vec<Diagnostic>) {
    let stream_tokens = prepare_tokens(tokens);
    let eof_span = SimpleSpan::from(eof_offset..eof_offset);
    let input = Stream::from_iter(stream_tokens).map(eof_span, |(t, s)| (t, s));
    let (output, errs) = program_parser().parse(input).into_output_errors();

    let diagnostics = errs
        .into_iter()
        .map(|err| {
            let sp = to_span(*err.span());
            Diagnostic::error(
                "E0102",
                format!("Syntax error: {err}"),
                sp,
                "unexpected token",
            )
        })
        .collect();

    (output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;

    #[test]
    fn parses_reference_body_and_flagship_programs() {
        for src in [
            include_str!("../../../examples/reference_body.aero"),
            include_str!("../../../examples/rocket_6dof.aero"),
            include_str!("../../../examples/quadcopter_pid.aero"),
        ] {
            let (tokens, l_errs) = lex(src);
            assert!(l_errs.is_empty(), "{l_errs:?}");
            let (ast, p_errs) = parse(&tokens, src.len());
            assert!(p_errs.is_empty(), "{p_errs:?}");
            let Some(prog) = ast else {
                panic!("expected parsed AST");
            };
            assert!(!prog.items.is_empty());
        }
    }

    #[test]
    fn distinguishes_literal_units_from_runtime_variable_division() {
        let src = r#"
            state Body { mass: kg; }
            system Step(dt: s) {
                step(prev: Body) -> next: Body {
                    let accel: m/s^2 = 100.0 N / prev.mass;
                    let accel_tight: m/s^2 = 100.0 N/prev.mass;
                    let dist: m = 5.0 m/s * dt;
                    next.mass = prev.mass;
                }
            }
        "#;
        let (tokens, l_errs) = lex(src);
        assert!(l_errs.is_empty(), "{l_errs:?}");
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        assert!(ast.is_some());
    }

    #[test]
    fn parses_ultra_v2_constructs() {
        let src = r#"
            unit Thrust = N;
            unit Inertia = kg*m^2;
            fn clamp_pos(x: m) -> m {
                return if x > 0.0 m { x } else { 0.0 m };
            }
            state Craft {
                pos: Vec3<m>;
                rot: Quat;
                inertia: Mat3<Inertia>;
            }
            system Step(dt: s) {
                step(prev: Craft) -> next: Craft {
                    for i in 0..2 {
                        let a: m = clamp_pos(prev.pos.y);
                    }
                    next.pos = prev.pos;
                    next.rot = prev.rot;
                    next.inertia = prev.inertia;
                }
            }
            test "unit_check" {
                assert_approx(12.0 in, 0.3048 m, 1e-6 m);
            }
        "#;
        let (tokens, l_errs) = lex(src);
        assert!(l_errs.is_empty(), "{l_errs:?}");
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let Some(prog) = ast else {
            panic!("expected parsed AST");
        };
        assert_eq!(prog.items.len(), 6);
    }

    #[test]
    fn recovers_across_multiple_syntax_errors() {
        let src = "let broken = ; let ok: m/s = 10.0 m / 2.0 s;";
        let (tokens, _) = lex(src);
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(!p_errs.is_empty());
        assert_eq!(p_errs[0].code, "E0102");
        let Some(prog) = ast else {
            panic!("expected recovered AST");
        };
        assert_eq!(prog.items.len(), 2);
    }
}
