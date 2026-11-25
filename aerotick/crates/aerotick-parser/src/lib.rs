//! Phase 2: Parser with `chumsky` error recovery and multi-error diagnostics
//! supporting Milestone 1 scalar units and Milestone 2 `const`/`state`/`system`/`step`/`Vec3`.

use aerotick_ast::{
    BinOp, ConstDecl, Expr, Item, Param, Program, StateDecl, StateField, StepBlock, Stmt,
    SystemDecl, TypeExpr, UnitExpr,
};
use aerotick_lexer::{SpannedToken, TokenKind};
use aerotick_span::{Diagnostic, Span};
use chumsky::input::ValueInput;
use chumsky::prelude::*;

fn to_span(s: SimpleSpan) -> Span {
    Span::new(s.start, s.end)
}

fn unit_parser<'a, I>(
    allow_one: bool,
) -> impl Parser<'a, I, UnitExpr, extra::Err<Rich<'a, TokenKind, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = TokenKind, Span = SimpleSpan>,
{
    recursive(move |unit_expr| {
        let sym = select! {
            TokenKind::Ident(name) => name,
        }
        .map_with(|name, e| UnitExpr::Symbol(name, to_span(e.span())));

        let atom = if allow_one {
            let one = select! {
                TokenKind::Number(n) if n == "1" => (),
            }
            .map_with(|(), e| UnitExpr::One(to_span(e.span())));

            choice((
                sym,
                one,
                unit_expr.delimited_by(just(TokenKind::LParen), just(TokenKind::RParen)),
            ))
            .boxed()
        } else {
            sym.boxed()
        };

        let pow = atom
            .then(
                just(TokenKind::Caret)
                    .ignore_then(just(TokenKind::Minus).or_not())
                    .then(
                        select! { TokenKind::Number(n) => n }.try_map(|num_str, span| {
                            num_str.parse::<i8>().map_err(|_| {
                                Rich::custom(
                                    span,
                                    format!("unit exponent '{num_str}' must be an integer"),
                                )
                            })
                        }),
                    )
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

        pow.clone().foldl(
            choice((
                just(TokenKind::Star).to(true),
                just(TokenKind::Slash).to(false),
            ))
            .then(pow)
            .repeated(),
            |lhs, (is_mul, rhs)| {
                let sp = lhs.span().merge(rhs.span());
                if is_mul {
                    UnitExpr::Mul(Box::new(lhs), Box::new(rhs), sp)
                } else {
                    UnitExpr::Div(Box::new(lhs), Box::new(rhs), sp)
                }
            },
        )
    })
}

fn type_parser<'a, I>() -> impl Parser<'a, I, TypeExpr, extra::Err<Rich<'a, TokenKind, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = TokenKind, Span = SimpleSpan>,
{
    let ann_unit = unit_parser(true);
    let vec3_ty = just(TokenKind::KwVec3)
        .ignore_then(
            ann_unit
                .clone()
                .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt)),
        )
        .map_with(|u, e| TypeExpr::Vec3(u, to_span(e.span())));

    let quat_ty = just(TokenKind::KwQuat).map_with(|_, e| TypeExpr::Quat(to_span(e.span())));
    let mat4_ty = just(TokenKind::KwMat4).map_with(|_, e| TypeExpr::Mat4(to_span(e.span())));
    let scalar_ty = ann_unit.map_with(|u, e| TypeExpr::ScalarUnit(u, to_span(e.span())));

    choice((vec3_ty, quat_ty, mat4_ty, scalar_ty))
}

fn expr_parser<'a, I>() -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, TokenKind, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = TokenKind, Span = SimpleSpan>,
{
    let literal_unit_suffix = unit_parser(false);

    recursive(|expr| {
        let quantity = select! {
            TokenKind::Number(n) => n,
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

        let vec3_lit = just(TokenKind::KwVec3)
            .ignore_then(
                expr.clone()
                    .then_ignore(just(TokenKind::Comma))
                    .then(expr.clone())
                    .then_ignore(just(TokenKind::Comma))
                    .then(expr.clone())
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen)),
            )
            .map_with(|((x, y), z), e| Expr::Vec3Lit {
                x: Box::new(x),
                y: Box::new(y),
                z: Box::new(z),
                span: to_span(e.span()),
            });

        let call = select! {
            TokenKind::Ident(name) => name,
        }
        .map_with(|name, e| (name, to_span(e.span())))
        .then(
            expr.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen)),
        )
        .map_with(|((callee, callee_span), args), e| Expr::Call {
            callee,
            callee_span,
            args,
            span: to_span(e.span()),
        });

        let ident = select! {
            TokenKind::Ident(name) => name,
        }
        .map_with(|name, e| Expr::Ident(name, to_span(e.span())));

        let parens = expr
            .clone()
            .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen));

        let atom = choice((vec3_lit, call, quantity, ident, parens));

        let postfix = atom.foldl(
            just(TokenKind::Dot)
                .ignore_then(
                    select! { TokenKind::Ident(f) => f }
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
            just(TokenKind::Caret).ignore_then(postfix).repeated(),
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

        let unary = just(TokenKind::Minus)
            .repeated()
            .foldr(power, |_op, rhs| {
                let sp = rhs.span();
                Expr::Neg(Box::new(rhs), sp)
            });

        let product = unary.clone().foldl(
            choice((
                just(TokenKind::Star).to(BinOp::Mul),
                just(TokenKind::Slash).to(BinOp::Div),
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
                just(TokenKind::Plus).to(BinOp::Add),
                just(TokenKind::Minus).to(BinOp::Sub),
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
                just(TokenKind::EqEq).to(BinOp::Eq),
                just(TokenKind::BangEq).to(BinOp::Ne),
                just(TokenKind::Lt).to(BinOp::Lt),
                just(TokenKind::LtEq).to(BinOp::Le),
                just(TokenKind::Gt).to(BinOp::Gt),
                just(TokenKind::GtEq).to(BinOp::Ge),
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

fn stmt_parser<'a, I>() -> impl Parser<'a, I, Stmt, extra::Err<Rich<'a, TokenKind, SimpleSpan>>> + Clone
where
    I: ValueInput<'a, Token = TokenKind, Span = SimpleSpan>,
{
    let expr = expr_parser();
    let ty = type_parser();

    recursive(move |stmt| {
        let block = stmt
            .clone()
            .repeated()
            .collect::<Vec<_>>()
            .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace));

        let let_stmt = just(TokenKind::KwLet)
            .ignore_then(
                select! { TokenKind::Ident(name) => name }.map_with(|n, e| (n, to_span(e.span()))),
            )
            .then(just(TokenKind::Colon).ignore_then(ty.clone()).or_not())
            .then_ignore(just(TokenKind::Eq))
            .then(expr.clone())
            .then_ignore(just(TokenKind::Semi))
            .map_with(|(((name, name_span), type_ann), value), e| Stmt::Let {
                name,
                name_span,
                type_ann,
                value,
                span: to_span(e.span()),
            });

        let assign_stmt = expr
            .clone()
            .then_ignore(just(TokenKind::Eq))
            .then(expr.clone())
            .then_ignore(just(TokenKind::Semi))
            .map_with(|(target, value), e| Stmt::Assign {
                target,
                value,
                span: to_span(e.span()),
            });

        let if_stmt = just(TokenKind::KwIf)
            .ignore_then(expr.clone())
            .then(block.clone())
            .then(just(TokenKind::KwElse).ignore_then(block.clone()).or_not())
            .map_with(|((cond, then_branch), else_branch), e| Stmt::If {
                cond,
                then_branch,
                else_branch,
                span: to_span(e.span()),
            });

        let loop_stmt = choice((
            just(TokenKind::KwWhile).to("while".to_string()),
            just(TokenKind::KwLoop).to("loop".to_string()),
        ))
        .then_ignore(none_of([TokenKind::LBrace]).repeated())
        .then_ignore(block)
        .map_with(|keyword, e| Stmt::ForbiddenLoop {
            keyword,
            span: to_span(e.span()),
        });

        let expr_stmt = expr
            .clone()
            .then_ignore(just(TokenKind::Semi))
            .map_with(|expr, e| Stmt::Expr {
                expr,
                span: to_span(e.span()),
            });

        choice((let_stmt, if_stmt, loop_stmt, assign_stmt, expr_stmt)).recover_with(via_parser(
            none_of([TokenKind::Semi, TokenKind::RBrace])
                .repeated()
                .then_ignore(just(TokenKind::Semi))
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

fn program_parser<'a, I>() -> impl Parser<'a, I, Program, extra::Err<Rich<'a, TokenKind, SimpleSpan>>>
where
    I: ValueInput<'a, Token = TokenKind, Span = SimpleSpan>,
{
    let expr = expr_parser();
    let ty = type_parser();
    let stmt = stmt_parser();

    let ident_spanned =
        select! { TokenKind::Ident(name) => name }.map_with(|n, e| (n, to_span(e.span())));

    let const_item = just(TokenKind::KwConst)
        .ignore_then(ident_spanned.clone())
        .then(just(TokenKind::Colon).ignore_then(ty.clone()).or_not())
        .then_ignore(just(TokenKind::Eq))
        .then(expr.clone())
        .then_ignore(just(TokenKind::Semi))
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
        .clone()
        .then_ignore(just(TokenKind::Colon))
        .then(ty.clone())
        .then_ignore(just(TokenKind::Semi))
        .map_with(|((name, name_span), type_expr), e| StateField {
            name,
            name_span,
            type_expr,
            span: to_span(e.span()),
        });

    let state_item = just(TokenKind::KwState)
        .ignore_then(ident_spanned.clone())
        .then(
            state_field
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace)),
        )
        .map_with(|((name, name_span), fields), e| {
            Item::State(StateDecl {
                name,
                name_span,
                fields,
                span: to_span(e.span()),
            })
        });

    let param = ident_spanned
        .clone()
        .then_ignore(just(TokenKind::Colon))
        .then(ty.clone())
        .map_with(|((name, name_span), type_expr), e| Param {
            name,
            name_span,
            type_expr,
            span: to_span(e.span()),
        });

    let step_block = just(TokenKind::KwStep)
        .ignore_then(
            ident_spanned
                .clone()
                .then_ignore(just(TokenKind::Colon))
                .then(ident_spanned.clone())
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen)),
        )
        .then_ignore(just(TokenKind::Arrow))
        .then(ident_spanned.clone())
        .then_ignore(just(TokenKind::Colon))
        .then(ident_spanned.clone())
        .then(
            stmt.repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace)),
        )
        .map_with(
            |(((((prev_name, _), (prev_type, prev_type_span)), (next_name, _)), (next_type, next_type_span)), stmts), e| {
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

    let system_item = just(TokenKind::KwSystem)
        .ignore_then(ident_spanned.clone())
        .then(
            param
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen)),
        )
        .then(step_block.delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace)))
        .map_with(|(((name, name_span), params), step), e| {
            Item::System(SystemDecl {
                name,
                name_span,
                params,
                step,
                span: to_span(e.span()),
            })
        });

    let top_let = just(TokenKind::KwLet)
        .ignore_then(ident_spanned)
        .then(just(TokenKind::Colon).ignore_then(ty).or_not())
        .then_ignore(just(TokenKind::Eq))
        .then(expr.clone())
        .then_ignore(just(TokenKind::Semi))
        .map_with(|(((name, name_span), type_ann), value), e| Item::TopLet {
            name,
            name_span,
            type_ann,
            value,
            span: to_span(e.span()),
        });

    let top_expr = expr
        .then_ignore(just(TokenKind::Semi))
        .map_with(|expr, e| Item::TopExpr {
            expr,
            span: to_span(e.span()),
        });

    let item = choice((const_item, state_item, system_item, top_let, top_expr)).recover_with(
        via_parser(
            none_of([TokenKind::Semi, TokenKind::RBrace])
                .repeated()
                .then_ignore(choice((just(TokenKind::Semi), just(TokenKind::RBrace))))
                .map_with(|_, e| Item::TopExpr {
                    expr: Expr::Quantity {
                        value: 0.0,
                        unit: None,
                        span: to_span(e.span()),
                    },
                    span: to_span(e.span()),
                }),
        ),
    );

    item.repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
        .map(|items| Program { items })
}

/// Parses a token stream into an untyped AST and collects all recovered syntax diagnostics.
pub fn parse(tokens: &[SpannedToken], eof_offset: usize) -> (Option<Program>, Vec<Diagnostic>) {
    let stream = tokens
        .iter()
        .map(|t| (t.kind.clone(), SimpleSpan::from(t.span.start..t.span.end)))
        .collect::<Vec<_>>();

    let eof_span = SimpleSpan::from(eof_offset..eof_offset);
    let input = stream.as_slice().map(eof_span, |(t, s)| (t.clone(), *s));
    let (output, errs) = program_parser().parse(input).into_output_errors();

    let diagnostics = errs
        .into_iter()
        .map(|err| {
            let sp = to_span(*err.span());
            Diagnostic::error("E0102", format!("Syntax error: {err}"), sp, "unexpected token")
        })
        .collect();

    (output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerotick_lexer::lex;

    #[test]
    fn parses_reference_body_integration_program() {
        let src = include_str!("../../../examples/reference_body.aero");
        let (tokens, l_errs) = lex(src);
        assert!(l_errs.is_empty());
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(p_errs.is_empty(), "{p_errs:?}");
        let prog = ast.expect("parsed program");
        assert_eq!(prog.items.len(), 3);
    }

    #[test]
    fn recovers_across_multiple_syntax_errors() {
        let src = "let broken = ; let ok: m/s = 10.0 m / 2.0 s;";
        let (tokens, _) = lex(src);
        let (ast, p_errs) = parse(&tokens, src.len());
        assert!(!p_errs.is_empty());
        assert_eq!(ast.expect("recovered").items.len(), 2);
    }
}
