#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Phase 1: Zero-copy DFA tokenizer powered by `logos` for AeroTick Ultra v2.0.

use aerotick_span::{Diagnostic, Span};
use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"//[^\n]*")]
pub enum TokenKind {
    // Keywords
    #[token("unit")]
    KwUnit,
    #[token("const")]
    KwConst,
    #[token("state")]
    KwState,
    #[token("fn")]
    KwFn,
    #[token("system")]
    KwSystem,
    #[token("step")]
    KwStep,
    #[token("test")]
    KwTest,
    #[token("let")]
    KwLet,
    #[token("return")]
    KwReturn,
    #[token("if")]
    KwIf,
    #[token("else")]
    KwElse,
    #[token("for")]
    KwFor,
    #[token("in")]
    KwIn,
    #[token("while")]
    KwWhile,
    #[token("loop")]
    KwLoop,

    // Built-in composite types
    #[token("Vec3")]
    KwVec3,
    #[token("Quat")]
    KwQuat,
    #[token("Mat3")]
    KwMat3,
    #[token("Mat4")]
    KwMat4,

    // Multi-char operators & punctuation
    #[token("->")]
    Arrow,
    #[token("..")]
    DotDot,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,

    // Single-char operators & delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("^")]
    Caret,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    // Literals & identifiers
    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", |lex| lex.slice().to_string())]
    Number(String),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[1..s.len().saturating_sub(1)].to_string()
    })]
    StringLit(String),

    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KwUnit => write!(f, "unit"),
            Self::KwConst => write!(f, "const"),
            Self::KwState => write!(f, "state"),
            Self::KwFn => write!(f, "fn"),
            Self::KwSystem => write!(f, "system"),
            Self::KwStep => write!(f, "step"),
            Self::KwTest => write!(f, "test"),
            Self::KwLet => write!(f, "let"),
            Self::KwReturn => write!(f, "return"),
            Self::KwIf => write!(f, "if"),
            Self::KwElse => write!(f, "else"),
            Self::KwFor => write!(f, "for"),
            Self::KwIn => write!(f, "in"),
            Self::KwWhile => write!(f, "while"),
            Self::KwLoop => write!(f, "loop"),
            Self::KwVec3 => write!(f, "Vec3"),
            Self::KwQuat => write!(f, "Quat"),
            Self::KwMat3 => write!(f, "Mat3"),
            Self::KwMat4 => write!(f, "Mat4"),
            Self::Arrow => write!(f, "->"),
            Self::DotDot => write!(f, ".."),
            Self::EqEq => write!(f, "=="),
            Self::BangEq => write!(f, "!="),
            Self::LtEq => write!(f, "<="),
            Self::GtEq => write!(f, ">="),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::Colon => write!(f, ":"),
            Self::Semi => write!(f, ";"),
            Self::Comma => write!(f, ","),
            Self::Dot => write!(f, "."),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Caret => write!(f, "^"),
            Self::Eq => write!(f, "="),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::Number(n) => write!(f, "{n}"),
            Self::StringLit(s) => write!(f, "\"{s}\""),
            Self::Ident(id) => write!(f, "{id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedToken {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String,
}

/// Lexes an `.aero` source buffer into spanned tokens and collects `E0101` lexer diagnostics.
pub fn lex(source: &str) -> (Vec<SpannedToken>, Vec<Diagnostic>) {
    let mut lexer = TokenKind::lexer(source);
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();

    while let Some(res) = lexer.next() {
        let range = lexer.span();
        let span = Span::new(range.start, range.end);
        let lexeme = lexer.slice().to_string();
        match res {
            Ok(kind) => tokens.push(SpannedToken { kind, span, lexeme }),
            Err(()) => diagnostics.push(Diagnostic::error(
                "E0101",
                format!("Invalid character or token '{lexeme}'"),
                span,
                "unrecognized lexical token",
            )),
        }
    }

    (tokens, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_6dof_units_and_keywords() {
        let src = r#"
            unit Thrust = N;
            fn f(q: Quat, m: Mat3<kg*m^2>) -> Vec3<m/s> {
                for i in 0..4 {
                    let x = 12.0 in;
                }
                return Vec3(1.0 m/s, 0.0, 0.0);
            }
            test "unit_check" {}
        "#;
        let (tokens, diags) = lex(src);
        assert!(diags.is_empty(), "{diags:?}");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwUnit));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwQuat));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwMat3));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::DotDot));
        assert!(
            tokens
                .iter()
                .any(|t| t.kind == TokenKind::StringLit("unit_check".into()))
        );
    }

    #[test]
    fn emits_e0101_for_invalid_characters() {
        let (tokens, diags) = lex("let x = 1.0 @ 2.0;");
        assert!(!tokens.is_empty());
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "E0101");
    }
}
