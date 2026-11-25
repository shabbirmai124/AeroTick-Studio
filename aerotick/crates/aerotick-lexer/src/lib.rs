//! Phase 1: High-speed lexical analysis powered by `logos` with byte `Span` tracking.

use aerotick_span::{Diagnostic, Span};
use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*+[^*/])*\*+/")]
pub enum TokenKind {
    #[token("const")]
    KwConst,
    #[token("state")]
    KwState,
    #[token("system")]
    KwSystem,
    #[token("step")]
    KwStep,
    #[token("let")]
    KwLet,
    #[token("if")]
    KwIf,
    #[token("else")]
    KwElse,
    #[token("while")]
    KwWhile,
    #[token("loop")]
    KwLoop,
    #[token("Vec3")]
    KwVec3,
    #[token("Quat")]
    KwQuat,
    #[token("Mat4")]
    KwMat4,

    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", |lex| lex.slice().to_string())]
    Number(String),

    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[token("->")]
    Arrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
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
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KwConst => write!(f, "const"),
            Self::KwState => write!(f, "state"),
            Self::KwSystem => write!(f, "system"),
            Self::KwStep => write!(f, "step"),
            Self::KwLet => write!(f, "let"),
            Self::KwIf => write!(f, "if"),
            Self::KwElse => write!(f, "else"),
            Self::KwWhile => write!(f, "while"),
            Self::KwLoop => write!(f, "loop"),
            Self::KwVec3 => write!(f, "Vec3"),
            Self::KwQuat => write!(f, "Quat"),
            Self::KwMat4 => write!(f, "Mat4"),
            Self::Number(s) | Self::Ident(s) => write!(f, "{s}"),
            Self::Arrow => write!(f, "->"),
            Self::EqEq => write!(f, "=="),
            Self::BangEq => write!(f, "!="),
            Self::LtEq => write!(f, "<="),
            Self::GtEq => write!(f, ">="),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Caret => write!(f, "^"),
            Self::Eq => write!(f, "="),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::Colon => write!(f, ":"),
            Self::Semi => write!(f, ";"),
            Self::Comma => write!(f, ","),
            Self::Dot => write!(f, "."),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedToken {
    pub kind: TokenKind,
    pub span: Span,
}

/// Tokenizes `.aero` source code into a flat vector of `SpannedToken`s and collects lex errors.
pub fn lex(source: &str) -> (Vec<SpannedToken>, Vec<Diagnostic>) {
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();
    let mut lexer = TokenKind::lexer(source);

    while let Some(res) = lexer.next() {
        let range = lexer.span();
        let span = Span::new(range.start, range.end);
        match res {
            Ok(kind) => tokens.push(SpannedToken { kind, span }),
            Err(()) => {
                let slice = &source[range.start..range.end];
                diagnostics.push(Diagnostic::error(
                    "E0002",
                    format!("Unexpected character '{slice}'"),
                    span,
                    "invalid token",
                ));
            }
        }
    }

    (tokens, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_control_flow_and_physics_tokens() {
        let src = "if x >= 0.0 { let v: Vec3<m/s> = Vec3(0.0, 1.0, 0.0); } else { while true {} }";
        let (tokens, diags) = lex(src);
        assert!(diags.is_empty());
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwIf));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwElse));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwWhile));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::GtEq));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwVec3));
    }
}
