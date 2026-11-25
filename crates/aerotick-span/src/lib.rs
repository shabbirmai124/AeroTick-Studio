#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Source span registry, line/column conversion helpers, and `ariadne` multi-span diagnostic renderer.

use ariadne::{Color, Label, Report, ReportKind, Source};
use std::fmt;

/// Half-open byte span `[start, end)` within a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn merge(self, other: Self) -> Self {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Self { start, end }
    }

    pub const fn contains(self, offset: usize) -> bool {
        offset >= self.start && offset <= self.end
    }
}

impl From<std::ops::Range<usize>> for Span {
    fn from(r: std::ops::Range<usize>) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
    pub is_primary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: Severity,
    pub title: String,
    pub labels: Vec<DiagnosticLabel>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(
        code: &'static str,
        title: impl Into<String>,
        span: Span,
        label: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity: Severity::Error,
            title: title.into(),
            labels: vec![DiagnosticLabel {
                span,
                message: label.into(),
                is_primary: true,
            }],
            help: None,
        }
    }

    pub fn warning(
        code: &'static str,
        title: impl Into<String>,
        span: Span,
        label: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity: Severity::Warning,
            title: title.into(),
            labels: vec![DiagnosticLabel {
                span,
                message: label.into(),
                is_primary: true,
            }],
            help: None,
        }
    }

    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: message.into(),
            is_primary: false,
        });
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Renders this diagnostic into a human-readable multi-span report using `ariadne`.
    pub fn render_to_string(&self, filename: &str, source: &str) -> std::io::Result<String> {
        let len = source.len();
        let primary_offset = self
            .labels
            .first()
            .map_or(0, |l| l.span.start.min(len));

        let kind = match self.severity {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
        };

        let mut builder = Report::build(kind, (filename, primary_offset..primary_offset))
            .with_code(self.code)
            .with_message(&self.title);

        for lbl in &self.labels {
            let start = lbl.span.start.min(len);
            let end = lbl.span.end.clamp(start, len);
            let color = if lbl.is_primary {
                match self.severity {
                    Severity::Error => Color::Red,
                    Severity::Warning => Color::Yellow,
                }
            } else {
                Color::Cyan
            };
            builder = builder.with_label(
                Label::new((filename, start..end))
                    .with_message(&lbl.message)
                    .with_color(color),
            );
        }

        if let Some(help) = &self.help {
            builder = builder.with_help(help);
        }

        let mut buf = Vec::new();
        builder
            .finish()
            .write((filename, Source::from(source)), &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.title)
    }
}

/// Converts a byte offset into 0-based `(line, character)` coordinates for LSP/diagnostics.
pub fn offset_to_line_col(source: &str, offset: usize) -> (u32, u32) {
    let clamped = offset.min(source.len());
    let mut line = 0u32;
    let mut col = 0u32;
    for (idx, ch) in source.char_indices() {
        if idx >= clamped {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Converts 0-based `(line, character)` coordinates back into a byte offset.
pub fn line_col_to_offset(source: &str, target_line: u32, target_col: u32) -> usize {
    let mut line = 0u32;
    let mut col = 0u32;
    for (idx, ch) in source.char_indices() {
        if line == target_line && col == target_col {
            return idx;
        }
        if ch == '\n' {
            if line == target_line {
                return idx;
            }
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    source.len()
}

/// Source registry storing named source buffers for multi-file compilation sessions.
#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<(String, String)>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, name: impl Into<String>, content: impl Into<String>) -> usize {
        let id = self.files.len();
        self.files.push((name.into(), content.into()));
        id
    }

    pub fn get(&self, id: usize) -> Option<(&str, &str)> {
        self.files
            .get(id)
            .map(|(n, c)| (n.as_str(), c.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_ariadne_diagnostic_and_converts_line_col() {
        let src = "let bad = 10.0 m + 5.0 kg;\nlet ok = 1.0 m;";
        let diag = Diagnostic::error(
            "E0202",
            "Physical unit mismatch",
            Span::new(10, 25),
            "cannot add m and kg",
        )
        .with_help("Convert operands to identical SI dimensions.");
        let Ok(rendered) = diag.render_to_string("test.aero", src) else {
            panic!("ariadne render failed");
        };
        assert!(rendered.contains("E0202"));
        assert!(rendered.contains("Physical unit mismatch"));

        let (line, col) = offset_to_line_col(src, 27);
        assert_eq!((line, col), (1, 0));
        assert_eq!(line_col_to_offset(src, 1, 0), 27);
    }
}
