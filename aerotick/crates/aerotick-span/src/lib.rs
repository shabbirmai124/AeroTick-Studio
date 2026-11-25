//! Source file registry, byte `Span` tracking, and Ariadne diagnostic rendering.

use ariadne::{Color, Label, Report, ReportKind, Source};
use thiserror::Error;

/// Opaque identifier for a registered source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FileId(pub usize);

/// Half-open byte range `[start, end)` within a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// Registered source file entry.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub content: String,
}

/// Registry storing loaded `.aero` source files for diagnostic reporting.
#[derive(Debug, Default, Clone)]
pub struct SourceRegistry {
    files: Vec<SourceFile>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, name: impl Into<String>, content: impl Into<String>) -> FileId {
        let id = FileId(self.files.len());
        self.files.push(SourceFile {
            name: name.into(),
            content: content.into(),
        });
        id
    }

    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0)
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

#[derive(Debug, Error)]
pub enum DiagnosticRenderError {
    #[error("failed to write formatted diagnostic: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: &'static str,
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
            severity: Severity::Error,
            code,
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
            severity: Severity::Warning,
            code,
            title: title.into(),
            labels: vec![DiagnosticLabel {
                span,
                message: label.into(),
                is_primary: true,
            }],
            help: None,
        }
    }

    pub fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: label.into(),
            is_primary: false,
        });
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Renders this diagnostic into a UTF-8 string using `ariadne`.
    pub fn render_to_string(
        &self,
        filename: &str,
        source: &str,
    ) -> Result<String, DiagnosticRenderError> {
        let kind = match self.severity {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
        };
        let primary_offset = self.labels.first().map(|l| l.span.start).unwrap_or(0);
        let mut builder = Report::build(kind, (filename, primary_offset..primary_offset))
            .with_code(self.code)
            .with_message(&self.title);

        for lbl in &self.labels {
            let color = if lbl.is_primary {
                Color::Red
            } else {
                Color::Cyan
            };
            let end = lbl.span.end.max(lbl.span.start.saturating_add(1));
            builder = builder.with_label(
                Label::new((filename, lbl.span.start..end))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_spans_correctly() {
        let a = Span::new(4, 8);
        let b = Span::new(12, 18);
        assert_eq!(a.merge(b), Span::new(4, 18));
    }

    #[test]
    fn registry_stores_and_retrieves_files() {
        let mut reg = SourceRegistry::new();
        let id = reg.add_file("main.aero", "let x = 10.0 m;");
        let file = reg.get(id).expect("registered file");
        assert_eq!(file.name, "main.aero");
        assert_eq!(file.content, "let x = 10.0 m;");
    }

    #[test]
    fn renders_ariadne_diagnostic() {
        let src = "let x = 10.0 m + 5.0 kg;";
        let diag = Diagnostic::error(
            "E0230",
            "Physical unit mismatch",
            Span::new(8, 23),
            "incompatible units m and kg",
        )
        .with_secondary(Span::new(8, 14), "has unit m")
        .with_secondary(Span::new(17, 23), "has unit kg")
        .with_help("Only identical SI dimensions may be added.");
        let rendered = diag.render_to_string("test.aero", src).expect("render");
        assert!(rendered.contains("E0230"));
        assert!(rendered.contains("Physical unit mismatch"));
        assert!(rendered.contains("incompatible units m and kg"));
    }
}
