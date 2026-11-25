#![deny(clippy::unwrap_used, clippy::expect_used)]
//! AeroTick Ultra v2.0 Language Server (`aerotick-lsp`).
//! Implements stdio JSON-RPC LSP (`Content-Length` framing) providing:
//! - Live SI dimensional & double-buffered step-safety diagnostics (`publishDiagnostics`)
//! - Hover tooltips showing deduced SI units & 32B SoA SIMD alignment (`hover`)
//! - Workspace & document outline symbols (`documentSymbol`)

use aerotick_sema::{SymbolInfo, SymbolKind, analyze};
use aerotick_span::{Diagnostic, Severity, line_col_to_offset, offset_to_line_col};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;

#[derive(Default)]
struct DocState {
    text: String,
    symbols: Vec<SymbolInfo>,
}

fn lsp_range(src: &str, start: usize, end: usize) -> Value {
    let (sl, sc) = offset_to_line_col(src, start);
    let (el, ec) = offset_to_line_col(src, end.max(start));
    json!({
        "start": { "line": sl, "character": sc },
        "end": { "line": el, "character": ec }
    })
}

fn to_lsp_diag(src: &str, d: &Diagnostic) -> Value {
    let sev = match d.severity {
        Severity::Error => 1,
        Severity::Warning => 2,
    };
    let (start, end, detail) = match d.labels.first() {
        Some(lbl) => (lbl.span.start, lbl.span.end, lbl.message.as_str()),
        None => (0, 0, ""),
    };
    let message = if detail.is_empty() {
        d.title.clone()
    } else {
        format!("{}: {detail}", d.title)
    };
    json!({
        "range": lsp_range(src, start, end),
        "severity": sev,
        "code": d.code,
        "source": "aerotick",
        "message": message,
    })
}

fn lsp_symbol_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::UnitAlias => 26, // TypeParameter
        SymbolKind::Const => 14,     // Constant
        SymbolKind::State => 23,     // Struct
        SymbolKind::StateField => 8, // Field
        SymbolKind::Function => 12,  // Function
        SymbolKind::System => 5,     // Class
        SymbolKind::Local | SymbolKind::Param => 13,
        SymbolKind::Test => 12,
    }
}

fn analyze_doc(text: &str) -> (Vec<Value>, Vec<SymbolInfo>) {
    let (tokens, mut diags) = aerotick_lexer::lex(text);
    let (ast_opt, parse_diags) = aerotick_parser::parse(&tokens, text.len());
    diags.extend(parse_diags);
    let mut symbols = Vec::new();
    if let Some(ast) = ast_opt {
        let (hir, sema_diags) = analyze(&ast);
        diags.extend(sema_diags);
        symbols = hir.symbols;
    }
    let lsp_diags = diags.iter().map(|d| to_lsp_diag(text, d)).collect();
    (lsp_diags, symbols)
}

fn read_lsp_message<R: BufRead>(reader: &mut R) -> io::Result<Option<String>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:")
            && let Ok(len) = rest.trim().parse::<usize>()
        {
            content_length = Some(len);
        }
    }
    let len = content_length.unwrap_or(0);
    if len == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).ok())
}

fn send_lsp_message<W: Write>(writer: &mut W, payload: &Value) -> io::Result<()> {
    let body = serde_json::to_string(payload).unwrap_or_default();
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()
}

fn handle_request<W: Write>(
    msg: &Value,
    docs: &mut HashMap<String, DocState>,
    writer: &mut W,
) -> io::Result<bool> {
    let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let id = msg.get("id").cloned();

    match method {
        "initialize" => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "serverInfo": { "name": "aerotick-lsp", "version": "2.0.0" },
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "documentSymbolProvider": true
                    }
                }
            });
            send_lsp_message(writer, &resp)?;
        }
        "shutdown" => {
            let resp = json!({ "jsonrpc": "2.0", "id": id, "result": Value::Null });
            send_lsp_message(writer, &resp)?;
        }
        "exit" => return Ok(false),
        "textDocument/didOpen" => {
            if let (Some(uri), Some(text)) = (
                msg.pointer("/params/textDocument/uri")
                    .and_then(|v| v.as_str()),
                msg.pointer("/params/textDocument/text")
                    .and_then(|v| v.as_str()),
            ) {
                let (diags, symbols) = analyze_doc(text);
                docs.insert(
                    uri.to_string(),
                    DocState {
                        text: text.to_string(),
                        symbols,
                    },
                );
                let notif = json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": { "uri": uri, "diagnostics": diags }
                });
                send_lsp_message(writer, &notif)?;
            }
        }
        "textDocument/didChange" => {
            if let (Some(uri), Some(text)) = (
                msg.pointer("/params/textDocument/uri")
                    .and_then(|v| v.as_str()),
                msg.pointer("/params/contentChanges/0/text")
                    .and_then(|v| v.as_str()),
            ) {
                let (diags, symbols) = analyze_doc(text);
                docs.insert(
                    uri.to_string(),
                    DocState {
                        text: text.to_string(),
                        symbols,
                    },
                );
                let notif = json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": { "uri": uri, "diagnostics": diags }
                });
                send_lsp_message(writer, &notif)?;
            }
        }
        "textDocument/hover" => {
            let mut result = Value::Null;
            if let (Some(uri), Some(line), Some(col)) = (
                msg.pointer("/params/textDocument/uri")
                    .and_then(|v| v.as_str()),
                msg.pointer("/params/position/line")
                    .and_then(|v| v.as_u64()),
                msg.pointer("/params/position/character")
                    .and_then(|v| v.as_u64()),
            ) && let Some(doc) = docs.get(uri)
            {
                let off = line_col_to_offset(&doc.text, line as u32, col as u32);
                if let Some(sym) = doc.symbols.iter().find(|s| s.span.contains(off)) {
                    result = json!({
                        "contents": {
                            "kind": "markdown",
                            "value": format!("```aero\n{}\n```", sym.detail)
                        },
                        "range": lsp_range(&doc.text, sym.span.start, sym.span.end)
                    });
                }
            }
            let resp = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            send_lsp_message(writer, &resp)?;
        }
        "textDocument/documentSymbol" => {
            let mut out = Vec::new();
            if let Some(uri) = msg
                .pointer("/params/textDocument/uri")
                .and_then(|v| v.as_str())
                && let Some(doc) = docs.get(uri)
            {
                for sym in &doc.symbols {
                    let r = lsp_range(&doc.text, sym.span.start, sym.span.end);
                    out.push(json!({
                        "name": sym.name,
                        "detail": sym.detail,
                        "kind": lsp_symbol_kind(sym.kind),
                        "range": r.clone(),
                        "selectionRange": r
                    }));
                }
            }
            let resp = json!({ "jsonrpc": "2.0", "id": id, "result": out });
            send_lsp_message(writer, &resp)?;
        }
        _ => {
            if let Some(req_id) = id {
                let resp = json!({ "jsonrpc": "2.0", "id": req_id, "result": Value::Null });
                send_lsp_message(writer, &resp)?;
            }
        }
    }
    Ok(true)
}

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut docs: HashMap<String, DocState> = HashMap::new();

    while let Ok(Some(raw)) = read_lsp_message(&mut reader) {
        if let Ok(msg) = serde_json::from_str::<Value>(&raw)
            && let Ok(keep_going) = handle_request(&msg, &mut docs, &mut writer)
            && !keep_going
        {
            break;
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzes_diagnostics_and_symbols_for_lsp() {
        let valid = "const G: m/s^2 = 9.8 m/s^2;\nstate B { p: Vec3<m>; }";
        let (diags, syms) = analyze_doc(valid);
        assert!(diags.is_empty());
        assert!(
            syms.iter()
                .any(|s| s.name == "G" && s.detail.contains("Scalar"))
        );
        assert!(
            syms.iter()
                .any(|s| s.name == "p" && s.detail.contains("32B SIMD4"))
        );

        let invalid = "let bad = 1.0 m + 2.0 kg;";
        let (bad_diags, _) = analyze_doc(invalid);
        assert!(!bad_diags.is_empty());
        assert_eq!(bad_diags[0]["code"], "E0202");
    }

    #[test]
    fn handles_json_rpc_initialize_open_hover_and_symbols() {
        let mut docs = HashMap::new();
        let mut out = Vec::new();
        let init_msg = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} });
        assert!(matches!(
            handle_request(&init_msg, &mut docs, &mut out),
            Ok(true)
        ));

        let open_msg = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.aero",
                    "text": "const G: m/s^2 = 9.8 m/s^2;\nstate B { pos: Vec3<m>; }"
                }
            }
        });
        assert!(matches!(
            handle_request(&open_msg, &mut docs, &mut out),
            Ok(true)
        ));

        let hover_msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {
                "textDocument": { "uri": "file:///test.aero" },
                "position": { "line": 0, "character": 6 }
            }
        });
        assert!(matches!(
            handle_request(&hover_msg, &mut docs, &mut out),
            Ok(true)
        ));

        let sym_msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/documentSymbol",
            "params": { "textDocument": { "uri": "file:///test.aero" } }
        });
        assert!(matches!(
            handle_request(&sym_msg, &mut docs, &mut out),
            Ok(true)
        ));

        let stream = String::from_utf8_lossy(&out);
        assert!(stream.contains("\"aerotick-lsp\""));
        assert!(stream.contains("publishDiagnostics"));
        assert!(stream.contains("Scalar<m/s^2>"));
        assert!(stream.contains("\"pos\""));
    }
}
