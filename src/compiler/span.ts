export interface Span {
  start: number;
  end: number;
}

export type Severity = 'error' | 'warning';

export interface DiagnosticLabel {
  span: Span;
  message: string;
  isPrimary: boolean;
}

export interface Diagnostic {
  severity: Severity;
  code: string;
  title: string;
  labels: DiagnosticLabel[];
  help?: string;
}

export function mergeSpans(a: Span, b: Span): Span {
  return {
    start: Math.min(a.start, b.start),
    end: Math.max(a.end, b.end),
  };
}

export function makeError(
  code: string,
  title: string,
  span: Span,
  label: string,
  secondary: Array<{ span: Span; message: string }> = [],
  help?: string,
): Diagnostic {
  return {
    severity: 'error',
    code,
    title,
    labels: [
      { span, message: label, isPrimary: true },
      ...secondary.map((s) => ({ span: s.span, message: s.message, isPrimary: false })),
    ],
    help,
  };
}

export function makeWarning(
  code: string,
  title: string,
  span: Span,
  label: string,
  help?: string,
): Diagnostic {
  return {
    severity: 'warning',
    code,
    title,
    labels: [{ span, message: label, isPrimary: true }],
    help,
  };
}

export function offsetToLineCol(source: string, offset: number): { line: number; col: number } {
  const clamped = Math.max(0, Math.min(offset, source.length));
  let line = 1;
  let lineStart = 0;
  for (let i = 0; i < clamped; i++) {
    if (source.charCodeAt(i) === 10) {
      line++;
      lineStart = i + 1;
    }
  }
  return { line, col: clamped - lineStart + 1 };
}

export function formatAriadneDiagnostic(
  diag: Diagnostic,
  filename: string,
  source: string,
): string {
  const lines = source.split('\n');
  const primary = diag.labels[0];
  const { line, col } = offsetToLineCol(source, primary ? primary.span.start : 0);
  const tag = diag.severity === 'error' ? 'Error' : 'Warning';
  const out: string[] = [];
  out.push(`[${diag.code}] ${tag}: ${diag.title}`);
  out.push(`   ╭─[ ${filename}:${line}:${col} ]`);

  const lineMap = new Map<number, DiagnosticLabel[]>();
  for (const lbl of diag.labels) {
    const loc = offsetToLineCol(source, lbl.span.start);
    const list = lineMap.get(loc.line) ?? [];
    list.push(lbl);
    lineMap.set(loc.line, list);
  }

  const sortedLines = [...lineMap.keys()].sort((a, b) => a - b);
  for (const ln of sortedLines) {
    const rawLine = lines[ln - 1] ?? '';
    const lineNumStr = String(ln).padStart(3, ' ');
    let lineStartOffset = 0;
    for (let i = 0; i < ln - 1; i++) {
      lineStartOffset += (lines[i]?.length ?? 0) + 1;
    }
    out.push(`${lineNumStr} │ ${rawLine}`);
    for (const lbl of lineMap.get(ln) ?? []) {
      const relStart = Math.max(0, lbl.span.start - lineStartOffset);
      const width = Math.max(1, Math.min(rawLine.length - relStart, lbl.span.end - lbl.span.start));
      const marker = lbl.isPrimary ? '┬' : '─';
      const pad = ' '.repeat(relStart);
      const underline = marker.repeat(width);
      out.push(`    │ ${pad}${underline} ${lbl.message}`);
    }
  }

  if (diag.help) {
    out.push(`    │ help: ${diag.help}`);
  }
  out.push('   ╯');
  return out.join('\n');
}
