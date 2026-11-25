import { formatAst, Program } from './ast';
import { emitCHeader, emitLlvmIr } from './codegen';
import { lex, Token } from './lexer';
import { formatMir, lowerToMir, MirProgram, SoAStateLayout } from './mir';
import { parse } from './parser';
import { analyze, HirProgram } from './sema';
import { Diagnostic, formatAriadneDiagnostic } from './span';

export interface CompileResult {
  tokens: Token[];
  ast: Program;
  astText: string;
  hir: HirProgram;
  mir: MirProgram | null;
  mirText: string;
  soaLayouts: SoAStateLayout[];
  llvmIr: string;
  cHeader: string;
  diagnostics: Diagnostic[];
  renderedDiagnostics: string[];
  hasErrors: boolean;
}

export function compileAero(source: string, filename = 'main.aero'): CompileResult {
  const { tokens, diagnostics: lexDiags } = lex(source);
  const { program: ast, diagnostics: parseDiags } = parse(tokens, source.length);
  const { hir, diagnostics: semaDiags } = analyze(ast);

  const diagnostics = [...lexDiags, ...parseDiags, ...semaDiags];
  const hasErrors = diagnostics.some((d) => d.severity === 'error');
  const renderedDiagnostics = diagnostics.map((d) =>
    formatAriadneDiagnostic(d, filename, source),
  );

  const astText = formatAst(ast);

  if (hasErrors) {
    return {
      tokens,
      ast,
      astText,
      hir,
      mir: null,
      mirText: '; Fix compiler errors above to lower HIR into SoA MIR.',
      soaLayouts: [],
      llvmIr: '; Fix compiler errors above to emit deterministic LLVM IR.',
      cHeader: '/* Fix compiler errors above to generate aerotick.h C-ABI header. */',
      diagnostics,
      renderedDiagnostics,
      hasErrors: true,
    };
  }

  const mir = lowerToMir(hir);
  return {
    tokens,
    ast,
    astText,
    hir,
    mir,
    mirText: formatMir(mir),
    soaLayouts: mir.layouts,
    llvmIr: emitLlvmIr(mir),
    cHeader: emitCHeader(mir),
    diagnostics,
    renderedDiagnostics,
    hasErrors: false,
  };
}
