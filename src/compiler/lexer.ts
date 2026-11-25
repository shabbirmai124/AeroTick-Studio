import { Diagnostic, makeError, Span } from './span';

export type TokenType =
  | 'KwConst'
  | 'KwState'
  | 'KwSystem'
  | 'KwStep'
  | 'KwLet'
  | 'KwIf'
  | 'KwElse'
  | 'KwWhile'
  | 'KwLoop'
  | 'KwVec3'
  | 'KwQuat'
  | 'KwMat4'
  | 'Number'
  | 'Ident'
  | 'Arrow'
  | 'EqEq'
  | 'BangEq'
  | 'LtEq'
  | 'GtEq'
  | 'Plus'
  | 'Minus'
  | 'Star'
  | 'Slash'
  | 'Caret'
  | 'Eq'
  | 'LParen'
  | 'RParen'
  | 'LBrace'
  | 'RBrace'
  | 'Lt'
  | 'Gt'
  | 'Colon'
  | 'Semi'
  | 'Comma'
  | 'Dot';

export interface Token {
  type: TokenType;
  lexeme: string;
  span: Span;
}

const KEYWORDS: Record<string, TokenType> = {
  const: 'KwConst',
  state: 'KwState',
  system: 'KwSystem',
  step: 'KwStep',
  let: 'KwLet',
  if: 'KwIf',
  else: 'KwElse',
  while: 'KwWhile',
  loop: 'KwLoop',
  Vec3: 'KwVec3',
  Quat: 'KwQuat',
  Mat4: 'KwMat4',
};

const TWO_CHAR_OPS: Array<[string, TokenType]> = [
  ['->', 'Arrow'],
  ['==', 'EqEq'],
  ['!=', 'BangEq'],
  ['<=', 'LtEq'],
  ['>=', 'GtEq'],
];

const ONE_CHAR_OPS: Record<string, TokenType> = {
  '+': 'Plus',
  '-': 'Minus',
  '*': 'Star',
  '/': 'Slash',
  '^': 'Caret',
  '=': 'Eq',
  '(': 'LParen',
  ')': 'RParen',
  '{': 'LBrace',
  '}': 'RBrace',
  '<': 'Lt',
  '>': 'Gt',
  ':': 'Colon',
  ';': 'Semi',
  ',': 'Comma',
  '.': 'Dot',
};

export function lex(source: string): { tokens: Token[]; diagnostics: Diagnostic[] } {
  const tokens: Token[] = [];
  const diagnostics: Diagnostic[] = [];
  let i = 0;
  const n = source.length;

  while (i < n) {
    const ch = source[i];

    // Whitespace
    if (ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n') {
      i++;
      continue;
    }

    // Line comment //
    if (ch === '/' && source[i + 1] === '/') {
      i += 2;
      while (i < n && source[i] !== '\n') i++;
      continue;
    }

    // Block comment /* ... */
    if (ch === '/' && source[i + 1] === '*') {
      const start = i;
      i += 2;
      let closed = false;
      while (i < n) {
        if (source[i] === '*' && source[i + 1] === '/') {
          i += 2;
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) {
        diagnostics.push(
          makeError('E0001', 'Unterminated block comment', { start, end: n }, 'missing `*/`'),
        );
      }
      continue;
    }

    // Number literal with optional scientific notation
    if (ch >= '0' && ch <= '9') {
      const rest = source.slice(i);
      const m = /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/.exec(rest);
      if (m) {
        const lexeme = m[0];
        tokens.push({ type: 'Number', lexeme, span: { start: i, end: i + lexeme.length } });
        i += lexeme.length;
        continue;
      }
    }

    // Identifier or Keyword
    if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch === '_') {
      const rest = source.slice(i);
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(rest);
      if (m) {
        const lexeme = m[0];
        const type = KEYWORDS[lexeme] ?? 'Ident';
        tokens.push({ type, lexeme, span: { start: i, end: i + lexeme.length } });
        i += lexeme.length;
        continue;
      }
    }

    // Two-char operator
    const two = source.slice(i, i + 2);
    const twoMatch = TWO_CHAR_OPS.find(([op]) => op === two);
    if (twoMatch) {
      tokens.push({ type: twoMatch[1], lexeme: twoMatch[0], span: { start: i, end: i + 2 } });
      i += 2;
      continue;
    }

    // Single-char operator
    const oneType = ONE_CHAR_OPS[ch];
    if (oneType) {
      tokens.push({ type: oneType, lexeme: ch, span: { start: i, end: i + 1 } });
      i++;
      continue;
    }

    diagnostics.push(
      makeError('E0002', `Unexpected character '${ch}'`, { start: i, end: i + 1 }, 'invalid token'),
    );
    i++;
  }

  return { tokens, diagnostics };
}
