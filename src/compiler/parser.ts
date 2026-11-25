import {
  BinOp,
  Expr,
  Item,
  Param,
  Program,
  StateField,
  Stmt,
  TypeExpr,
  UnitExpr,
} from './ast';
import { Token, TokenType } from './lexer';
import { Diagnostic, makeError, mergeSpans, Span } from './span';

class Parser {
  private pos = 0;
  readonly diagnostics: Diagnostic[] = [];

  constructor(
    private readonly tokens: Token[],
    private readonly eofOffset: number,
  ) {}

  private peek(offset = 0): Token | null {
    return this.tokens[this.pos + offset] ?? null;
  }

  private at(type: TokenType, offset = 0): boolean {
    return this.peek(offset)?.type === type;
  }

  private eofSpan(): Span {
    return { start: this.eofOffset, end: this.eofOffset };
  }

  private currentSpan(): Span {
    return this.peek()?.span ?? this.eofSpan();
  }

  private advance(): Token {
    const t = this.tokens[this.pos];
    if (t) this.pos++;
    return t ?? { type: 'Semi', lexeme: '<eof>', span: this.eofSpan() };
  }

  private expect(type: TokenType, expectedLabel: string): Token | null {
    const t = this.peek();
    if (t && t.type === type) {
      this.pos++;
      return t;
    }
    const sp = this.currentSpan();
    this.diagnostics.push(
      makeError(
        'E0102',
        `Syntax error: expected ${expectedLabel}, found '${t ? t.lexeme : 'end of file'}'`,
        sp,
        `expected ${expectedLabel}`,
      ),
    );
    return null;
  }

  private syncStatement(): void {
    while (this.pos < this.tokens.length) {
      const t = this.peek();
      if (!t) break;
      if (t.type === 'Semi') {
        this.pos++;
        break;
      }
      if (t.type === 'RBrace') break;
      this.pos++;
    }
  }

  private syncItem(): void {
    while (this.pos < this.tokens.length) {
      const t = this.peek();
      if (!t) break;
      if (t.type === 'Semi' || t.type === 'RBrace') {
        this.pos++;
        break;
      }
      if (
        t.type === 'KwConst' ||
        t.type === 'KwState' ||
        t.type === 'KwSystem' ||
        t.type === 'KwLet'
      ) {
        break;
      }
      this.pos++;
    }
  }

  parseProgram(): Program {
    const items: Item[] = [];
    while (this.pos < this.tokens.length) {
      const before = this.pos;
      const item = this.parseItem();
      if (item) {
        items.push(item);
      } else {
        this.syncItem();
        if (this.pos === before && this.pos < this.tokens.length) {
          this.pos++;
        }
      }
    }
    return { items };
  }

  private parseItem(): Item | null {
    const t = this.peek();
    if (!t) return null;
    if (t.type === 'KwConst') return this.parseConst();
    if (t.type === 'KwState') return this.parseState();
    if (t.type === 'KwSystem') return this.parseSystem();
    if (t.type === 'KwLet') return this.parseTopLet();

    const expr = this.parseExpr();
    if (!expr) return null;
    const semi = this.expect('Semi', "';'");
    return { kind: 'TopExpr', expr, span: mergeSpans(expr.span, semi?.span ?? expr.span) };
  }

  private parseConst(): Item | null {
    const kw = this.advance();
    const nameTok = this.expect('Ident', 'constant identifier');
    if (!nameTok) return null;

    let typeAnn: TypeExpr | null = null;
    if (this.at('Colon')) {
      this.advance();
      typeAnn = this.parseTypeExpr();
      if (!typeAnn) return null;
    }

    if (!this.expect('Eq', "'='")) return null;
    const value = this.parseExpr();
    if (!value) return null;
    const semi = this.expect('Semi', "';'");

    return {
      kind: 'Const',
      decl: {
        name: nameTok.lexeme,
        nameSpan: nameTok.span,
        typeAnn,
        value,
        span: mergeSpans(kw.span, semi?.span ?? value.span),
      },
    };
  }

  private parseState(): Item | null {
    const kw = this.advance();
    const nameTok = this.expect('Ident', 'state identifier');
    if (!nameTok) return null;
    if (!this.expect('LBrace', "'{'")) return null;

    const fields: StateField[] = [];
    while (this.pos < this.tokens.length && !this.at('RBrace')) {
      const fName = this.expect('Ident', 'field name');
      if (!fName) {
        this.syncStatement();
        continue;
      }
      if (!this.expect('Colon', "':'")) {
        this.syncStatement();
        continue;
      }
      const fType = this.parseTypeExpr();
      if (!fType) {
        this.syncStatement();
        continue;
      }
      const semi = this.expect('Semi', "';'");
      fields.push({
        name: fName.lexeme,
        nameSpan: fName.span,
        typeExpr: fType,
        span: mergeSpans(fName.span, semi?.span ?? fType.span),
      });
    }
    const rbrace = this.expect('RBrace', "'}'");
    return {
      kind: 'State',
      decl: {
        name: nameTok.lexeme,
        nameSpan: nameTok.span,
        fields,
        span: mergeSpans(kw.span, rbrace?.span ?? nameTok.span),
      },
    };
  }

  private parseSystem(): Item | null {
    const kw = this.advance();
    const nameTok = this.expect('Ident', 'system name');
    if (!nameTok) return null;
    if (!this.expect('LParen', "'('")) return null;

    const params: Param[] = [];
    while (this.pos < this.tokens.length && !this.at('RParen')) {
      const pName = this.expect('Ident', 'parameter name');
      if (!pName) return null;
      if (!this.expect('Colon', "':'")) return null;
      const pType = this.parseTypeExpr();
      if (!pType) return null;
      params.push({
        name: pName.lexeme,
        nameSpan: pName.span,
        typeExpr: pType,
        span: mergeSpans(pName.span, pType.span),
      });
      if (this.at('Comma')) this.advance();
      else break;
    }
    if (!this.expect('RParen', "')'")) return null;
    if (!this.expect('LBrace', "'{'")) return null;

    const stepKw = this.expect('KwStep', "'step'");
    if (!stepKw) return null;
    if (!this.expect('LParen', "'('")) return null;
    const prevName = this.expect('Ident', 'prev state binding');
    if (!prevName) return null;
    if (!this.expect('Colon', "':'")) return null;
    const prevType = this.expect('Ident', 'prev state type');
    if (!prevType) return null;
    if (!this.expect('RParen', "')'")) return null;
    if (!this.expect('Arrow', "'->'")) return null;
    const nextName = this.expect('Ident', 'next state binding');
    if (!nextName) return null;
    if (!this.expect('Colon', "':'")) return null;
    const nextType = this.expect('Ident', 'next state type');
    if (!nextType) return null;

    const stmts = this.parseBlock();
    if (!stmts) return null;
    const endBrace = this.expect('RBrace', "'}'");

    return {
      kind: 'System',
      decl: {
        name: nameTok.lexeme,
        nameSpan: nameTok.span,
        params,
        step: {
          prevName: prevName.lexeme,
          prevType: prevType.lexeme,
          prevTypeSpan: prevType.span,
          nextName: nextName.lexeme,
          nextType: nextType.lexeme,
          nextTypeSpan: nextType.span,
          stmts,
          span: mergeSpans(stepKw.span, endBrace?.span ?? nextType.span),
        },
        span: mergeSpans(kw.span, endBrace?.span ?? stepKw.span),
      },
    };
  }

  private parseTopLet(): Item | null {
    const kw = this.advance();
    const nameTok = this.expect('Ident', 'variable name');
    if (!nameTok) return null;
    let typeAnn: TypeExpr | null = null;
    if (this.at('Colon')) {
      this.advance();
      typeAnn = this.parseTypeExpr();
      if (!typeAnn) return null;
    }
    if (!this.expect('Eq', "'='")) return null;
    const value = this.parseExpr();
    if (!value) return null;
    const semi = this.expect('Semi', "';'");
    return {
      kind: 'TopLet',
      name: nameTok.lexeme,
      nameSpan: nameTok.span,
      typeAnn,
      value,
      span: mergeSpans(kw.span, semi?.span ?? value.span),
    };
  }

  private parseBlock(): Stmt[] | null {
    if (!this.expect('LBrace', "'{'")) return null;
    const stmts: Stmt[] = [];
    while (this.pos < this.tokens.length && !this.at('RBrace')) {
      const before = this.pos;
      const st = this.parseStmt();
      if (st) {
        stmts.push(st);
      } else {
        this.syncStatement();
        if (this.pos === before && this.pos < this.tokens.length && !this.at('RBrace')) {
          this.pos++;
        }
      }
    }
    this.expect('RBrace', "'}'");
    return stmts;
  }

  private parseStmt(): Stmt | null {
    const t = this.peek();
    if (!t) return null;

    if (t.type === 'KwLet') {
      const kw = this.advance();
      const nameTok = this.expect('Ident', 'variable name');
      if (!nameTok) return null;
      let typeAnn: TypeExpr | null = null;
      if (this.at('Colon')) {
        this.advance();
        typeAnn = this.parseTypeExpr();
        if (!typeAnn) return null;
      }
      if (!this.expect('Eq', "'='")) return null;
      const value = this.parseExpr();
      if (!value) return null;
      const semi = this.expect('Semi', "';'");
      return {
        kind: 'Let',
        name: nameTok.lexeme,
        nameSpan: nameTok.span,
        typeAnn,
        value,
        span: mergeSpans(kw.span, semi?.span ?? value.span),
      };
    }

    if (t.type === 'KwIf') {
      const kw = this.advance();
      const cond = this.parseExpr();
      if (!cond) return null;
      const thenBranch = this.parseBlock();
      if (!thenBranch) return null;
      let elseBranch: Stmt[] | null = null;
      if (this.at('KwElse')) {
        this.advance();
        elseBranch = this.parseBlock();
      }
      return {
        kind: 'If',
        cond,
        thenBranch,
        elseBranch,
        span: mergeSpans(kw.span, cond.span),
      };
    }

    if (t.type === 'KwWhile' || t.type === 'KwLoop') {
      const kw = this.advance();
      while (this.pos < this.tokens.length && !this.at('LBrace')) {
        this.pos++;
      }
      if (this.at('LBrace')) {
        let depth = 0;
        while (this.pos < this.tokens.length) {
          const cur = this.advance();
          if (cur.type === 'LBrace') depth++;
          else if (cur.type === 'RBrace') {
            depth--;
            if (depth <= 0) break;
          }
        }
      }
      return { kind: 'ForbiddenLoop', keyword: kw.lexeme, span: kw.span };
    }

    const lhs = this.parseExpr();
    if (!lhs) return null;
    if (this.at('Eq')) {
      this.advance();
      const rhs = this.parseExpr();
      if (!rhs) return null;
      const semi = this.expect('Semi', "';'");
      return {
        kind: 'Assign',
        target: lhs,
        value: rhs,
        span: mergeSpans(lhs.span, semi?.span ?? rhs.span),
      };
    }
    const semi = this.expect('Semi', "';'");
    return {
      kind: 'Expr',
      expr: lhs,
      span: mergeSpans(lhs.span, semi?.span ?? lhs.span),
    };
  }

  private parseTypeExpr(): TypeExpr | null {
    const t = this.peek();
    if (!t) {
      this.diagnostics.push(
        makeError('E0102', 'Syntax error: expected type annotation', this.eofSpan(), 'expected type'),
      );
      return null;
    }

    if (t.type === 'KwVec3') {
      const kw = this.advance();
      if (!this.expect('Lt', "'<'")) return null;
      const unit = this.parseUnitExpr(true);
      if (!unit) return null;
      const gt = this.expect('Gt', "'>'");
      return { kind: 'Vec3', unit, span: mergeSpans(kw.span, gt?.span ?? unit.span) };
    }
    if (t.type === 'KwQuat') {
      this.advance();
      return { kind: 'Quat', span: t.span };
    }
    if (t.type === 'KwMat4') {
      this.advance();
      return { kind: 'Mat4', span: t.span };
    }
    const unit = this.parseUnitExpr(true);
    if (!unit) return null;
    return { kind: 'ScalarUnit', unit, span: unit.span };
  }

  private parseUnitExpr(allowOne: boolean): UnitExpr | null {
    let left = this.parseUnitPow(allowOne);
    if (!left) return null;

    while (this.at('Star') || this.at('Slash')) {
      const nextAfterOp = this.peek(1);
      if (!allowOne && (!nextAfterOp || nextAfterOp.type !== 'Ident')) {
        break;
      }
      const op = this.advance();
      const right = this.parseUnitPow(allowOne);
      if (!right) return null;
      const span = mergeSpans(left.span, right.span);
      left =
        op.type === 'Star'
          ? { kind: 'Mul', left, right, span }
          : { kind: 'Div', left, right, span };
    }
    return left;
  }

  private parseUnitPow(allowOne: boolean): UnitExpr | null {
    const base = this.parseUnitAtom(allowOne);
    if (!base) return null;

    if (this.at('Caret')) {
      this.advance();
      let neg = false;
      if (this.at('Minus')) {
        this.advance();
        neg = true;
      }
      const expTok = this.expect('Number', 'integer unit exponent');
      if (!expTok) return null;
      const parsed = Number(expTok.lexeme);
      if (!Number.isInteger(parsed) || Math.abs(parsed) > 127) {
        this.diagnostics.push(
          makeError(
            'E0102',
            `Unit exponent '${expTok.lexeme}' must be an 8-bit integer`,
            expTok.span,
            'invalid exponent',
          ),
        );
        return null;
      }
      const exp = neg ? -parsed : parsed;
      return { kind: 'Pow', base, exp, span: mergeSpans(base.span, expTok.span) };
    }

    return base;
  }

  private parseUnitAtom(allowOne: boolean): UnitExpr | null {
    const t = this.peek();
    if (!t) {
      this.diagnostics.push(
        makeError('E0102', 'Expected physical unit', this.eofSpan(), 'expected unit'),
      );
      return null;
    }

    if (t.type === 'Ident') {
      this.advance();
      return { kind: 'Symbol', name: t.lexeme, span: t.span };
    }
    if (allowOne && t.type === 'Number' && t.lexeme === '1') {
      this.advance();
      return { kind: 'One', span: t.span };
    }
    if (allowOne && t.type === 'LParen') {
      this.advance();
      const inner = this.parseUnitExpr(true);
      if (!inner) return null;
      this.expect('RParen', "')'");
      return inner;
    }

    this.diagnostics.push(
      makeError('E0102', `Expected unit symbol, found '${t.lexeme}'`, t.span, 'invalid unit'),
    );
    return null;
  }

  private parseExpr(): Expr | null {
    return this.parseComparison();
  }

  private parseComparison(): Expr | null {
    let left = this.parseSum();
    if (!left) return null;

    const mapOp: Partial<Record<TokenType, BinOp>> = {
      EqEq: 'Eq',
      BangEq: 'Ne',
      Lt: 'Lt',
      LtEq: 'Le',
      Gt: 'Gt',
      GtEq: 'Ge',
    };

    while (true) {
      const t = this.peek();
      const op = t ? mapOp[t.type] : undefined;
      if (!op) break;
      this.advance();
      const right = this.parseSum();
      if (!right) return null;
      left = { kind: 'Binary', op, left, right, span: mergeSpans(left.span, right.span) };
    }
    return left;
  }

  private parseSum(): Expr | null {
    let left = this.parseProduct();
    if (!left) return null;

    while (this.at('Plus') || this.at('Minus')) {
      const opTok = this.advance();
      const right = this.parseProduct();
      if (!right) return null;
      left = {
        kind: 'Binary',
        op: opTok.type === 'Plus' ? 'Add' : 'Sub',
        left,
        right,
        span: mergeSpans(left.span, right.span),
      };
    }
    return left;
  }

  private parseProduct(): Expr | null {
    let left = this.parseUnary();
    if (!left) return null;

    while (this.at('Star') || this.at('Slash')) {
      const opTok = this.advance();
      const right = this.parseUnary();
      if (!right) return null;
      left = {
        kind: 'Binary',
        op: opTok.type === 'Star' ? 'Mul' : 'Div',
        left,
        right,
        span: mergeSpans(left.span, right.span),
      };
    }
    return left;
  }

  private parseUnary(): Expr | null {
    if (this.at('Minus')) {
      const minus = this.advance();
      const inner = this.parseUnary();
      if (!inner) return null;
      return { kind: 'Neg', operand: inner, span: mergeSpans(minus.span, inner.span) };
    }
    return this.parsePower();
  }

  private parsePower(): Expr | null {
    let left = this.parsePostfix();
    if (!left) return null;
    while (this.at('Caret')) {
      this.advance();
      const right = this.parsePostfix();
      if (!right) return null;
      left = { kind: 'Binary', op: 'Pow', left, right, span: mergeSpans(left.span, right.span) };
    }
    return left;
  }

  private parsePostfix(): Expr | null {
    let expr = this.parseAtom();
    if (!expr) return null;

    while (this.at('Dot')) {
      this.advance();
      const fieldTok = this.expect('Ident', 'field name');
      if (!fieldTok) return null;
      expr = {
        kind: 'MemberAccess',
        object: expr,
        field: fieldTok.lexeme,
        fieldSpan: fieldTok.span,
        span: mergeSpans(expr.span, fieldTok.span),
      };
    }
    return expr;
  }

  private parseAtom(): Expr | null {
    const t = this.peek();
    if (!t) {
      this.diagnostics.push(
        makeError('E0102', 'Syntax error: unexpected end of input', this.eofSpan(), 'expected expression'),
      );
      return null;
    }

    if (t.type === 'KwVec3') {
      const kw = this.advance();
      if (!this.expect('LParen', "'('")) return null;
      const x = this.parseExpr();
      if (!x || !this.expect('Comma', "','")) return null;
      const y = this.parseExpr();
      if (!y || !this.expect('Comma', "','")) return null;
      const z = this.parseExpr();
      if (!z) return null;
      const rp = this.expect('RParen', "')'");
      return { kind: 'Vec3Lit', x, y, z, span: mergeSpans(kw.span, rp?.span ?? z.span) };
    }

    if (t.type === 'Ident' && this.at('LParen', 1)) {
      const fnTok = this.advance();
      this.advance(); // '('
      const args: Expr[] = [];
      while (this.pos < this.tokens.length && !this.at('RParen')) {
        const arg = this.parseExpr();
        if (!arg) return null;
        args.push(arg);
        if (this.at('Comma')) this.advance();
        else break;
      }
      const rp = this.expect('RParen', "')'");
      return {
        kind: 'Call',
        callee: fnTok.lexeme,
        calleeSpan: fnTok.span,
        args,
        span: mergeSpans(fnTok.span, rp?.span ?? fnTok.span),
      };
    }

    if (t.type === 'Number') {
      const numTok = this.advance();
      const value = Number(numTok.lexeme);
      let unit: UnitExpr | null = null;
      if (this.at('Ident')) {
        unit = this.parseUnitExpr(false);
      }
      return {
        kind: 'Quantity',
        value,
        unit,
        span: unit ? mergeSpans(numTok.span, unit.span) : numTok.span,
      };
    }

    if (t.type === 'Ident') {
      this.advance();
      return { kind: 'Ident', name: t.lexeme, span: t.span };
    }

    if (t.type === 'LParen') {
      this.advance();
      const inner = this.parseExpr();
      if (!inner) return null;
      this.expect('RParen', "')'");
      return inner;
    }

    this.diagnostics.push(
      makeError('E0102', `Syntax error: unexpected token '${t.lexeme}'`, t.span, 'unexpected token'),
    );
    return null;
  }
}

export function parse(
  tokens: Token[],
  eofOffset: number,
): { program: Program; diagnostics: Diagnostic[] } {
  const parser = new Parser(tokens, eofOffset);
  const program = parser.parseProgram();
  return { program, diagnostics: parser.diagnostics };
}
