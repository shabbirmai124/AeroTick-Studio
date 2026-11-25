import { Span } from './span';

export type BinOp = 'Add' | 'Sub' | 'Mul' | 'Div' | 'Pow' | 'Eq' | 'Ne' | 'Lt' | 'Le' | 'Gt' | 'Ge';

export type UnitExpr =
  | { kind: 'Symbol'; name: string; span: Span }
  | { kind: 'One'; span: Span }
  | { kind: 'Mul'; left: UnitExpr; right: UnitExpr; span: Span }
  | { kind: 'Div'; left: UnitExpr; right: UnitExpr; span: Span }
  | { kind: 'Pow'; base: UnitExpr; exp: number; span: Span };

export type TypeExpr =
  | { kind: 'ScalarUnit'; unit: UnitExpr; span: Span }
  | { kind: 'Vec3'; unit: UnitExpr; span: Span }
  | { kind: 'Quat'; span: Span }
  | { kind: 'Mat4'; span: Span };

export type Expr =
  | { kind: 'Quantity'; value: number; unit: UnitExpr | null; span: Span }
  | { kind: 'Ident'; name: string; span: Span }
  | { kind: 'MemberAccess'; object: Expr; field: string; fieldSpan: Span; span: Span }
  | { kind: 'Vec3Lit'; x: Expr; y: Expr; z: Expr; span: Span }
  | { kind: 'Call'; callee: string; calleeSpan: Span; args: Expr[]; span: Span }
  | { kind: 'Neg'; operand: Expr; span: Span }
  | { kind: 'Binary'; op: BinOp; left: Expr; right: Expr; span: Span };

export type Stmt =
  | { kind: 'Let'; name: string; nameSpan: Span; typeAnn: TypeExpr | null; value: Expr; span: Span }
  | { kind: 'Assign'; target: Expr; value: Expr; span: Span }
  | { kind: 'If'; cond: Expr; thenBranch: Stmt[]; elseBranch: Stmt[] | null; span: Span }
  | { kind: 'ForbiddenLoop'; keyword: string; span: Span }
  | { kind: 'Expr'; expr: Expr; span: Span };

export interface StateField {
  name: string;
  nameSpan: Span;
  typeExpr: TypeExpr;
  span: Span;
}

export interface StateDecl {
  name: string;
  nameSpan: Span;
  fields: StateField[];
  span: Span;
}

export interface ConstDecl {
  name: string;
  nameSpan: Span;
  typeAnn: TypeExpr | null;
  value: Expr;
  span: Span;
}

export interface Param {
  name: string;
  nameSpan: Span;
  typeExpr: TypeExpr;
  span: Span;
}

export interface StepBlock {
  prevName: string;
  prevType: string;
  prevTypeSpan: Span;
  nextName: string;
  nextType: string;
  nextTypeSpan: Span;
  stmts: Stmt[];
  span: Span;
}

export interface SystemDecl {
  name: string;
  nameSpan: Span;
  params: Param[];
  step: StepBlock;
  span: Span;
}

export type Item =
  | { kind: 'Const'; decl: ConstDecl }
  | { kind: 'State'; decl: StateDecl }
  | { kind: 'System'; decl: SystemDecl }
  | { kind: 'TopLet'; name: string; nameSpan: Span; typeAnn: TypeExpr | null; value: Expr; span: Span }
  | { kind: 'TopExpr'; expr: Expr; span: Span };

export interface Program {
  items: Item[];
}

export function formatUnitExpr(u: UnitExpr): string {
  switch (u.kind) {
    case 'One':
      return '1';
    case 'Symbol':
      return u.name;
    case 'Mul':
      return `${formatUnitExpr(u.left)}*${formatUnitExpr(u.right)}`;
    case 'Div':
      return `${formatUnitExpr(u.left)}/${formatUnitExpr(u.right)}`;
    case 'Pow':
      return `${formatUnitExpr(u.base)}^${u.exp}`;
  }
}

export function formatTypeExpr(t: TypeExpr): string {
  switch (t.kind) {
    case 'ScalarUnit':
      return formatUnitExpr(t.unit);
    case 'Vec3':
      return `Vec3<${formatUnitExpr(t.unit)}>`;
    case 'Quat':
      return 'Quat';
    case 'Mat4':
      return 'Mat4';
  }
}

function formatExpr(e: Expr): string {
  switch (e.kind) {
    case 'Quantity':
      return e.unit ? `${e.value} ${formatUnitExpr(e.unit)}` : `${e.value}`;
    case 'Ident':
      return e.name;
    case 'MemberAccess':
      return `${formatExpr(e.object)}.${e.field}`;
    case 'Vec3Lit':
      return `Vec3(${formatExpr(e.x)}, ${formatExpr(e.y)}, ${formatExpr(e.z)})`;
    case 'Call':
      return `${e.callee}(${e.args.map(formatExpr).join(', ')})`;
    case 'Neg':
      return `-${formatExpr(e.operand)}`;
    case 'Binary':
      return `(${formatExpr(e.left)} ${e.op} ${formatExpr(e.right)})`;
  }
}

function formatStmt(s: Stmt, indent: string, out: string[]): void {
  switch (s.kind) {
    case 'Let':
      out.push(
        `${indent}Stmt::Let ${s.name}${s.typeAnn ? `: ${formatTypeExpr(s.typeAnn)}` : ''} = ${formatExpr(s.value)}`,
      );
      break;
    case 'Assign':
      out.push(`${indent}Stmt::Assign ${formatExpr(s.target)} = ${formatExpr(s.value)}`);
      break;
    case 'If':
      out.push(`${indent}Stmt::If ${formatExpr(s.cond)} {`);
      for (const child of s.thenBranch) formatStmt(child, `${indent}  `, out);
      if (s.elseBranch) {
        out.push(`${indent}} else {`);
        for (const child of s.elseBranch) formatStmt(child, `${indent}  `, out);
      }
      out.push(`${indent}}`);
      break;
    case 'ForbiddenLoop':
      out.push(`${indent}Stmt::ForbiddenLoop(${s.keyword})`);
      break;
    case 'Expr':
      out.push(`${indent}Stmt::Expr ${formatExpr(s.expr)}`);
      break;
  }
}

export function formatAst(program: Program): string {
  const out: string[] = ['Program {'];
  for (const item of program.items) {
    switch (item.kind) {
      case 'Const':
        out.push(
          `  Item::Const ${item.decl.name}${item.decl.typeAnn ? `: ${formatTypeExpr(item.decl.typeAnn)}` : ''} = ${formatExpr(item.decl.value)}`,
        );
        break;
      case 'State':
        out.push(`  Item::State ${item.decl.name} {`);
        for (const f of item.decl.fields) {
          out.push(`    ${f.name}: ${formatTypeExpr(f.typeExpr)}`);
        }
        out.push('  }');
        break;
      case 'System':
        out.push(
          `  Item::System ${item.decl.name}(${item.decl.params.map((p) => `${p.name}: ${formatTypeExpr(p.typeExpr)}`).join(', ')}) {`,
        );
        out.push(
          `    Step(${item.decl.step.prevName}: ${item.decl.step.prevType}) -> ${item.decl.step.nextName}: ${item.decl.step.nextType} {`,
        );
        for (const s of item.decl.step.stmts) {
          formatStmt(s, '      ', out);
        }
        out.push('    }');
        out.push('  }');
        break;
      case 'TopLet':
        out.push(
          `  Item::TopLet ${item.name}${item.typeAnn ? `: ${formatTypeExpr(item.typeAnn)}` : ''} = ${formatExpr(item.value)}`,
        );
        break;
      case 'TopExpr':
        out.push(`  Item::TopExpr ${formatExpr(item.expr)}`);
        break;
    }
  }
  out.push('}');
  return out.join('\n');
}
