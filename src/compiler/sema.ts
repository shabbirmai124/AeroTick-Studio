import { BinOp, Expr, Program, Stmt, TypeExpr, UnitExpr } from './ast';
import { Diagnostic, makeError, makeWarning, Span } from './span';
import {
  DIM_DIMENSIONLESS,
  dimsEqual,
  DimVector,
  divDim,
  formatDim,
  lookupUnit,
  mulDim,
  powDim,
  sqrtDim,
} from './units';

export type ValueType =
  | { kind: 'Scalar'; dim: DimVector }
  | { kind: 'Vec3'; dim: DimVector }
  | { kind: 'Quat' }
  | { kind: 'Mat4' }
  | { kind: 'Bool' }
  | { kind: 'StateRef'; stateName: string; isNext: boolean };

export function formatValueType(vt: ValueType): string {
  switch (vt.kind) {
    case 'Scalar':
      return `Scalar<${formatDim(vt.dim)}>`;
    case 'Vec3':
      return `Vec3<${formatDim(vt.dim)}>`;
    case 'Quat':
      return 'Quat';
    case 'Mat4':
      return 'Mat4';
    case 'Bool':
      return 'Bool';
    case 'StateRef':
      return `${vt.isNext ? 'next' : 'prev'}:${vt.stateName}`;
  }
}

export function typesCompatible(a: ValueType, b: ValueType): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'Scalar' && b.kind === 'Scalar') return dimsEqual(a.dim, b.dim);
  if (a.kind === 'Vec3' && b.kind === 'Vec3') return dimsEqual(a.dim, b.dim);
  if (a.kind === 'Quat' || a.kind === 'Mat4' || a.kind === 'Bool') return true;
  return false;
}

export interface HirConst {
  name: string;
  ty: ValueType;
  scalarValue: number | null;
  span: Span;
}

export interface HirStateField {
  name: string;
  ty: ValueType;
  span: Span;
}

export interface HirState {
  name: string;
  fields: HirStateField[];
  span: Span;
}

export type HirExpr =
  | { kind: 'Number'; value: number; ty: ValueType; span: Span }
  | { kind: 'ConstRef'; name: string; ty: ValueType; span: Span }
  | { kind: 'LocalRef'; name: string; ty: ValueType; span: Span }
  | { kind: 'ParamRef'; name: string; ty: ValueType; span: Span }
  | { kind: 'StateFieldRead'; isNext: boolean; stateName: string; field: string; ty: ValueType; span: Span }
  | { kind: 'VecComponent'; vec: HirExpr; component: 'x' | 'y' | 'z'; ty: ValueType; span: Span }
  | { kind: 'Vec3'; x: HirExpr; y: HirExpr; z: HirExpr; ty: ValueType; span: Span }
  | { kind: 'Call'; callee: string; args: HirExpr[]; ty: ValueType; span: Span }
  | { kind: 'UnaryNeg'; operand: HirExpr; ty: ValueType; span: Span }
  | { kind: 'Binary'; op: BinOp; left: HirExpr; right: HirExpr; ty: ValueType; span: Span };

export type HirStmt =
  | { kind: 'Let'; name: string; ty: ValueType; value: HirExpr; span: Span }
  | { kind: 'AssignField'; stateName: string; field: string; value: HirExpr; span: Span }
  | { kind: 'If'; cond: HirExpr; thenStmts: HirStmt[]; elseStmts: HirStmt[] | null; span: Span }
  | { kind: 'Expr'; expr: HirExpr; span: Span };

export interface HirSystem {
  name: string;
  dtParam: string;
  prevName: string;
  nextName: string;
  stateName: string;
  stmts: HirStmt[];
  locals: Array<{ name: string; ty: ValueType; span: Span }>;
  span: Span;
}

export interface TypedBinding {
  name: string;
  ty: ValueType;
  span: Span;
}

export interface HirProgram {
  consts: HirConst[];
  states: HirState[];
  systems: HirSystem[];
  topBindings: TypedBinding[];
}

const FORBIDDEN_ALLOC = new Set([
  'alloc',
  'malloc',
  'calloc',
  'realloc',
  'new',
  'Box',
  'Vec',
  'push',
  'spawn',
]);

interface Scope {
  consts: Map<string, HirConst>;
  states: Map<string, HirState>;
  locals: Map<string, ValueType>;
  params: Map<string, ValueType>;
  inStep: boolean;
  systemName: string | null;
  prevName: string | null;
  nextName: string | null;
  stateName: string | null;
  assignedNextFields: Set<string>;
  systemLocals: Array<{ name: string; ty: ValueType; span: Span }>;
}

function resolveUnit(
  unit: UnitExpr,
  diagnostics: Diagnostic[],
): { dim: DimVector; scale: number } | null {
  switch (unit.kind) {
    case 'One':
      return { dim: DIM_DIMENSIONLESS, scale: 1 };
    case 'Symbol': {
      const hit = lookupUnit(unit.name);
      if (!hit) {
        diagnostics.push(
          makeError(
            'E0201',
            `Unknown physical unit '${unit.name}'`,
            unit.span,
            'unrecognized SI unit',
          ),
        );
        return null;
      }
      return hit;
    }
    case 'Mul': {
      const l = resolveUnit(unit.left, diagnostics);
      const r = resolveUnit(unit.right, diagnostics);
      if (!l || !r) return null;
      return { dim: mulDim(l.dim, r.dim), scale: l.scale * r.scale };
    }
    case 'Div': {
      const l = resolveUnit(unit.left, diagnostics);
      const r = resolveUnit(unit.right, diagnostics);
      if (!l || !r) return null;
      return { dim: divDim(l.dim, r.dim), scale: l.scale / r.scale };
    }
    case 'Pow': {
      const b = resolveUnit(unit.base, diagnostics);
      if (!b) return null;
      return { dim: powDim(b.dim, unit.exp), scale: Math.pow(b.scale, unit.exp) };
    }
  }
}

function resolveType(te: TypeExpr, diagnostics: Diagnostic[]): ValueType | null {
  switch (te.kind) {
    case 'ScalarUnit': {
      const u = resolveUnit(te.unit, diagnostics);
      return u ? { kind: 'Scalar', dim: u.dim } : null;
    }
    case 'Vec3': {
      const u = resolveUnit(te.unit, diagnostics);
      return u ? { kind: 'Vec3', dim: u.dim } : null;
    }
    case 'Quat':
      return { kind: 'Quat' };
    case 'Mat4':
      return { kind: 'Mat4' };
  }
}

function isZeroLiteral(expr: Expr): boolean {
  return expr.kind === 'Quantity' && expr.unit === null && expr.value === 0;
}

function checkExpr(expr: Expr, scope: Scope, diagnostics: Diagnostic[]): HirExpr | null {
  switch (expr.kind) {
    case 'Quantity': {
      if (!expr.unit) {
        return {
          kind: 'Number',
          value: expr.value,
          ty: { kind: 'Scalar', dim: DIM_DIMENSIONLESS },
          span: expr.span,
        };
      }
      const u = resolveUnit(expr.unit, diagnostics);
      if (!u) return null;
      return {
        kind: 'Number',
        value: expr.value * u.scale,
        ty: { kind: 'Scalar', dim: u.dim },
        span: expr.span,
      };
    }

    case 'Ident': {
      const localTy = scope.locals.get(expr.name);
      if (localTy) return { kind: 'LocalRef', name: expr.name, ty: localTy, span: expr.span };
      const paramTy = scope.params.get(expr.name);
      if (paramTy) return { kind: 'ParamRef', name: expr.name, ty: paramTy, span: expr.span };
      const c = scope.consts.get(expr.name);
      if (c) return { kind: 'ConstRef', name: expr.name, ty: c.ty, span: expr.span };
      diagnostics.push(
        makeError('E0203', `Undefined symbol '${expr.name}'`, expr.span, 'not found in scope'),
      );
      return null;
    }

    case 'MemberAccess': {
      const obj = checkExpr(expr.object, scope, diagnostics);
      if (!obj) return null;
      if (obj.ty.kind === 'StateRef') {
        const stateDecl = scope.states.get(obj.ty.stateName);
        if (!stateDecl) return null;
        const sf = stateDecl.fields.find((f) => f.name === expr.field);
        if (!sf) {
          diagnostics.push(
            makeError(
              'E0204',
              `Unknown field '${expr.field}' on state '${stateDecl.name}'`,
              expr.fieldSpan,
              'unknown field',
            ),
          );
          return null;
        }
        if (obj.ty.isNext && !scope.assignedNextFields.has(expr.field)) {
          diagnostics.push(
            makeError(
              'E0302',
              `Read from uninitialized next state field '${expr.field}'`,
              expr.span,
              'read before assignment in `next`',
            ),
          );
        }
        return {
          kind: 'StateFieldRead',
          isNext: obj.ty.isNext,
          stateName: stateDecl.name,
          field: expr.field,
          ty: sf.ty,
          span: expr.span,
        };
      }

      if (obj.ty.kind === 'Vec3') {
        if (expr.field !== 'x' && expr.field !== 'y' && expr.field !== 'z') {
          diagnostics.push(
            makeError(
              'E0205',
              `Unknown vector component '.${expr.field}'`,
              expr.fieldSpan,
              'expected x, y, or z',
            ),
          );
          return null;
        }
        return {
          kind: 'VecComponent',
          vec: obj,
          component: expr.field,
          ty: { kind: 'Scalar', dim: obj.ty.dim },
          span: expr.span,
        };
      }

      diagnostics.push(
        makeError(
          'E0206',
          `Cannot access '.${expr.field}' on ${formatValueType(obj.ty)}`,
          expr.fieldSpan,
          'invalid member access',
        ),
      );
      return null;
    }

    case 'Vec3Lit': {
      const hx = checkExpr(expr.x, scope, diagnostics);
      const hy = checkExpr(expr.y, scope, diagnostics);
      const hz = checkExpr(expr.z, scope, diagnostics);
      if (!hx || !hy || !hz) return null;
      if (hx.ty.kind !== 'Scalar' || hy.ty.kind !== 'Scalar' || hz.ty.kind !== 'Scalar') {
        diagnostics.push(
          makeError(
            'E0210',
            'Vec3 components must be scalar quantities',
            expr.span,
            'non-scalar Vec3 component',
          ),
        );
        return null;
      }

      let targetDim = hx.ty.dim;
      if (isZeroLiteral(expr.x) && !dimsEqual(hy.ty.dim, DIM_DIMENSIONLESS)) {
        targetDim = hy.ty.dim;
      } else if (isZeroLiteral(expr.x) && !dimsEqual(hz.ty.dim, DIM_DIMENSIONLESS)) {
        targetDim = hz.ty.dim;
      }

      const items: Array<{ d: DimVector; ast: Expr }> = [
        { d: hx.ty.dim, ast: expr.x },
        { d: hy.ty.dim, ast: expr.y },
        { d: hz.ty.dim, ast: expr.z },
      ];
      for (const item of items) {
        if (!dimsEqual(item.d, targetDim) && !isZeroLiteral(item.ast)) {
          diagnostics.push(
            makeError(
              'E0211',
              `Inconsistent physical units across Vec3 components: expected ${formatDim(targetDim)}, found ${formatDim(item.d)}`,
              item.ast.span,
              `has unit ${formatDim(item.d)}`,
            ),
          );
          return null;
        }
      }

      return {
        kind: 'Vec3',
        x: hx,
        y: hy,
        z: hz,
        ty: { kind: 'Vec3', dim: targetDim },
        span: expr.span,
      };
    }

    case 'Call': {
      if (scope.inStep && FORBIDDEN_ALLOC.has(expr.callee)) {
        diagnostics.push(
          makeError(
            'E0303',
            `Heap allocation '${expr.callee}(...)' is forbidden inside step`,
            expr.calleeSpan,
            'heap allocation prohibited in hot loop',
          ),
        );
        return null;
      }
      if (scope.inStep && scope.systemName === expr.callee) {
        diagnostics.push(
          makeError(
            'E0304',
            `Recursion into '${expr.callee}' is forbidden inside step`,
            expr.calleeSpan,
            'recursion prohibited',
          ),
        );
        return null;
      }

      const hArgs: HirExpr[] = [];
      for (const a of expr.args) {
        const ha = checkExpr(a, scope, diagnostics);
        if (!ha) return null;
        hArgs.push(ha);
      }

      let retTy: ValueType | null = null;
      if (expr.callee === 'dot' && hArgs.length === 2) {
        if (hArgs[0].ty.kind === 'Vec3' && hArgs[1].ty.kind === 'Vec3') {
          retTy = { kind: 'Scalar', dim: mulDim(hArgs[0].ty.dim, hArgs[1].ty.dim) };
        }
      } else if (expr.callee === 'cross' && hArgs.length === 2) {
        if (hArgs[0].ty.kind === 'Vec3' && hArgs[1].ty.kind === 'Vec3') {
          retTy = { kind: 'Vec3', dim: mulDim(hArgs[0].ty.dim, hArgs[1].ty.dim) };
        }
      } else if (expr.callee === 'norm' && hArgs.length === 1) {
        if (hArgs[0].ty.kind === 'Vec3') {
          retTy = { kind: 'Scalar', dim: hArgs[0].ty.dim };
        }
      } else if (expr.callee === 'normalize' && hArgs.length === 1) {
        if (hArgs[0].ty.kind === 'Vec3') {
          retTy = { kind: 'Vec3', dim: DIM_DIMENSIONLESS };
        }
      } else if (expr.callee === 'sqrt' && hArgs.length === 1) {
        if (hArgs[0].ty.kind === 'Scalar') {
          const sq = sqrtDim(hArgs[0].ty.dim);
          if (!sq) {
            diagnostics.push(
              makeError(
                'E0221',
                `Cannot take sqrt of odd unit ${formatDim(hArgs[0].ty.dim)}`,
                expr.span,
                'odd unit exponent',
              ),
            );
            return null;
          }
          retTy = { kind: 'Scalar', dim: sq };
        }
      } else if (
        (expr.callee === 'sin' || expr.callee === 'cos' || expr.callee === 'tan') &&
        hArgs.length === 1
      ) {
        if (
          hArgs[0].ty.kind === 'Scalar' &&
          hArgs[0].ty.dim.si.every((e) => e === 0) &&
          (hArgs[0].ty.dim.angle === 0 || hArgs[0].ty.dim.angle === 1)
        ) {
          retTy = { kind: 'Scalar', dim: DIM_DIMENSIONLESS };
        } else {
          diagnostics.push(
            makeError(
              'E0222',
              `\`${expr.callee}\` requires rad or dimensionless angle`,
              expr.span,
              'invalid angle unit',
            ),
          );
          return null;
        }
      } else if (expr.callee === 'abs' && hArgs.length === 1) {
        retTy = hArgs[0].ty;
      } else if (
        (expr.callee === 'min' || expr.callee === 'max') &&
        hArgs.length === 2 &&
        typesCompatible(hArgs[0].ty, hArgs[1].ty)
      ) {
        retTy = hArgs[0].ty;
      } else if (
        expr.callee === 'select' &&
        hArgs.length === 3 &&
        hArgs[0].ty.kind === 'Bool' &&
        typesCompatible(hArgs[1].ty, hArgs[2].ty)
      ) {
        retTy = hArgs[1].ty;
      }

      if (!retTy) {
        diagnostics.push(
          makeError(
            'E0223',
            `Invalid call to '${expr.callee}'`,
            expr.calleeSpan,
            'unknown or ill-typed intrinsic',
          ),
        );
        return null;
      }

      return { kind: 'Call', callee: expr.callee, args: hArgs, ty: retTy, span: expr.span };
    }

    case 'Neg': {
      const inner = checkExpr(expr.operand, scope, diagnostics);
      if (!inner) return null;
      return { kind: 'UnaryNeg', operand: inner, ty: inner.ty, span: expr.span };
    }

    case 'Binary': {
      const hl = checkExpr(expr.left, scope, diagnostics);
      const hr = checkExpr(expr.right, scope, diagnostics);
      if (!hl || !hr) return null;
      const lt = hl.ty;
      const rt = hr.ty;

      let ty: ValueType | null = null;
      if (expr.op === 'Add' || expr.op === 'Sub') {
        if (
          (lt.kind === 'Scalar' && rt.kind === 'Scalar' && dimsEqual(lt.dim, rt.dim)) ||
          (lt.kind === 'Vec3' && rt.kind === 'Vec3' && dimsEqual(lt.dim, rt.dim))
        ) {
          ty = lt;
        } else {
          diagnostics.push(
            makeError(
              'E0230',
              `Physical unit mismatch: cannot apply \`${expr.op}\` to \`${formatValueType(lt)}\` and \`${formatValueType(rt)}\``,
              expr.span,
              `incompatible dimensions: ${formatValueType(lt)} vs ${formatValueType(rt)}`,
              [
                { span: expr.left.span, message: `has type ${formatValueType(lt)}` },
                { span: expr.right.span, message: `has type ${formatValueType(rt)}` },
              ],
              'Only identical physical dimensions may be added or subtracted.',
            ),
          );
          return null;
        }
      } else if (expr.op === 'Mul') {
        if (lt.kind === 'Scalar' && rt.kind === 'Scalar') {
          ty = { kind: 'Scalar', dim: mulDim(lt.dim, rt.dim) };
        } else if (lt.kind === 'Vec3' && rt.kind === 'Scalar') {
          ty = { kind: 'Vec3', dim: mulDim(lt.dim, rt.dim) };
        } else if (lt.kind === 'Scalar' && rt.kind === 'Vec3') {
          ty = { kind: 'Vec3', dim: mulDim(rt.dim, lt.dim) };
        } else {
          diagnostics.push(
            makeError(
              'E0233',
              `Unsupported \`*\` between ${formatValueType(lt)} and ${formatValueType(rt)}`,
              expr.span,
              'use dot() or cross() for Vec3 * Vec3',
            ),
          );
          return null;
        }
      } else if (expr.op === 'Div') {
        if (expr.right.kind === 'Quantity' && expr.right.value === 0) {
          diagnostics.push(
            makeError(
              'E0237',
              'Division by zero physical quantity',
              expr.right.span,
              'divisor is zero',
              [],
              'Ensure divisors in deterministic physics kernels are non-zero.',
            ),
          );
          return null;
        }
        if (lt.kind === 'Scalar' && rt.kind === 'Scalar') {
          ty = { kind: 'Scalar', dim: divDim(lt.dim, rt.dim) };
        } else if (lt.kind === 'Vec3' && rt.kind === 'Scalar') {
          ty = { kind: 'Vec3', dim: divDim(lt.dim, rt.dim) };
        } else {
          diagnostics.push(
            makeError(
              'E0234',
              `Unsupported \`/\` between ${formatValueType(lt)} and ${formatValueType(rt)}`,
              expr.span,
              'invalid division',
            ),
          );
          return null;
        }
      } else if (expr.op === 'Pow') {
        if (
          lt.kind === 'Scalar' &&
          expr.right.kind === 'Quantity' &&
          expr.right.unit === null &&
          Number.isInteger(expr.right.value) &&
          Math.abs(expr.right.value) <= 127
        ) {
          ty = { kind: 'Scalar', dim: powDim(lt.dim, expr.right.value) };
        } else {
          diagnostics.push(
            makeError(
              'E0235',
              'Power operator requires scalar base and integer exponent',
              expr.span,
              'invalid exponent',
            ),
          );
          return null;
        }
      } else {
        if (lt.kind === 'Scalar' && rt.kind === 'Scalar' && dimsEqual(lt.dim, rt.dim)) {
          ty = { kind: 'Bool' };
        } else {
          diagnostics.push(
            makeError(
              'E0236',
              `Cannot compare ${formatValueType(lt)} with ${formatValueType(rt)}`,
              expr.span,
              'unit mismatch in comparison',
            ),
          );
          return null;
        }
      }

      return { kind: 'Binary', op: expr.op, left: hl, right: hr, ty, span: expr.span };
    }
  }
}

function checkStmt(stmt: Stmt, scope: Scope, diagnostics: Diagnostic[]): HirStmt | null {
  switch (stmt.kind) {
    case 'ForbiddenLoop':
      diagnostics.push(
        makeError(
          'E0305',
          `Unbounded '${stmt.keyword}' loop forbidden inside step`,
          stmt.span,
          'unbounded loop violates real-time determinism',
        ),
      );
      return null;

    case 'Let': {
      const val = checkExpr(stmt.value, scope, diagnostics);
      if (!val) return null;
      let ty = val.ty;
      if (stmt.typeAnn) {
        const expected = resolveType(stmt.typeAnn, diagnostics);
        if (!expected) return null;
        if (!typesCompatible(expected, val.ty)) {
          diagnostics.push(
            makeError(
              'E0240',
              `Type mismatch in 'let ${stmt.name}'`,
              stmt.value.span,
              `deduced ${formatValueType(val.ty)}`,
              [{ span: stmt.typeAnn.span, message: `expected ${formatValueType(expected)}` }],
            ),
          );
        }
        ty = expected;
      }
      scope.locals.set(stmt.name, ty);
      scope.systemLocals.push({ name: stmt.name, ty, span: stmt.nameSpan });
      return { kind: 'Let', name: stmt.name, ty, value: val, span: stmt.span };
    }

    case 'Assign': {
      if (stmt.target.kind === 'MemberAccess' && stmt.target.object.kind === 'Ident') {
        const root = stmt.target.object.name;
        const targetField = stmt.target.field;
        const fieldSpan = stmt.target.fieldSpan;

        if (scope.prevName === root) {
          diagnostics.push(
            makeError(
              'E0301',
              `Cannot mutate immutable prev state '${root}.${targetField}'`,
              stmt.target.span,
              'cannot assign to `prev`',
              [],
              `Write result to \`${scope.nextName ?? 'next'}.${targetField}\` instead.`,
            ),
          );
          checkExpr(stmt.value, scope, diagnostics);
          return null;
        }

        if (scope.nextName === root && scope.stateName) {
          const st = scope.states.get(scope.stateName);
          if (st) {
            const sf = st.fields.find((f) => f.name === targetField);
            if (!sf) {
              diagnostics.push(
                makeError(
                  'E0204',
                  `Unknown field '${targetField}' on state '${st.name}'`,
                  fieldSpan,
                  'unknown state field',
                ),
              );
              return null;
            }
            const val = checkExpr(stmt.value, scope, diagnostics);
            if (!val) return null;
            if (!typesCompatible(sf.ty, val.ty)) {
              diagnostics.push(
                makeError(
                  'E0241',
                  `Unit mismatch assigning to '${root}.${targetField}'`,
                  stmt.value.span,
                  `deduced ${formatValueType(val.ty)}`,
                  [{ span: sf.span, message: `declared as ${formatValueType(sf.ty)}` }],
                ),
              );
            }
            scope.assignedNextFields.add(targetField);
            return {
              kind: 'AssignField',
              stateName: st.name,
              field: targetField,
              value: val,
              span: stmt.span,
            };
          }
        }
      }

      diagnostics.push(
        makeError(
          'E0306',
          'Invalid assignment target inside step',
          stmt.target.span,
          'expected `next.<field>`',
        ),
      );
      return null;
    }

    case 'If': {
      const condH = checkExpr(stmt.cond, scope, diagnostics);
      if (!condH) return null;
      if (condH.ty.kind !== 'Bool') {
        diagnostics.push(
          makeError(
            'E0242',
            'If condition must be a boolean comparison',
            stmt.cond.span,
            `found ${formatValueType(condH.ty)}`,
          ),
        );
      }
      const thenStmts: HirStmt[] = [];
      for (const s of stmt.thenBranch) {
        const hs = checkStmt(s, scope, diagnostics);
        if (hs) thenStmts.push(hs);
      }
      let elseStmts: HirStmt[] | null = null;
      if (stmt.elseBranch) {
        elseStmts = [];
        for (const s of stmt.elseBranch) {
          const hs = checkStmt(s, scope, diagnostics);
          if (hs) elseStmts.push(hs);
        }
      }
      return { kind: 'If', cond: condH, thenStmts, elseStmts, span: stmt.span };
    }

    case 'Expr': {
      const e = checkExpr(stmt.expr, scope, diagnostics);
      return e ? { kind: 'Expr', expr: e, span: stmt.span } : null;
    }
  }
}

export function analyze(program: Program): { hir: HirProgram; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  const constsMap = new Map<string, HirConst>();
  const constsOrder: HirConst[] = [];
  const statesMap = new Map<string, HirState>();
  const statesOrder: HirState[] = [];
  const systems: HirSystem[] = [];
  const topBindings: TypedBinding[] = [];
  const topLocals = new Map<string, ValueType>();

  // Pass 1: const & state definitions
  for (const item of program.items) {
    if (item.kind === 'Const') {
      const c = item.decl;
      const scope: Scope = {
        consts: constsMap,
        states: statesMap,
        locals: new Map(topLocals),
        params: new Map(),
        inStep: false,
        systemName: null,
        prevName: null,
        nextName: null,
        stateName: null,
        assignedNextFields: new Set(),
        systemLocals: [],
      };
      const val = checkExpr(c.value, scope, diagnostics);
      if (val) {
        let finalTy = val.ty;
        if (c.typeAnn) {
          const expected = resolveType(c.typeAnn, diagnostics);
          if (expected) {
            if (!typesCompatible(expected, val.ty)) {
              diagnostics.push(
                makeError(
                  'E0240',
                  `Unit mismatch in constant '${c.name}'`,
                  c.value.span,
                  `deduced ${formatValueType(val.ty)}`,
                  [{ span: c.typeAnn.span, message: `annotated as ${formatValueType(expected)}` }],
                ),
              );
            }
            finalTy = expected;
          }
        }
        const hc: HirConst = {
          name: c.name,
          ty: finalTy,
          scalarValue: val.kind === 'Number' ? val.value : null,
          span: c.span,
        };
        constsMap.set(c.name, hc);
        constsOrder.push(hc);
      }
    } else if (item.kind === 'State') {
      const st = item.decl;
      if (statesMap.has(st.name)) {
        diagnostics.push(
          makeError('E0250', `Duplicate state '${st.name}'`, st.nameSpan, 'already defined'),
        );
        continue;
      }
      const fields: HirStateField[] = [];
      const seen = new Set<string>();
      for (const f of st.fields) {
        if (seen.has(f.name)) {
          diagnostics.push(
            makeError(
              'E0251',
              `Duplicate field '${f.name}' in state '${st.name}'`,
              f.nameSpan,
              'duplicate field',
            ),
          );
          continue;
        }
        seen.add(f.name);
        const ty = resolveType(f.typeExpr, diagnostics);
        if (ty) fields.push({ name: f.name, ty, span: f.span });
      }
      const hs: HirState = { name: st.name, fields, span: st.span };
      statesMap.set(st.name, hs);
      statesOrder.push(hs);
    }
  }

  // Pass 2: systems and top-level bindings
  for (const item of program.items) {
    if (item.kind === 'TopLet') {
      const scope: Scope = {
        consts: constsMap,
        states: statesMap,
        locals: new Map(topLocals),
        params: new Map(),
        inStep: false,
        systemName: null,
        prevName: null,
        nextName: null,
        stateName: null,
        assignedNextFields: new Set(),
        systemLocals: [],
      };
      const val = checkExpr(item.value, scope, diagnostics);
      if (val) {
        let finalTy = val.ty;
        if (item.typeAnn) {
          const expected = resolveType(item.typeAnn, diagnostics);
          if (expected) {
            if (!typesCompatible(expected, val.ty)) {
              diagnostics.push(
                makeError(
                  'E0240',
                  `Unit mismatch in binding '${item.name}': expected ${formatValueType(expected)}, found ${formatValueType(val.ty)}`,
                  item.value.span,
                  `deduced ${formatValueType(val.ty)}`,
                  [{ span: item.typeAnn.span, message: `annotated as ${formatValueType(expected)}` }],
                ),
              );
            }
            finalTy = expected;
          }
        }
        topLocals.set(item.name, finalTy);
        topBindings.push({ name: item.name, ty: finalTy, span: item.nameSpan });
      }
    } else if (item.kind === 'TopExpr') {
      const scope: Scope = {
        consts: constsMap,
        states: statesMap,
        locals: new Map(topLocals),
        params: new Map(),
        inStep: false,
        systemName: null,
        prevName: null,
        nextName: null,
        stateName: null,
        assignedNextFields: new Set(),
        systemLocals: [],
      };
      checkExpr(item.expr, scope, diagnostics);
    } else if (item.kind === 'System') {
      const sys = item.decl;
      const params = new Map<string, ValueType>();
      let dtParam = 'dt';
      for (const p of sys.params) {
        const pty = resolveType(p.typeExpr, diagnostics);
        if (pty) {
          params.set(p.name, pty);
          dtParam = p.name;
        }
      }
      const prevState = statesMap.get(sys.step.prevType);
      if (!prevState) {
        diagnostics.push(
          makeError(
            'E0261',
            `Unknown state '${sys.step.prevType}'`,
            sys.step.prevTypeSpan,
            'unknown state',
          ),
        );
        continue;
      }
      if (sys.step.prevType !== sys.step.nextType) {
        diagnostics.push(
          makeError(
            'E0262',
            'Double-buffered step states must share the same type',
            sys.step.nextTypeSpan,
            `expected '${sys.step.prevType}'`,
            [{ span: sys.step.prevTypeSpan, message: `prev is '${sys.step.prevType}'` }],
          ),
        );
      }

      params.set(sys.step.prevName, { kind: 'StateRef', stateName: prevState.name, isNext: false });
      params.set(sys.step.nextName, { kind: 'StateRef', stateName: prevState.name, isNext: true });

      const scope: Scope = {
        consts: constsMap,
        states: statesMap,
        locals: new Map(),
        params,
        inStep: true,
        systemName: sys.name,
        prevName: sys.step.prevName,
        nextName: sys.step.nextName,
        stateName: prevState.name,
        assignedNextFields: new Set(),
        systemLocals: [],
      };

      const hirStmts: HirStmt[] = [];
      for (const s of sys.step.stmts) {
        const hs = checkStmt(s, scope, diagnostics);
        if (hs) hirStmts.push(hs);
      }

      for (const f of prevState.fields) {
        if (!scope.assignedNextFields.has(f.name)) {
          diagnostics.push(
            makeWarning(
              'W0301',
              `Output field '${sys.step.nextName}.${f.name}' is never written in step`,
              sys.step.span,
              `missing assignment to \`${sys.step.nextName}.${f.name}\``,
              `Add \`${sys.step.nextName}.${f.name} = ${sys.step.prevName}.${f.name};\` to preserve it across ticks.`,
            ),
          );
        }
      }

      systems.push({
        name: sys.name,
        dtParam,
        prevName: sys.step.prevName,
        nextName: sys.step.nextName,
        stateName: prevState.name,
        stmts: hirStmts,
        locals: scope.systemLocals,
        span: sys.span,
      });
    }
  }

  return {
    hir: { consts: constsOrder, states: statesOrder, systems, topBindings },
    diagnostics,
  };
}
