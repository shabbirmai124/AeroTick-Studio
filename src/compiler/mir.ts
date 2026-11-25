import { BinOp } from './ast';
import { HirExpr, HirProgram, HirStmt, ValueType } from './sema';

export type MirType = 'f64' | 'vec3_simd4' | 'quat_simd4' | 'mat4_simd16' | 'bool';

export interface SoAFieldLayout {
  name: string;
  mirType: MirType;
  lanes: number;
  activeLanes: number;
  descriptorOffsetBytes: number;
  bytesPerEntity: number;
  alignmentBytes: number;
  paddingBytesPerEntity: number;
}

export interface SoAStateLayout {
  stateName: string;
  fields: SoAFieldLayout[];
  totalBytesPerEntity: number;
  totalPaddingBytesPerEntity: number;
}

export type MirInst =
  | { kind: 'const_f64'; dst: string; value: number }
  | { kind: 'const_vec3'; dst: string; x: string; y: string; z: string }
  | { kind: 'load_param'; dst: string; name: string }
  | { kind: 'load_soa'; dst: string; state: string; field: string; isNext: boolean; mirType: MirType }
  | { kind: 'store_soa'; state: string; field: string; src: string; mirType: MirType }
  | { kind: 'vec_extract'; dst: string; vec: string; lane: number }
  | { kind: 'unary_neg'; dst: string; src: string; mirType: MirType }
  | { kind: 'bin_op'; dst: string; op: BinOp; lhs: string; rhs: string; mirType: MirType }
  | { kind: 'call'; dst: string; intrinsic: string; args: string[]; mirType: MirType }
  | { kind: 'branch'; cond: string; thenInsts: MirInst[]; elseInsts: MirInst[] | null };

export interface MirSystem {
  name: string;
  dtParam: string;
  stateName: string;
  insts: MirInst[];
}

export interface MirProgram {
  layouts: SoAStateLayout[];
  systems: MirSystem[];
}

function inferMirType(ty: ValueType): MirType {
  switch (ty.kind) {
    case 'Vec3':
      return 'vec3_simd4';
    case 'Quat':
      return 'quat_simd4';
    case 'Mat4':
      return 'mat4_simd16';
    case 'Bool':
      return 'bool';
    default:
      return 'f64';
  }
}

export function lowerToMir(hir: HirProgram): MirProgram {
  const layouts: SoAStateLayout[] = hir.states.map((st) => {
    const fields: SoAFieldLayout[] = [];
    let descOff = 0;
    let totalBytes = 0;
    let totalPad = 0;

    for (const f of st.fields) {
      const mirType = inferMirType(f.ty);
      let lanes = 1;
      let activeLanes = 1;
      let alignmentBytes = 8;

      if (mirType === 'vec3_simd4') {
        lanes = 4;
        activeLanes = 3;
        alignmentBytes = 32;
      } else if (mirType === 'quat_simd4') {
        lanes = 4;
        activeLanes = 4;
        alignmentBytes = 32;
      } else if (mirType === 'mat4_simd16') {
        lanes = 16;
        activeLanes = 16;
        alignmentBytes = 32;
      }

      const bytesPerEntity = lanes * 8;
      const paddingBytesPerEntity = (lanes - activeLanes) * 8;

      fields.push({
        name: f.name,
        mirType,
        lanes,
        activeLanes,
        descriptorOffsetBytes: descOff,
        bytesPerEntity,
        alignmentBytes,
        paddingBytesPerEntity,
      });

      descOff += 8;
      totalBytes += bytesPerEntity;
      totalPad += paddingBytesPerEntity;
    }

    return {
      stateName: st.name,
      fields,
      totalBytesPerEntity: totalBytes,
      totalPaddingBytesPerEntity: totalPad,
    };
  });

  const constMap = new Map<string, number>();
  for (const c of hir.consts) {
    if (c.scalarValue !== null) constMap.set(c.name, c.scalarValue);
  }

  const systems: MirSystem[] = hir.systems.map((sys) => {
    let regCounter = 0;
    const nextReg = () => `%r${regCounter++}`;

    const lowerExpr = (
      e: HirExpr,
      out: MirInst[],
      locals: Map<string, string>,
    ): string => {
      switch (e.kind) {
        case 'Number': {
          const dst = nextReg();
          out.push({ kind: 'const_f64', dst, value: e.value });
          return dst;
        }
        case 'ConstRef': {
          const dst = nextReg();
          out.push({ kind: 'const_f64', dst, value: constMap.get(e.name) ?? 0 });
          return dst;
        }
        case 'LocalRef':
          return locals.get(e.name) ?? '%undef';
        case 'ParamRef': {
          const dst = nextReg();
          out.push({ kind: 'load_param', dst, name: e.name });
          return dst;
        }
        case 'StateFieldRead': {
          const dst = nextReg();
          out.push({
            kind: 'load_soa',
            dst,
            state: e.stateName,
            field: e.field,
            isNext: e.isNext,
            mirType: inferMirType(e.ty),
          });
          return dst;
        }
        case 'VecComponent': {
          const v = lowerExpr(e.vec, out, locals);
          const dst = nextReg();
          const lane = e.component === 'x' ? 0 : e.component === 'y' ? 1 : 2;
          out.push({ kind: 'vec_extract', dst, vec: v, lane });
          return dst;
        }
        case 'Vec3': {
          const rx = lowerExpr(e.x, out, locals);
          const ry = lowerExpr(e.y, out, locals);
          const rz = lowerExpr(e.z, out, locals);
          const dst = nextReg();
          out.push({ kind: 'const_vec3', dst, x: rx, y: ry, z: rz });
          return dst;
        }
        case 'UnaryNeg': {
          const src = lowerExpr(e.operand, out, locals);
          const dst = nextReg();
          out.push({ kind: 'unary_neg', dst, src, mirType: inferMirType(e.ty) });
          return dst;
        }
        case 'Binary': {
          const lhs = lowerExpr(e.left, out, locals);
          const rhs = lowerExpr(e.right, out, locals);
          const dst = nextReg();
          out.push({ kind: 'bin_op', dst, op: e.op, lhs, rhs, mirType: inferMirType(e.ty) });
          return dst;
        }
        case 'Call': {
          const args = e.args.map((a) => lowerExpr(a, out, locals));
          const dst = nextReg();
          out.push({ kind: 'call', dst, intrinsic: e.callee, args, mirType: inferMirType(e.ty) });
          return dst;
        }
      }
    };

    const lowerStmtsBlock = (
      stmts: HirStmt[],
      locals: Map<string, string>,
    ): MirInst[] => {
      const out: MirInst[] = [];
      for (const s of stmts) {
        if (s.kind === 'Let') {
          const r = lowerExpr(s.value, out, locals);
          locals.set(s.name, r);
        } else if (s.kind === 'AssignField') {
          const r = lowerExpr(s.value, out, locals);
          out.push({
            kind: 'store_soa',
            state: s.stateName,
            field: s.field,
            src: r,
            mirType: inferMirType(s.value.ty),
          });
        } else if (s.kind === 'If') {
          const c = lowerExpr(s.cond, out, locals);
          const thenInsts = lowerStmtsBlock(s.thenStmts, new Map(locals));
          const elseInsts = s.elseStmts ? lowerStmtsBlock(s.elseStmts, new Map(locals)) : null;
          out.push({ kind: 'branch', cond: c, thenInsts, elseInsts });
        } else if (s.kind === 'Expr') {
          lowerExpr(s.expr, out, locals);
        }
      }
      return out;
    };

    const insts = lowerStmtsBlock(sys.stmts, new Map());

    return {
      name: sys.name,
      dtParam: sys.dtParam,
      stateName: sys.stateName,
      insts,
    };
  });

  return { layouts, systems };
}

function dumpInsts(insts: MirInst[], indent: string, out: string[]): void {
  for (const inst of insts) {
    switch (inst.kind) {
      case 'const_f64':
        out.push(`${indent}${inst.dst} = const.f64 ${inst.value}`);
        break;
      case 'const_vec3':
        out.push(`${indent}${inst.dst} = simd4.pack <${inst.x}, ${inst.y}, ${inst.z}, 0.0>`);
        break;
      case 'load_param':
        out.push(`${indent}${inst.dst} = param.f64 %${inst.name}`);
        break;
      case 'load_soa':
        out.push(
          `${indent}${inst.dst} = soa.load.${inst.mirType} @${inst.isNext ? 'next' : 'prev'}.${inst.state}.${inst.field}[%i]`,
        );
        break;
      case 'store_soa':
        out.push(`${indent}soa.store.${inst.mirType} @next.${inst.state}.${inst.field}[%i], ${inst.src}`);
        break;
      case 'vec_extract':
        out.push(`${indent}${inst.dst} = simd4.extract ${inst.vec}, lane ${inst.lane}`);
        break;
      case 'unary_neg':
        out.push(`${indent}${inst.dst} = neg.${inst.mirType} ${inst.src}`);
        break;
      case 'bin_op':
        out.push(`${indent}${inst.dst} = ${inst.op}.${inst.mirType} ${inst.lhs}, ${inst.rhs}`);
        break;
      case 'call':
        out.push(`${indent}${inst.dst} = call @det_${inst.intrinsic}(${inst.args.join(', ')}) : ${inst.mirType}`);
        break;
      case 'branch':
        out.push(`${indent}if ${inst.cond} {`);
        dumpInsts(inst.thenInsts, `${indent}  `, out);
        if (inst.elseInsts) {
          out.push(`${indent}} else {`);
          dumpInsts(inst.elseInsts, `${indent}  `, out);
        }
        out.push(`${indent}}`);
        break;
    }
  }
}

export function formatMir(mir: MirProgram): string {
  const lines: string[] = ['; AeroTick Unit-Erased SoA MIR', ''];
  for (const layout of mir.layouts) {
    lines.push(
      `soa_layout ${layout.stateName} (${layout.totalBytesPerEntity} bytes/entity, ${layout.totalPaddingBytesPerEntity} padding bytes) {`,
    );
    for (const f of layout.fields) {
      lines.push(
        `  ${f.name.padEnd(10)} : ${f.mirType.padEnd(11)} | desc_off=${String(f.descriptorOffsetBytes).padStart(2)}B | align=${String(f.alignmentBytes).padStart(2)}B | stride=${String(f.bytesPerEntity).padStart(2)}B | pad=${f.paddingBytesPerEntity}B`,
      );
    }
    lines.push('}', '');
  }

  for (const sys of mir.systems) {
    lines.push(
      `kernel @${sys.name}(%prev: *const ${sys.stateName}_SoA, %next: *mut ${sys.stateName}_SoA, %count: usize, %${sys.dtParam}: f64) {`,
    );
    lines.push('  for %i in 0..%count {');
    dumpInsts(sys.insts, '    ', lines);
    lines.push('  }');
    lines.push('}', '');
  }

  return lines.join('\n');
}
