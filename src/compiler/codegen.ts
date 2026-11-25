import { BinOp } from './ast';
import { MirInst, MirProgram, MirSystem, MirType } from './mir';

function llvmTy(mt: MirType): { ty: string; align: number } {
  if (mt === 'vec3_simd4' || mt === 'quat_simd4') {
    return { ty: '<4 x double>', align: 32 };
  }
  if (mt === 'bool') {
    return { ty: 'i1', align: 1 };
  }
  return { ty: 'double', align: 8 };
}

function formatFloatLit(v: number): string {
  if (Object.is(v, -0)) return '0.0';
  return Number.isInteger(v) ? `${v}.0` : String(v);
}

function emitLlvmInsts(
  insts: MirInst[],
  sys: MirSystem,
  mir: MirProgram,
  counter: { n: number },
  out: string[],
  regTypes: Map<string, MirType>,
): void {
  const layout = mir.layouts.find((l) => l.stateName === sys.stateName);
  const fieldIndex = (name: string) => {
    const idx = layout?.fields.findIndex((f) => f.name === name) ?? 0;
    return idx < 0 ? 0 : idx;
  };

  for (const inst of insts) {
    switch (inst.kind) {
      case 'const_f64':
        regTypes.set(inst.dst, 'f64');
        out.push(`  ${inst.dst} = fadd double 0.0, ${formatFloatLit(inst.value)}`);
        break;
      case 'const_vec3':
        regTypes.set(inst.dst, 'vec3_simd4');
        out.push(`  ${inst.dst}_0 = insertelement <4 x double> poison, double ${inst.x}, i32 0`);
        out.push(`  ${inst.dst}_1 = insertelement <4 x double> ${inst.dst}_0, double ${inst.y}, i32 1`);
        out.push(`  ${inst.dst}_2 = insertelement <4 x double> ${inst.dst}_1, double ${inst.z}, i32 2`);
        out.push(`  ${inst.dst} = insertelement <4 x double> ${inst.dst}_2, double 0.0, i32 3`);
        break;
      case 'load_param':
        regTypes.set(inst.dst, 'f64');
        out.push(`  ${inst.dst} = fadd double 0.0, %${inst.name}`);
        break;
      case 'load_soa': {
        regTypes.set(inst.dst, inst.mirType);
        const fIdx = fieldIndex(inst.field);
        const buf = inst.isNext ? '%next' : '%prev';
        const { ty, align } = llvmTy(inst.mirType);
        const c = counter.n++;
        out.push(`  %p_${c} = getelementptr inbounds %struct.${inst.state}_SoA, ptr ${buf}, i32 0, i32 ${fIdx}`);
        out.push(`  %b_${c} = load ptr, ptr %p_${c}, align 8`);
        out.push(`  %g_${c} = getelementptr inbounds ${ty}, ptr %b_${c}, i64 %idx`);
        out.push(`  ${inst.dst} = load ${ty}, ptr %g_${c}, align ${align}`);
        break;
      }
      case 'store_soa': {
        const fIdx = fieldIndex(inst.field);
        const { ty, align } = llvmTy(inst.mirType);
        const c = counter.n++;
        out.push(`  %p_${c} = getelementptr inbounds %struct.${inst.state}_SoA, ptr %next, i32 0, i32 ${fIdx}`);
        out.push(`  %b_${c} = load ptr, ptr %p_${c}, align 8`);
        out.push(`  %g_${c} = getelementptr inbounds ${ty}, ptr %b_${c}, i64 %idx`);
        out.push(`  store ${ty} ${inst.src}, ptr %g_${c}, align ${align}`);
        break;
      }
      case 'vec_extract':
        regTypes.set(inst.dst, 'f64');
        out.push(`  ${inst.dst} = extractelement <4 x double> ${inst.vec}, i32 ${inst.lane}`);
        break;
      case 'unary_neg': {
        regTypes.set(inst.dst, inst.mirType);
        const { ty } = llvmTy(inst.mirType);
        out.push(`  ${inst.dst} = fneg ${ty} ${inst.src}`);
        break;
      }
      case 'bin_op': {
        regTypes.set(inst.dst, inst.mirType);
        if (inst.op === 'Pow') {
          out.push(`  ${inst.dst} = call double @llvm.pow.f64(double ${inst.lhs}, double ${inst.rhs})`);
          break;
        }
        const { ty } = llvmTy(inst.mirType);
        let lOp = inst.lhs;
        let rOp = inst.rhs;
        if (inst.mirType === 'vec3_simd4') {
          if (regTypes.get(inst.lhs) === 'f64') {
            const c = counter.n++;
            out.push(`  %splat_ins_${c} = insertelement <4 x double> poison, double ${inst.lhs}, i32 0`);
            out.push(
              `  %splat_${c} = shufflevector <4 x double> %splat_ins_${c}, <4 x double> poison, <4 x i32> zeroinitializer`,
            );
            lOp = `%splat_${c}`;
          }
          if (regTypes.get(inst.rhs) === 'f64') {
            const c = counter.n++;
            out.push(`  %splat_ins_${c} = insertelement <4 x double> poison, double ${inst.rhs}, i32 0`);
            out.push(
              `  %splat_${c} = shufflevector <4 x double> %splat_ins_${c}, <4 x double> poison, <4 x i32> zeroinitializer`,
            );
            rOp = `%splat_${c}`;
          }
        }
        const opMap: Record<Exclude<BinOp, 'Pow'>, string> = {
          Add: 'fadd',
          Sub: 'fsub',
          Mul: 'fmul',
          Div: 'fdiv',
          Eq: 'fcmp oeq',
          Ne: 'fcmp one',
          Lt: 'fcmp olt',
          Le: 'fcmp ole',
          Gt: 'fcmp ogt',
          Ge: 'fcmp oge',
        };
        out.push(`  ${inst.dst} = ${opMap[inst.op]} ${ty} ${lOp}, ${rOp} ; strict IEEE-754`);
        break;
      }
      case 'call': {
        regTypes.set(inst.dst, inst.mirType);
        const { ty } = llvmTy(inst.mirType);
        const argList = inst.args
          .map((a) => `${llvmTy(regTypes.get(a) ?? inst.mirType).ty} ${a}`)
          .join(', ');
        out.push(`  ${inst.dst} = call ${ty} @aerotick_det_${inst.intrinsic}(${argList})`);
        break;
      }
      case 'branch': {
        const c = counter.n++;
        const elseLbl = inst.elseInsts ? `else_${c}` : `merge_${c}`;
        out.push(`  br i1 ${inst.cond}, label %then_${c}, label %${elseLbl}`);
        out.push(`then_${c}:`);
        emitLlvmInsts(inst.thenInsts, sys, mir, counter, out, regTypes);
        out.push(`  br label %merge_${c}`);
        if (inst.elseInsts) {
          out.push(`else_${c}:`);
          emitLlvmInsts(inst.elseInsts, sys, mir, counter, out, regTypes);
          out.push(`  br label %merge_${c}`);
        }
        out.push(`merge_${c}:`);
        break;
      }
    }
  }
}

export function emitLlvmIr(mir: MirProgram): string {
  const lines: string[] = [
    '; ModuleID = "aerotick_module"',
    '; Target: deterministic IEEE-754 (fp-contract=off, no fast-math)',
    'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"',
    '',
  ];

  for (const layout of mir.layouts) {
    const ptrs = layout.fields.map(() => 'ptr').join(', ');
    lines.push(`%struct.${layout.stateName}_SoA = type { ${ptrs} }`);
  }
  lines.push('');

  for (const sys of mir.systems) {
    lines.push(
      `define void @aerotick_step_${sys.name.toLowerCase()}(ptr noalias readonly %prev, ptr noalias writeonly %next, i64 %count, double %${sys.dtParam}) #0 {`,
    );
    lines.push('entry:');
    lines.push('  %cmp0 = icmp eq i64 %count, 0');
    lines.push('  br i1 %cmp0, label %exit, label %loop');
    lines.push('');
    lines.push('loop:');
    lines.push('  %idx = phi i64 [ 0, %entry ], [ %idx.next, %loop.latch ]');
    const regTypes = new Map<string, MirType>();
    emitLlvmInsts(sys.insts, sys, mir, { n: 0 }, lines, regTypes);
    lines.push('loop.latch:');
    lines.push('  %idx.next = add nuw nsw i64 %idx, 1');
    lines.push('  %done = icmp eq i64 %idx.next, %count');
    lines.push('  br i1 %done, label %exit, label %loop');
    lines.push('');
    lines.push('exit:');
    lines.push('  ret void');
    lines.push('}', '');
  }

  lines.push(
    'attributes #0 = { nounwind strictfp "fp-contract"="off" "no-infs-fp-math"="false" "no-nans-fp-math"="false" }',
  );
  return lines.join('\n');
}

export function emitCHeader(mir: MirProgram): string {
  const lines: string[] = [
    '/* Auto-generated by AeroTick Compiler (aeroc) — C-ABI Host Interface */',
    '#pragma once',
    '#include <stddef.h>',
    '#include <stdint.h>',
    '',
    '#ifdef __cplusplus',
    'extern "C" {',
    '#endif',
    '',
    '/* 32-byte aligned 4-lane SIMD vector (x, y, z, _pad) */',
    'typedef struct __attribute__((aligned(32))) {',
    '    double x;',
    '    double y;',
    '    double z;',
    '    double _pad;',
    '} AeroVec3Simd4;',
    '',
  ];

  for (const layout of mir.layouts) {
    lines.push(
      `/* Structure-of-Arrays (SoA) descriptor for state '${layout.stateName}' (${layout.totalBytesPerEntity} bytes/entity) */`,
    );
    lines.push('typedef struct {');
    for (const f of layout.fields) {
      if (f.mirType === 'vec3_simd4' || f.mirType === 'quat_simd4') {
        lines.push(`    AeroVec3Simd4* ${f.name}; /* 32B aligned */`);
      } else {
        lines.push(`    double* ${f.name}; /* 8B aligned */`);
      }
    }
    lines.push(`} ${layout.stateName}_SoA;`, '');
  }

  for (const sys of mir.systems) {
    lines.push(
      `void aerotick_step_${sys.name.toLowerCase()}(const ${sys.stateName}_SoA* __restrict prev, ${sys.stateName}_SoA* __restrict next, size_t count, double ${sys.dtParam});`,
    );
  }

  lines.push('', '#ifdef __cplusplus', '}', '#endif');
  return lines.join('\n');
}
