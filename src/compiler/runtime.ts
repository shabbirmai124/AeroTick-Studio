import { BinOp } from './ast';
import { MirInst, MirProgram, MirSystem, SoAStateLayout } from './mir';
import { sha256Hex } from './sha256';

export type NumericMode = 'float64' | 'fixed64';

type RegVal = number | [number, number, number];

const Q32_SCALE = 4294967296;
const Q32_SHIFT = 32n;

function toFixed64(v: number): bigint {
  if (!Number.isFinite(v)) return 0n;
  return BigInt(Math.round(v * Q32_SCALE));
}

function fromFixed64(q: bigint): number {
  return Number(q) / Q32_SCALE;
}

function scalarBinop(a: number, b: number, op: BinOp, mode: NumericMode): number {
  if (mode === 'fixed64') {
    const qa = toFixed64(a);
    const qb = toFixed64(b);
    switch (op) {
      case 'Add':
        return fromFixed64(qa + qb);
      case 'Sub':
        return fromFixed64(qa - qb);
      case 'Mul':
        return fromFixed64((qa * qb) >> Q32_SHIFT);
      case 'Div':
        return qb === 0n ? 0 : fromFixed64((qa << Q32_SHIFT) / qb);
      case 'Pow':
        return Math.pow(a, b);
      case 'Eq':
        return qa === qb ? 1 : 0;
      case 'Ne':
        return qa !== qb ? 1 : 0;
      case 'Lt':
        return qa < qb ? 1 : 0;
      case 'Le':
        return qa <= qb ? 1 : 0;
      case 'Gt':
        return qa > qb ? 1 : 0;
      case 'Ge':
        return qa >= qb ? 1 : 0;
    }
  }

  switch (op) {
    case 'Add':
      return a + b;
    case 'Sub':
      return a - b;
    case 'Mul':
      return a * b;
    case 'Div':
      return b === 0 ? 0 : a / b;
    case 'Pow':
      return Math.pow(a, b);
    case 'Eq':
      return a === b ? 1 : 0;
    case 'Ne':
      return a !== b ? 1 : 0;
    case 'Lt':
      return a < b ? 1 : 0;
    case 'Le':
      return a <= b ? 1 : 0;
    case 'Gt':
      return a > b ? 1 : 0;
    case 'Ge':
      return a >= b ? 1 : 0;
  }
}

export interface SimulationSnapshot {
  tick: number;
  simTime: number;
  entityCount: number;
  sha256: string;
  entities: Array<{
    index: number;
    pos: [number, number, number];
    vel: [number, number, number];
    scalars: Record<string, number>;
  }>;
  trail: Array<[number, number]>;
  entityTrails: Array<Array<[number, number]>>;
}

export interface ReplayVerificationResult {
  ticks: number;
  elapsedMs: number;
  ticksPerSec: number;
  hashA: string;
  hashB: string;
  bitExact: boolean;
  finalPos: [number, number, number];
  finalVel: [number, number, number];
}

const MAX_TRAIL_POINTS = 1024;
const MAX_TRACKED_ENTITIES = 8;

export class DeterministicRuntime {
  readonly system: MirSystem;
  readonly layout: SoAStateLayout;
  readonly count: number;
  dt: number;
  mode: NumericMode;
  private prevBuf: Map<string, Float64Array>;
  private nextBuf: Map<string, Float64Array>;
  tick = 0;
  simTime = 0;
  readonly trail: Array<[number, number]> = [];
  readonly entityTrails: Array<Array<[number, number]>> = [];
  private trailStride = 1;

  constructor(mir: MirProgram, count = 1, dt = 0.01, mode: NumericMode = 'float64') {
    const sys = mir.systems[0];
    if (!sys) throw new Error('MIR program has no runnable system');
    const layout = mir.layouts.find((l) => l.stateName === sys.stateName);
    if (!layout) throw new Error(`State layout '${sys.stateName}' missing`);

    this.system = sys;
    this.layout = layout;
    this.count = Math.max(1, count);
    this.dt = dt;
    this.mode = mode;
    this.prevBuf = this.allocBuf();
    this.nextBuf = this.allocBuf();
    this.reset();
  }

  private allocBuf(): Map<string, Float64Array> {
    const m = new Map<string, Float64Array>();
    for (const f of this.layout.fields) {
      m.set(f.name, new Float64Array(this.count * f.lanes));
    }
    return m;
  }

  reset(): void {
    this.tick = 0;
    this.simTime = 0;
    this.trail.length = 0;
    this.entityTrails.length = 0;
    this.trailStride = 1;

    const isOrbital = this.layout.stateName.toLowerCase().includes('orbit');
    const isRocket = this.layout.fields.some(
      (f) => f.name === 'fuel_rate' || f.name === 'drag_k',
    );
    const orbitSatCount = Math.max(1, Math.min(this.count, MAX_TRACKED_ENTITIES));

    for (let i = 0; i < this.count; i++) {
      const orbitTheta = i === 0 ? 0 : (i / orbitSatCount) * Math.PI * 1.5;
      const orbitRadius = i === 0 ? 6_771_000.0 : 6_771_000.0 + (i % 4) * 320_000.0;
      const orbitSpeed =
        i === 0 ? 7_672.0 : Math.sqrt(3.986_004_418e14 / orbitRadius);

      for (const f of this.layout.fields) {
        const arr = this.prevBuf.get(f.name);
        if (!arr) continue;
        const base = i * f.lanes;
        if (f.mirType === 'vec3_simd4') {
          if (f.name === 'pos') {
            if (isOrbital) {
              arr[base] = orbitRadius * Math.cos(orbitTheta);
              arr[base + 1] = orbitRadius * Math.sin(orbitTheta);
            } else if (isRocket) {
              arr[base] = i * 55.0;
              arr[base + 1] = 0.0;
            } else {
              arr[base] = i === 0 ? 0.0 : -i * 12.0;
              arr[base + 1] = i === 0 ? 120.0 : 120.0 - i * 10.0;
            }
            arr[base + 2] = 0.0;
          } else if (f.name === 'vel') {
            if (isOrbital) {
              arr[base] = i === 0 ? 0.0 : -orbitSpeed * Math.sin(orbitTheta);
              arr[base + 1] = i === 0 ? 7_672.0 : orbitSpeed * Math.cos(orbitTheta);
            } else if (isRocket) {
              arr[base] = 14.0 + i * 5.5;
              arr[base + 1] = 0.0;
            } else {
              arr[base] = i === 0 ? 28.0 : 28.0 + i * 3.5;
              arr[base + 1] = i === 0 ? 42.0 : 42.0 - i * 2.2;
            }
            arr[base + 2] = 0.0;
          } else if (f.name === 'thrust') {
            if (isRocket) {
              arr[base] = i * 450.0;
              arr[base + 1] = 24_000.0 - i * 450.0;
            } else {
              arr[base] = 0.0;
              arr[base + 1] = i === 0 ? 1_180.0 : 1_180.0 + i * 65.0;
            }
            arr[base + 2] = 0.0;
          } else {
            arr[base] = 0.0;
            arr[base + 1] = 0.0;
            arr[base + 2] = 0.0;
          }
          arr[base + 3] = 0.0;
        } else {
          if (f.name === 'mass') arr[base] = isRocket ? 1_200.0 + i * 25.0 : 100.0;
          else if (f.name === 'fuel_rate') arr[base] = 4.2;
          else if (f.name === 'drag_k') arr[base] = 0.32;
          else if (f.name === 'mu') arr[base] = 3.986_004_418e14;
          else arr[base] = 1.0;
        }
      }
    }

    const tracked = Math.min(this.count, MAX_TRACKED_ENTITIES);
    for (let i = 0; i < tracked; i++) {
      const p = this.readEntityVec3(i, 'pos');
      this.entityTrails.push([[p[0], p[1]]]);
    }
    if (this.entityTrails[0]) {
      this.trail.push([this.entityTrails[0][0][0], this.entityTrails[0][0][1]]);
    }
  }

  private execInsts(insts: MirInst[], idx: number, regs: Map<string, RegVal>): void {
    for (const inst of insts) {
      switch (inst.kind) {
        case 'const_f64':
          regs.set(inst.dst, inst.value);
          break;
        case 'load_param':
          regs.set(inst.dst, this.dt);
          break;
        case 'const_vec3': {
          const rx = regs.get(inst.x);
          const ry = regs.get(inst.y);
          const rz = regs.get(inst.z);
          regs.set(inst.dst, [
            typeof rx === 'number' ? rx : 0,
            typeof ry === 'number' ? ry : 0,
            typeof rz === 'number' ? rz : 0,
          ]);
          break;
        }
        case 'load_soa': {
          const buf = inst.isNext ? this.nextBuf : this.prevBuf;
          const arr = buf.get(inst.field);
          if (!arr) break;
          if (inst.mirType === 'vec3_simd4') {
            const base = idx * 4;
            regs.set(inst.dst, [arr[base], arr[base + 1], arr[base + 2]]);
          } else {
            regs.set(inst.dst, arr[idx]);
          }
          break;
        }
        case 'store_soa': {
          const arr = this.nextBuf.get(inst.field);
          const val = regs.get(inst.src);
          if (!arr || val === undefined) break;
          if (inst.mirType === 'vec3_simd4') {
            const v = Array.isArray(val) ? val : [val, val, val];
            const base = idx * 4;
            arr[base] = v[0];
            arr[base + 1] = v[1];
            arr[base + 2] = v[2];
            arr[base + 3] = 0;
          } else {
            arr[idx] = Array.isArray(val) ? val[0] : val;
          }
          break;
        }
        case 'vec_extract': {
          const v = regs.get(inst.vec);
          regs.set(inst.dst, Array.isArray(v) ? v[inst.lane] : 0);
          break;
        }
        case 'unary_neg': {
          const v = regs.get(inst.src);
          if (v !== undefined) {
            regs.set(inst.dst, Array.isArray(v) ? [-v[0], -v[1], -v[2]] : -v);
          }
          break;
        }
        case 'bin_op': {
          const l = regs.get(inst.lhs);
          const r = regs.get(inst.rhs);
          if (l === undefined || r === undefined) break;
          if (Array.isArray(l) && Array.isArray(r)) {
            regs.set(inst.dst, [
              scalarBinop(l[0], r[0], inst.op, this.mode),
              scalarBinop(l[1], r[1], inst.op, this.mode),
              scalarBinop(l[2], r[2], inst.op, this.mode),
            ]);
          } else if (Array.isArray(l) && typeof r === 'number') {
            regs.set(inst.dst, [
              scalarBinop(l[0], r, inst.op, this.mode),
              scalarBinop(l[1], r, inst.op, this.mode),
              scalarBinop(l[2], r, inst.op, this.mode),
            ]);
          } else if (typeof l === 'number' && Array.isArray(r)) {
            regs.set(inst.dst, [
              scalarBinop(l, r[0], inst.op, this.mode),
              scalarBinop(l, r[1], inst.op, this.mode),
              scalarBinop(l, r[2], inst.op, this.mode),
            ]);
          } else if (typeof l === 'number' && typeof r === 'number') {
            regs.set(inst.dst, scalarBinop(l, r, inst.op, this.mode));
          }
          break;
        }
        case 'call': {
          const vals = inst.args.map((a) => regs.get(a));
          let res: RegVal = 0;
          if (inst.intrinsic === 'norm' && Array.isArray(vals[0])) {
            const v = vals[0];
            res = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
          } else if (inst.intrinsic === 'normalize' && Array.isArray(vals[0])) {
            const v = vals[0];
            const m = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            res = m === 0 ? [0, 0, 0] : [v[0] / m, v[1] / m, v[2] / m];
          } else if (inst.intrinsic === 'dot' && Array.isArray(vals[0]) && Array.isArray(vals[1])) {
            const a = vals[0];
            const b = vals[1];
            res = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
          } else if (inst.intrinsic === 'cross' && Array.isArray(vals[0]) && Array.isArray(vals[1])) {
            const a = vals[0];
            const b = vals[1];
            res = [
              a[1] * b[2] - a[2] * b[1],
              a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0],
            ];
          } else if (inst.intrinsic === 'sqrt' && typeof vals[0] === 'number') {
            res = Math.sqrt(Math.max(0, vals[0]));
          } else if (inst.intrinsic === 'sin' && typeof vals[0] === 'number') {
            res = Math.sin(vals[0]);
          } else if (inst.intrinsic === 'cos' && typeof vals[0] === 'number') {
            res = Math.cos(vals[0]);
          } else if (inst.intrinsic === 'tan' && typeof vals[0] === 'number') {
            res = Math.tan(vals[0]);
          } else if (inst.intrinsic === 'abs' && typeof vals[0] === 'number') {
            res = Math.abs(vals[0]);
          } else if (
            inst.intrinsic === 'min' &&
            typeof vals[0] === 'number' &&
            typeof vals[1] === 'number'
          ) {
            res = Math.min(vals[0], vals[1]);
          } else if (
            inst.intrinsic === 'max' &&
            typeof vals[0] === 'number' &&
            typeof vals[1] === 'number'
          ) {
            res = Math.max(vals[0], vals[1]);
          } else if (
            inst.intrinsic === 'select' &&
            typeof vals[0] === 'number' &&
            vals[1] !== undefined &&
            vals[2] !== undefined
          ) {
            res = vals[0] !== 0 ? vals[1] : vals[2];
          }
          regs.set(inst.dst, res);
          break;
        }
        case 'branch': {
          const cond = regs.get(inst.cond);
          const taken = typeof cond === 'number' ? cond !== 0 : false;
          if (taken) this.execInsts(inst.thenInsts, idx, regs);
          else if (inst.elseInsts) this.execInsts(inst.elseInsts, idx, regs);
          break;
        }
      }
    }
  }

  stepTick(recordTrail = true): void {
    const regs = new Map<string, RegVal>();
    for (let i = 0; i < this.count; i++) {
      regs.clear();
      this.execInsts(this.system.insts, i, regs);
    }
    const tmp = this.prevBuf;
    this.prevBuf = this.nextBuf;
    this.nextBuf = tmp;
    this.tick++;
    this.simTime += this.dt;

    if (recordTrail && this.tick % this.trailStride === 0) {
      const tracked = Math.min(this.count, MAX_TRACKED_ENTITIES);
      for (let i = 0; i < tracked; i++) {
        const p = this.readEntityVec3(i, 'pos');
        if (!this.entityTrails[i]) this.entityTrails[i] = [];
        this.entityTrails[i].push([p[0], p[1]]);
      }
      if (this.entityTrails[0]) {
        const latest = this.entityTrails[0][this.entityTrails[0].length - 1];
        this.trail.push([latest[0], latest[1]]);
      }

      if (this.trail.length >= MAX_TRAIL_POINTS) {
        const compact = (arr: Array<[number, number]>) => {
          const out: Array<[number, number]> = [];
          for (let k = 0; k < arr.length; k++) {
            if (k === 0 || k % 2 === 0) out.push(arr[k]);
          }
          return out;
        };
        const compacted0 = compact(this.trail);
        this.trail.length = 0;
        this.trail.push(...compacted0);
        for (let i = 0; i < this.entityTrails.length; i++) {
          this.entityTrails[i] = compact(this.entityTrails[i]);
        }
        this.trailStride *= 2;
      }
    }
  }

  stepMany(ticks: number, recordTrail = false): void {
    for (let t = 0; t < ticks; t++) {
      this.stepTick(recordTrail);
    }
  }

  readEntityVec3(index: number, field: string): [number, number, number] {
    const arr = this.prevBuf.get(field);
    if (!arr || arr.length < (index + 1) * 4) return [0, 0, 0];
    const base = index * 4;
    return [arr[base], arr[base + 1], arr[base + 2]];
  }

  stateSha256Hex(): string {
    let totalFloats = 0;
    for (const f of this.layout.fields) {
      totalFloats += this.prevBuf.get(f.name)?.length ?? 0;
    }
    const raw = new Uint8Array(totalFloats * 8);
    const dv = new DataView(raw.buffer);
    let byteOffset = 0;

    for (const f of this.layout.fields) {
      const arr = this.prevBuf.get(f.name);
      if (!arr) continue;
      for (let i = 0; i < arr.length; i++) {
        const canonical = arr[i] === 0 ? 0 : arr[i];
        dv.setFloat64(byteOffset, canonical, true);
        byteOffset += 8;
      }
    }

    return sha256Hex(raw);
  }

  snapshot(): SimulationSnapshot {
    const entities: SimulationSnapshot['entities'] = [];
    const maxSample = Math.min(this.count, MAX_TRACKED_ENTITIES);
    for (let i = 0; i < maxSample; i++) {
      const scalars: Record<string, number> = {};
      for (const f of this.layout.fields) {
        if (f.mirType === 'f64') {
          const arr = this.prevBuf.get(f.name);
          if (arr) scalars[f.name] = arr[i];
        }
      }
      entities.push({
        index: i,
        pos: this.readEntityVec3(i, 'pos'),
        vel: this.readEntityVec3(i, 'vel'),
        scalars,
      });
    }

    const snappedEntityTrails = this.entityTrails.map((t, i) => {
      const copy = [...t];
      const cur = entities[i]?.pos;
      const last = copy[copy.length - 1];
      if (cur && (!last || last[0] !== cur[0] || last[1] !== cur[1])) {
        copy.push([cur[0], cur[1]]);
      }
      return copy;
    });

    const snappedTrail = snappedEntityTrails[0] ?? [...this.trail];

    return {
      tick: this.tick,
      simTime: this.simTime,
      entityCount: this.count,
      sha256: this.stateSha256Hex(),
      entities,
      trail: snappedTrail,
      entityTrails: snappedEntityTrails,
    };
  }
}

export function verifyReplayDeterminism(
  mir: MirProgram,
  ticks = 10000,
  dt = 0.01,
  mode: NumericMode = 'float64',
  count = 1,
): ReplayVerificationResult {
  const vmA = new DeterministicRuntime(mir, count, dt, mode);
  const vmB = new DeterministicRuntime(mir, count, dt, mode);

  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  vmA.stepMany(ticks, false);
  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  vmB.stepMany(ticks, false);

  const elapsedMs = Math.max(0.01, t1 - t0);
  const hashA = vmA.stateSha256Hex();
  const hashB = vmB.stateSha256Hex();

  return {
    ticks,
    elapsedMs,
    ticksPerSec: (ticks / elapsedMs) * 1000,
    hashA,
    hashB,
    bitExact: hashA === hashB,
    finalPos: vmA.readEntityVec3(0, 'pos'),
    finalVel: vmA.readEntityVec3(0, 'vel'),
  };
}
