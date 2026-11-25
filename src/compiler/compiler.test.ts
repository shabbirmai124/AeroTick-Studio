import assert from 'node:assert/strict';
import test from 'node:test';
import { compileAero } from './index.ts';
import { AERO_PRESETS } from './presets.ts';
import { DeterministicRuntime, verifyReplayDeterminism } from './runtime.ts';
import { DIM_DIMENSIONLESS, DIM_METER, DIM_NEWTON, formatDim, sqrtDim } from './units.ts';

test('compiles reference_body.aero with exact 104B SoA layout and valid LLVM/C-ABI', () => {
  const preset = AERO_PRESETS.find((p) => p.id === 'reference_body');
  assert.ok(preset);
  const res = compileAero(preset.source, preset.filename);
  assert.equal(res.hasErrors, false, res.renderedDiagnostics.join('\n'));
  assert.equal(res.soaLayouts.length, 1);
  assert.equal(res.soaLayouts[0].totalBytesPerEntity, 104);
  assert.equal(res.soaLayouts[0].totalPaddingBytesPerEntity, 24);
  assert.match(res.mirText, /soa\.load\.vec3_simd4 @prev\.Body\.thrust/);
  assert.match(res.llvmIr, /ptr noalias readonly %prev/);
  assert.match(res.llvmIr, /ptr noalias writeonly %next/);
  assert.match(res.llvmIr, /"fp-contract"="off"/);
  assert.match(res.llvmIr, /shufflevector <4 x double>/);
  assert.match(res.cHeader, /AeroVec3Simd4/);
  assert.match(res.cHeader, /void aerotick_step_integrate/);
  assert.match(res.astText, /Item::System Integrate/);
});

test('compiles rocket_ascent.aero and orbital_kepler.aero cleanly', () => {
  for (const id of ['rocket_ascent', 'orbital_kepler', 'milestone1']) {
    const preset = AERO_PRESETS.find((p) => p.id === id);
    assert.ok(preset);
    const res = compileAero(preset.source, preset.filename);
    assert.equal(res.hasErrors, false, `Unexpected errors in ${id}: ${res.renderedDiagnostics.join('\n')}`);
  }
});

test('rejects unit mismatch (10m + 5kg), undefined vars, and division by zero', () => {
  const src = `
    let bad = 10.0 m + 5.0 kg;
    let undef = missing_var + 1.0 m;
    let zero_div = 10.0 m / 0.0 s;
  `;
  const res = compileAero(src, 'neg.aero');
  assert.equal(res.hasErrors, true);
  const codes = new Set(res.diagnostics.map((d) => d.code));
  assert.ok(codes.has('E0230'));
  assert.ok(codes.has('E0203'));
  assert.ok(codes.has('E0237'));
});

test('rejects prev mutation, heap allocation, read-before-write, and unbounded while loops', () => {
  const preset = AERO_PRESETS.find((p) => p.id === 'deterministic_faults');
  assert.ok(preset);
  const res = compileAero(preset.source, preset.filename);
  assert.equal(res.hasErrors, true);
  const codes = new Set(res.diagnostics.map((d) => d.code));
  assert.ok(codes.has('E0301'));
  assert.ok(codes.has('E0302'));
  assert.ok(codes.has('E0303'));
  assert.ok(codes.has('E0305'));
});

test('recovers cleanly from syntax error and continues checking subsequent items', () => {
  const src = `let broken = ; let ok: m/s = 10.0 m / 2.0 s;`;
  const res = compileAero(src, 'recover.aero');
  assert.equal(res.hasErrors, true);
  assert.ok(res.diagnostics.some((d) => d.code === 'E0102'));
  assert.equal(res.hir.topBindings.length, 1);
  assert.equal(res.hir.topBindings[0].name, 'ok');
});

test('SI unit algebra handles derived units, sqrt parity, and integer divisors', () => {
  assert.equal(formatDim(DIM_NEWTON), 'N (kg*m/s^2)');
  assert.equal(formatDim(DIM_DIMENSIONLESS), '1');
  assert.equal(sqrtDim(DIM_METER), null);

  const res = compileAero('let scaled: m = 10.0 m / 2; let root: m/s = sqrt(100.0 m^2/s^2);', 'alg.aero');
  assert.equal(res.hasErrors, false);
  assert.equal(res.hir.topBindings.length, 2);
});

test('executes 1,000 ticks with exact analytical velocity and bit-exact SHA-256 replay', () => {
  const preset = AERO_PRESETS.find((p) => p.id === 'reference_body');
  assert.ok(preset);
  const res = compileAero(preset.source, preset.filename);
  assert.ok(res.mir);

  const vm1 = new DeterministicRuntime(res.mir, 1, 0.01, 'float64');
  vm1.stepMany(1000, false);
  const vel = vm1.readEntityVec3(0, 'vel');
  assert.ok(Math.abs(vel[1] - 61.9335) < 1e-9, `expected 61.9335, got ${vel[1]}`);

  const replayFloat = verifyReplayDeterminism(res.mir, 10000, 0.01, 'float64', 1);
  assert.equal(replayFloat.bitExact, true);
  assert.equal(replayFloat.hashA.length, 64);

  const replayFixed = verifyReplayDeterminism(res.mir, 1000, 0.01, 'fixed64', 1);
  assert.equal(replayFixed.bitExact, true);
});

test('preserves t=0 launch origin across >1,500 ticks and separates multi-entity SoA lanes', () => {
  const preset = AERO_PRESETS.find((p) => p.id === 'reference_body');
  assert.ok(preset);
  const res = compileAero(preset.source, preset.filename);
  assert.ok(res.mir);

  const vm = new DeterministicRuntime(res.mir, 8, 0.01, 'float64');
  vm.stepMany(1600, true);
  const snap = vm.snapshot();

  assert.equal(snap.entities.length, 8);
  assert.equal(snap.trail[0][0], 0);
  assert.equal(snap.trail[0][1], 120);
  assert.ok(snap.trail.length <= 1025);
  assert.equal(snap.entityTrails.length, 8);
  assert.notEqual(snap.entities[0].pos[0], snap.entities[1].pos[0]);
  assert.notEqual(snap.entities[0].pos[1], snap.entities[1].pos[1]);
});
