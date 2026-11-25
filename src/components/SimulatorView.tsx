import { Gauge, Pause, Play, RotateCcw, ShieldCheck, StepForward } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { MirProgram } from '../compiler/mir';
import {
  DeterministicRuntime,
  NumericMode,
  ReplayVerificationResult,
  SimulationSnapshot,
  verifyReplayDeterminism,
} from '../compiler/runtime';

interface SimulatorViewProps {
  mir: MirProgram | null;
  hasErrors: boolean;
  onGoToWorkbench: () => void;
}

const DT_OPTIONS = [0.001, 0.005, 0.01, 0.05, 1.0];

export function SimulatorView({ mir, hasErrors, onGoToWorkbench }: SimulatorViewProps) {
  const canRun = Boolean(mir && mir.systems.length > 0 && !hasErrors);
  const isOrbitalDefault = Boolean(
    mir?.layouts[0]?.stateName.toLowerCase().includes('orbit'),
  );

  const [dt, setDt] = useState<number>(isOrbitalDefault ? 1.0 : 0.01);
  const [entityCount, setEntityCount] = useState<number>(8);
  const [mode, setMode] = useState<NumericMode>('float64');
  const [running, setRunning] = useState<boolean>(false);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [replayResult, setReplayResult] = useState<ReplayVerificationResult | null>(null);

  const runtimeRef = useRef<DeterministicRuntime | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    setDt(isOrbitalDefault ? 1.0 : 0.01);
  }, [isOrbitalDefault, mir]);

  const drawCanvas = (rt: DeterministicRuntime) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(240, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Light technical grid
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let x = 40; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 40; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const snap = rt.snapshot();
    const trail = snap.trail;
    const allX = [...trail.map((p) => p[0]), ...snap.entities.map((e) => e.pos[0])];
    const allY = [...trail.map((p) => p[1]), ...snap.entities.map((e) => e.pos[1])];

    const isOrbital = rt.layout.stateName.toLowerCase().includes('orbit');
    let minX = Math.min(...allX, isOrbital ? -7_200_000 : -10);
    let maxX = Math.max(...allX, isOrbital ? 7_200_000 : 10);
    let minY = Math.min(...allY, isOrbital ? -7_200_000 : 0);
    let maxY = Math.max(...allY, isOrbital ? 7_200_000 : 200);

    const spanX = Math.max(1e-3, maxX - minX);
    const spanY = Math.max(1e-3, maxY - minY);
    const pad = 42;
    const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    const toPx = (x: number, y: number): [number, number] => [
      w / 2 + (x - cx) * scale,
      h / 2 - (y - cy) * scale,
    ];

    if (isOrbital) {
      const [ex, ey] = toPx(0, 0);
      const earthR = 6_371_000 * scale;
      ctx.beginPath();
      ctx.arc(ex, ey, Math.max(8, earthR), 0, Math.PI * 2);
      ctx.fillStyle = '#dbeafe';
      ctx.fill();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      const [, gy] = toPx(0, 0);
      if (gy >= 0 && gy <= h) {
        ctx.strokeStyle = '#94a3b8';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Trajectory trail
    if (trail.length > 1) {
      ctx.beginPath();
      const [sx, sy] = toPx(trail[0][0], trail[0][1]);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < trail.length; i++) {
        const [px, py] = toPx(trail[i][0], trail[i][1]);
        ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // SoA entities + velocity vectors
    snap.entities.forEach((ent, idx) => {
      const [px, py] = toPx(ent.pos[0], ent.pos[1]);
      const vNorm = Math.hypot(ent.vel[0], ent.vel[1]);
      if (vNorm > 1e-6) {
        const vx = (ent.vel[0] / vNorm) * 24;
        const vy = -(ent.vel[1] / vNorm) * 24;
        ctx.strokeStyle = idx === 0 ? '#0284c7' : '#94a3b8';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + vx, py + vy);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(px, py, idx === 0 ? 5.5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? '#0f172a' : '#64748b';
      ctx.fill();
    });

    ctx.restore();
  };

  useEffect(() => {
    setRunning(false);
    setReplayResult(null);
    if (!canRun || !mir) {
      runtimeRef.current = null;
      setSnapshot(null);
      return;
    }
    const rt = new DeterministicRuntime(mir, entityCount, dt, mode);
    runtimeRef.current = rt;
    setSnapshot(rt.snapshot());
    drawCanvas(rt);
  }, [mir, canRun, entityCount, mode]);

  useEffect(() => {
    if (runtimeRef.current) {
      runtimeRef.current.dt = dt;
    }
  }, [dt]);

  useEffect(() => {
    if (!running || !runtimeRef.current) return;
    let rafId = 0;
    let lastUiUpdate = 0;
    const stepsPerFrame = isOrbitalDefault ? 6 : 2;

    const frame = (now: number) => {
      const rt = runtimeRef.current;
      if (!rt) return;
      for (let s = 0; s < stepsPerFrame; s++) {
        rt.stepTick(true);
      }
      if (now - lastUiUpdate >= 33) {
        lastUiUpdate = now;
        drawCanvas(rt);
        setSnapshot(rt.snapshot());
      }
      rafId = requestAnimationFrame(frame);
    };

    rafId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafId);
  }, [running, isOrbitalDefault]);

  const handleStepOne = () => {
    const rt = runtimeRef.current;
    if (!rt) return;
    rt.stepTick(true);
    drawCanvas(rt);
    setSnapshot(rt.snapshot());
  };

  const handleReset = () => {
    setRunning(false);
    const rt = runtimeRef.current;
    if (!rt) return;
    rt.reset();
    drawCanvas(rt);
    setSnapshot(rt.snapshot());
  };

  const handleRunReplayBench = () => {
    if (!mir || !canRun) return;
    const res = verifyReplayDeterminism(mir, 100_000, dt, mode, 1);
    setReplayResult(res);
  };

  if (!canRun) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md bg-white border border-slate-200 rounded p-6 text-center space-y-3">
          <h2 className="text-sm font-semibold text-slate-900">
            No Valid Compiled Kernel Ready
          </h2>
          <p className="text-xs text-slate-600 leading-5">
            {hasErrors
              ? 'Resolve the compiler diagnostics in the workbench first, or select a valid physics kernel such as Reference Rigid Body, Rocket Drag & Staging, or LEO Orbital Kepler.'
              : 'The current file only contains scalar constant equations. Select a preset with a `state` and `system/step` kernel to run the deterministic tick simulator.'}
          </p>
          <button
            type="button"
            onClick={onGoToWorkbench}
            className="px-3.5 py-1.5 rounded bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 cursor-pointer"
          >
            Back to Compiler Workbench
          </button>
        </div>
      </div>
    );
  }

  const ent0 = snapshot?.entities[0];

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-slate-50 overflow-y-auto">
      <div className="border-b border-slate-200 bg-white px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setRunning((r) => !r)}
            className="px-3 py-1.5 rounded bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 flex items-center gap-1.5 cursor-pointer"
          >
            {running ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{running ? 'Pause' : 'Start Loop'}</span>
          </button>

          <button
            type="button"
            onClick={handleStepOne}
            disabled={running}
            className="px-2.5 py-1.5 rounded border border-slate-200 bg-white text-slate-800 text-xs font-medium hover:bg-slate-50 disabled:opacity-50 flex items-center gap-1 cursor-pointer"
          >
            <StepForward className="w-3.5 h-3.5" />
            <span>Step 1 Tick</span>
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="px-2.5 py-1.5 rounded border border-slate-200 bg-white text-slate-700 text-xs font-medium hover:bg-slate-50 flex items-center gap-1 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-1.5 text-slate-600">
            <span className="font-medium">dt (s):</span>
            <select
              value={dt}
              onChange={(e) => setDt(Number(e.target.value))}
              className="border border-slate-200 rounded px-2 py-1 bg-white font-mono text-xs text-slate-900"
            >
              {DT_OPTIONS.map((v) => (
                <option key={v} value={v}>
                  {v} s
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-1.5 text-slate-600">
            <span className="font-medium">Entities:</span>
            <select
              value={entityCount}
              onChange={(e) => setEntityCount(Number(e.target.value))}
              className="border border-slate-200 rounded px-2 py-1 bg-white font-mono text-xs text-slate-900"
            >
              {[1, 8, 32, 64].map((c) => (
                <option key={c} value={c}>
                  {c} SoA
                </option>
              ))}
            </select>
          </label>

          <div className="flex items-center border border-slate-200 rounded overflow-hidden bg-white">
            <button
              type="button"
              onClick={() => setMode('float64')}
              className={`px-2.5 py-1 font-mono text-[11px] cursor-pointer ${
                mode === 'float64' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              IEEE-754 f64
            </button>
            <button
              type="button"
              onClick={() => setMode('fixed64')}
              className={`px-2.5 py-1 font-mono text-[11px] cursor-pointer ${
                mode === 'fixed64' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              Q32.32 Fixed64
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded flex flex-col min-h-[340px]">
          <div className="px-3.5 py-2 border-b border-slate-200 flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-800">
              Double-Buffered Kinematics Viewport (
              <code className="font-mono text-sky-700">{mir?.systems[0]?.name}</code>)
            </span>
            <span className="font-mono text-slate-500">
              tick={snapshot?.tick ?? 0} · t={(snapshot?.simTime ?? 0).toFixed(3)} s
            </span>
          </div>
          <div className="flex-1 min-h-0 p-2">
            <canvas ref={canvasRef} className="w-full h-full rounded border border-slate-100 block" />
          </div>
        </div>

        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white border border-slate-200 rounded p-3.5 space-y-2.5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Entity[0] Live State &amp; SHA-256 Digest
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="border border-slate-200 rounded p-2 bg-slate-50/50">
                <div className="text-[11px] text-slate-500">Tick / Sim Time</div>
                <div className="text-slate-900 font-semibold mt-0.5">
                  {snapshot?.tick ?? 0} · {(snapshot?.simTime ?? 0).toFixed(3)} s
                </div>
              </div>
              <div className="border border-slate-200 rounded p-2 bg-slate-50/50">
                <div className="text-[11px] text-slate-500">SoA Stride</div>
                <div className="text-slate-900 font-semibold mt-0.5">
                  {mir?.layouts[0]?.totalBytesPerEntity ?? 0} B × {entityCount}
                </div>
              </div>
            </div>

            {ent0 && (
              <div className="border border-slate-200 rounded divide-y divide-slate-100 text-xs font-mono">
                <div className="px-3 py-1.5 flex justify-between">
                  <span className="text-slate-500">pos (m)</span>
                  <span className="text-slate-900">
                    ({ent0.pos[0].toFixed(2)}, {ent0.pos[1].toFixed(2)}, {ent0.pos[2].toFixed(2)})
                  </span>
                </div>
                <div className="px-3 py-1.5 flex justify-between">
                  <span className="text-slate-500">vel (m/s)</span>
                  <span className="text-slate-900">
                    ({ent0.vel[0].toFixed(3)}, {ent0.vel[1].toFixed(3)}, {ent0.vel[2].toFixed(3)})
                  </span>
                </div>
                {Object.entries(ent0.scalars).map(([k, v]) => (
                  <div key={k} className="px-3 py-1.5 flex justify-between">
                    <span className="text-slate-500">{k}</span>
                    <span className="text-slate-900">{v.toPrecision(6)}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="border border-slate-200 rounded p-2.5 bg-slate-50/60">
              <div className="text-[11px] font-medium text-slate-500 mb-1">
                Live SoA Buffer SHA-256 (Canonical Little-Endian IEEE-754)
              </div>
              <div className="text-[11px] font-mono text-slate-800 break-all">
                {snapshot?.sha256 ?? ''}
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded p-3.5 space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Bit-Exact Replay Benchmark
                </h3>
                <p className="text-[11px] text-slate-500">
                  Runs two fresh VM instances for 100,000 ticks and compares SHA-256 state hashes.
                </p>
              </div>
              <button
                type="button"
                onClick={handleRunReplayBench}
                className="px-3 py-1.5 rounded bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 flex items-center gap-1.5 whitespace-nowrap cursor-pointer shrink-0"
              >
                <Gauge className="w-3.5 h-3.5" />
                <span>Verify 100k Ticks</span>
              </button>
            </div>

            {replayResult && (
              <div className="border border-emerald-200 bg-emerald-50/60 rounded p-3 text-xs space-y-1.5">
                <div className="flex items-center gap-1.5 font-semibold text-emerald-800">
                  <ShieldCheck className="w-4 h-4 shrink-0" />
                  <span>
                    {replayResult.bitExact
                      ? 'BIT-EXACT MATCH VERIFIED (Run A SHA-256 === Run B SHA-256)'
                      : 'HASH MISMATCH DETECTED'}
                  </span>
                </div>
                <div className="font-mono text-[11px] text-slate-700 space-y-0.5">
                  <div>
                    Throughput: {Math.round(replayResult.ticksPerSec).toLocaleString()} ticks/sec (
                    {replayResult.elapsedMs.toFixed(1)} ms for 100,000 ticks)
                  </div>
                  <div className="break-all">Run A: {replayResult.hashA}</div>
                  <div className="break-all">Run B: {replayResult.hashB}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
