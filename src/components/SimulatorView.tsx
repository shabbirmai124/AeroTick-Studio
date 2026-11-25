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

function niceStep(rawStep: number): number {
  const safe = Math.max(1e-6, rawStep);
  const exp = Math.floor(Math.log10(safe));
  const base = Math.pow(10, exp);
  const frac = safe / base;
  if (frac <= 1.5) return base;
  if (frac <= 3.5) return 2 * base;
  if (frac <= 7.5) return 5 * base;
  return 10 * base;
}

function formatAxisTick(val: number, isOrbital: boolean): string {
  if (isOrbital) {
    const km = val / 1000;
    return Math.abs(km) < 0.5 ? '0' : `${Math.round(km).toLocaleString()}km`;
  }
  if (Math.abs(val) < 1e-6) return '0';
  if (Math.abs(val) >= 1000) return `${(val / 1000).toFixed(1)}km`;
  return `${Math.round(val)}m`;
}

export function SimulatorView({ mir, hasErrors, onGoToWorkbench }: SimulatorViewProps) {
  const canRun = Boolean(mir && mir.systems.length > 0 && !hasErrors);
  const isOrbitalDefault = Boolean(
    mir?.layouts[0]?.stateName.toLowerCase().includes('orbit'),
  );
  const isRocketDefault = Boolean(
    mir?.layouts[0]?.fields.some((f) => f.name === 'fuel_rate' || f.name === 'drag_k'),
  );

  const [dt, setDt] = useState<number>(isOrbitalDefault ? 1.0 : 0.01);
  const [entityCount, setEntityCount] = useState<number>(8);
  const [mode, setMode] = useState<NumericMode>('float64');
  const [running, setRunning] = useState<boolean>(false);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [replayResult, setReplayResult] = useState<ReplayVerificationResult | null>(null);

  const runtimeRef = useRef<DeterministicRuntime | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasBoxRef = useRef<HTMLDivElement | null>(null);

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
    const h = Math.max(280, Math.floor(rect.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Outer viewport canvas surface
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const marginLeft = 58;
    const marginRight = 16;
    const marginTop = 14;
    const marginBottom = 26;
    const plotW = Math.max(120, w - marginLeft - marginRight);
    const plotH = Math.max(120, h - marginTop - marginBottom);

    const snap = rt.snapshot();
    const isOrbital = rt.layout.stateName.toLowerCase().includes('orbit');
    const entityTrails = snap.entityTrails.length > 0 ? snap.entityTrails : [snap.trail];

    const allX: number[] = [];
    const allY: number[] = [];
    for (const tr of entityTrails) {
      for (const p of tr) {
        allX.push(p[0]);
        allY.push(p[1]);
      }
    }
    for (const ent of snap.entities) {
      allX.push(ent.pos[0]);
      allY.push(ent.pos[1]);
    }

    let minX: number;
    let maxX: number;
    let minY: number;
    let maxY: number;

    if (isOrbital) {
      minX = Math.min(...allX, -7_400_000);
      maxX = Math.max(...allX, 7_400_000);
      minY = Math.min(...allY, -7_400_000);
      maxY = Math.max(...allY, 7_400_000);
    } else {
      minX = allX.length > 0 ? Math.min(...allX) : -50;
      maxX = allX.length > 0 ? Math.max(...allX) : 50;
      minY = allY.length > 0 ? Math.min(...allY) : 0;
      maxY = allY.length > 0 ? Math.max(...allY) : 200;

      // Include ground datum y=0 when near launch altitude
      if (minY > 0 && minY < 180 && maxY < 450) {
        minY = 0;
      }

      const rawSpanX = maxX - minX;
      const rawSpanY = maxY - minY;
      const minSpan = 140;
      if (rawSpanX < minSpan) {
        const midX = (minX + maxX) / 2;
        minX = midX - minSpan / 2;
        maxX = midX + minSpan / 2;
      }
      if (rawSpanY < minSpan) {
        const midY = (minY + maxY) / 2;
        minY = midY - minSpan / 2;
        maxY = midY + minSpan / 2;
      }
    }

    const padFrac = isOrbital ? 0.1 : 0.16;
    const spanX0 = Math.max(1e-3, maxX - minX);
    const spanY0 = Math.max(1e-3, maxY - minY);
    minX -= spanX0 * padFrac;
    maxX += spanX0 * padFrac;
    minY -= spanY0 * padFrac;
    maxY += spanY0 * padFrac;

    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const scale = Math.min(plotW / spanX, plotH / spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const plotCenterX = marginLeft + plotW / 2;
    const plotCenterY = marginTop + plotH / 2;

    const toPx = (x: number, y: number): [number, number] => [
      plotCenterX + (x - cx) * scale,
      plotCenterY - (y - cy) * scale,
    ];

    const worldLeft = cx - plotW / (2 * scale);
    const worldRight = cx + plotW / (2 * scale);
    const worldBottom = cy - plotH / (2 * scale);
    const worldTop = cy + plotH / (2 * scale);
    const gridStep = niceStep(Math.max(worldRight - worldLeft, worldTop - worldBottom) / 6);

    // Plot area background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(marginLeft, marginTop, plotW, plotH);

    // Clip dynamic geometry inside plot box
    ctx.save();
    ctx.beginPath();
    ctx.rect(marginLeft, marginTop, plotW, plotH);
    ctx.clip();

    // Ground datum band (y <= 0) for non-orbital modes
    if (!isOrbital) {
      const [, gy] = toPx(0, 0);
      if (gy < marginTop + plotH) {
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(marginLeft, Math.max(marginTop, gy), plotW, marginTop + plotH - Math.max(marginTop, gy));
      }
    }

    // World-space adaptive grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    const xStart = Math.ceil(worldLeft / gridStep) * gridStep;
    for (let wx = xStart; wx <= worldRight; wx += gridStep) {
      const [px] = toPx(wx, 0);
      ctx.beginPath();
      ctx.moveTo(px, marginTop);
      ctx.lineTo(px, marginTop + plotH);
      ctx.stroke();
    }
    const yStart = Math.ceil(worldBottom / gridStep) * gridStep;
    for (let wy = yStart; wy <= worldTop; wy += gridStep) {
      const [, py] = toPx(0, wy);
      ctx.beginPath();
      ctx.moveTo(marginLeft, py);
      ctx.lineTo(marginLeft + plotW, py);
      ctx.stroke();
    }

    if (isOrbital) {
      const [ex, ey] = toPx(0, 0);
      const earthR = 6_371_000 * scale;
      const leoR = 6_771_000 * scale;

      // Crosshair axes through Earth center
      ctx.strokeStyle = '#cbd5e1';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(marginLeft, ey);
      ctx.lineTo(marginLeft + plotW, ey);
      ctx.moveTo(ex, marginTop);
      ctx.lineTo(ex, marginTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      // 400km LEO reference ring
      ctx.beginPath();
      ctx.arc(ex, ey, Math.max(12, leoR), 0, Math.PI * 2);
      ctx.strokeStyle = '#93c5fd';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      // Earth body
      ctx.beginPath();
      ctx.arc(ex, ey, Math.max(8, earthR), 0, Math.PI * 2);
      ctx.fillStyle = '#dbeafe';
      ctx.fill();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#1d4ed8';
      ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (earthR > 26) {
        ctx.fillText('Earth (6,371 km)', ex, ey);
      }
    } else {
      const [, gy] = toPx(0, 0);
      if (gy >= marginTop && gy <= marginTop + plotH) {
        ctx.strokeStyle = '#94a3b8';
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(marginLeft, gy);
        ctx.lineTo(marginLeft + plotW, gy);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#64748b';
        ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText('Ground y = 0 m', marginLeft + plotW - 8, gy - 4);
      }
    }

    // Secondary SoA entity trails (1..7)
    for (let i = 1; i < entityTrails.length; i++) {
      const tr = entityTrails[i];
      if (!tr || tr.length < 2) continue;
      ctx.beginPath();
      const [sx, sy] = toPx(tr[0][0], tr[0][1]);
      ctx.moveTo(sx, sy);
      for (let k = 1; k < tr.length; k++) {
        const [px, py] = toPx(tr[k][0], tr[k][1]);
        ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1.25;
      ctx.stroke();
    }

    // Primary Entity[0] trajectory trail
    const primaryTrail = entityTrails[0] ?? snap.trail;
    if (primaryTrail.length > 1) {
      ctx.beginPath();
      const [sx, sy] = toPx(primaryTrail[0][0], primaryTrail[0][1]);
      ctx.moveTo(sx, sy);
      for (let k = 1; k < primaryTrail.length; k++) {
        const [px, py] = toPx(primaryTrail[k][0], primaryTrail[k][1]);
        ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Launch origin marker for Entity[0]
    if (primaryTrail.length > 0 && snap.tick > 0) {
      const [ox, oy] = toPx(primaryTrail[0][0], primaryTrail[0][1]);
      ctx.beginPath();
      ctx.arc(ox, oy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#475569';
      ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('t=0', ox + 6, oy + 4);
    }

    // SoA entities + directional velocity vectors with arrowheads
    for (let idx = snap.entities.length - 1; idx >= 0; idx--) {
      const ent = snap.entities[idx];
      const [px, py] = toPx(ent.pos[0], ent.pos[1]);
      const vNorm = Math.hypot(ent.vel[0], ent.vel[1]);
      if (vNorm > 1e-6) {
        const arrowLen = idx === 0 ? 28 : 18;
        const ux = ent.vel[0] / vNorm;
        const uy = -ent.vel[1] / vNorm;
        const ex = px + ux * arrowLen;
        const ey = py + uy * arrowLen;

        ctx.strokeStyle = idx === 0 ? '#0284c7' : '#94a3b8';
        ctx.lineWidth = idx === 0 ? 1.8 : 1.25;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Arrowhead
        const headLen = idx === 0 ? 6 : 4.5;
        const angle = Math.atan2(uy, ux);
        ctx.fillStyle = idx === 0 ? '#0284c7' : '#94a3b8';
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(
          ex - headLen * Math.cos(angle - Math.PI / 6),
          ey - headLen * Math.sin(angle - Math.PI / 6),
        );
        ctx.lineTo(
          ex - headLen * Math.cos(angle + Math.PI / 6),
          ey - headLen * Math.sin(angle + Math.PI / 6),
        );
        ctx.closePath();
        ctx.fill();
      }

      // White halo for legibility over dark trails
      ctx.beginPath();
      ctx.arc(px, py, idx === 0 ? 6.5 : 4.2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, idx === 0 ? 4.8 : 3.0, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? '#0f172a' : '#64748b';
      ctx.fill();

      if (idx === 0) {
        ctx.fillStyle = '#0f172a';
        ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText('E0', px + 8, py - 4);
      }
    }

    // Minimal floating scale/telemetry bar inside top-left of plot
    const ent0 = snap.entities[0];
    if (ent0) {
      const speed0 = Math.hypot(ent0.vel[0], ent0.vel[1], ent0.vel[2]);
      const altLabel = isOrbital
        ? `alt=${((Math.hypot(ent0.pos[0], ent0.pos[1]) - 6_371_000) / 1000).toFixed(1)} km`
        : `y=${ent0.pos[1].toFixed(1)} m`;
      const hudText = `Grid: ${formatAxisTick(gridStep, isOrbital)}/div · |v₀|=${speed0.toFixed(1)} m/s · ${altLabel}`;
      ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      const tw = ctx.measureText(hudText).width;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.fillRect(marginLeft + 8, marginTop + 8, tw + 14, 20);
      ctx.strokeRect(marginLeft + 8, marginTop + 8, tw + 14, 20);
      ctx.fillStyle = '#334155';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(hudText, marginLeft + 15, marginTop + 18);
    }

    ctx.restore();

    // Plot border
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

    // Axis numeric labels outside clipped plot box
    ctx.fillStyle = '#64748b';
    ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let wx = xStart; wx <= worldRight; wx += gridStep) {
      const [px] = toPx(wx, 0);
      if (px >= marginLeft + 18 && px <= marginLeft + plotW - 18) {
        ctx.fillText(formatAxisTick(wx, isOrbital), px, marginTop + plotH + 6);
      }
    }

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let wy = yStart; wy <= worldTop; wy += gridStep) {
      const [, py] = toPx(0, wy);
      if (py >= marginTop + 8 && py <= marginTop + plotH - 8) {
        ctx.fillText(formatAxisTick(wy, isOrbital), marginLeft - 6, py);
      }
    }

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
    const el = canvasBoxRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      if (runtimeRef.current) drawCanvas(runtimeRef.current);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [canRun]);

  useEffect(() => {
    if (!running || !runtimeRef.current) return;
    let rafId = 0;
    let lastUiUpdate = 0;
    const stepsPerFrame = isOrbitalDefault ? 12 : isRocketDefault ? 6 : 2;

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
  }, [running, isOrbitalDefault, isRocketDefault]);

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
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded flex flex-col min-h-[420px]">
          <div className="px-3.5 py-2 border-b border-slate-200 flex items-center justify-between text-xs shrink-0">
            <span className="font-semibold text-slate-800">
              Double-Buffered Kinematics Viewport (
              <code className="font-mono text-sky-700">{mir?.systems[0]?.name}</code>)
            </span>
            <span className="font-mono text-slate-500">
              tick={snapshot?.tick ?? 0} · t={(snapshot?.simTime ?? 0).toFixed(3)} s
            </span>
          </div>
          <div ref={canvasBoxRef} className="flex-1 min-h-0 p-2">
            <canvas
              ref={canvasRef}
              data-testid="kinematics-canvas"
              className="w-full h-full rounded border border-slate-100 block"
            />
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
