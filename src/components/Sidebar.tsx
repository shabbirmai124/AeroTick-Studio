import { AlertTriangle, CheckCircle2, FileCode2, Layers } from 'lucide-react';
import { AeroPreset } from '../compiler/presets';

interface SidebarProps {
  presets: AeroPreset[];
  activePresetId: string;
  onSelectPreset: (preset: AeroPreset) => void;
  hasErrors: boolean;
  hasSystem: boolean;
}

const PIPELINE_STAGES = [
  { phase: 'P1', name: 'logos Lexer', detail: 'Zero-copy spanned tokens' },
  { phase: 'P2', name: 'chumsky Parser', detail: 'Pratt expr + error recovery' },
  { phase: 'P3', name: 'SI Sema', detail: '7-dim SI + prev/next verifier' },
  { phase: 'P4', name: 'SoA MIR', detail: 'Unit erasure + 32B alignment' },
  { phase: 'P5', name: 'LLVM IR', detail: 'noalias + fp-contract=off' },
  { phase: 'P6', name: 'C-ABI / VM', detail: 'aerotick.h + SHA-256 replay' },
];

export function Sidebar({
  presets,
  activePresetId,
  onSelectPreset,
  hasErrors,
  hasSystem,
}: SidebarProps) {
  return (
    <aside className="w-72 border-r border-slate-200 bg-slate-50/70 flex flex-col shrink-0 overflow-y-auto">
      <div className="p-3 border-b border-slate-200">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
          .aero Kernels &amp; Tests
        </div>
        <div className="space-y-1.5">
          {presets.map((preset) => {
            const active = preset.id === activePresetId;
            const isFail = preset.category === 'Compile-Fail';
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => onSelectPreset(preset)}
                className={`w-full text-left p-2.5 rounded border cursor-pointer ${
                  active
                    ? 'bg-white border-slate-900 shadow-xs'
                    : 'bg-white/80 border-slate-200 hover:border-slate-300 hover:bg-white'
                }`}
              >
                <div className="flex items-center gap-1.5 min-w-0 mb-1">
                  <FileCode2 className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  <span className="text-xs font-semibold text-slate-900 truncate">
                    {preset.title}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-mono text-slate-500 min-w-0 truncate">
                    {preset.filename}
                  </span>
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded whitespace-nowrap shrink-0 ${
                      isFail
                        ? 'bg-rose-50 text-rose-700 border border-rose-200'
                        : 'bg-slate-100 text-slate-600 border border-slate-200'
                    }`}
                  >
                    {preset.category}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="p-3">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
          <Layers className="w-3.5 h-3.5" />
          <span>Compiler Pipeline</span>
        </div>
        <div className="space-y-1.5">
          {PIPELINE_STAGES.map((st, idx) => {
            const blocked = hasErrors && idx >= 3;
            const inactiveSim = !hasSystem && idx >= 4;
            return (
              <div
                key={st.phase}
                className={`px-2.5 py-2 rounded border bg-white flex items-center justify-between gap-2 ${
                  blocked ? 'border-rose-200 opacity-65' : 'border-slate-200'
                }`}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold text-slate-500">
                      {st.phase}
                    </span>
                    <span className="text-xs font-medium text-slate-800 truncate">{st.name}</span>
                  </div>
                  <div className="text-[11px] text-slate-500 truncate">{st.detail}</div>
                </div>
                {blocked ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-rose-500 shrink-0" />
                ) : (
                  <CheckCircle2
                    className={`w-3.5 h-3.5 shrink-0 ${
                      inactiveSim ? 'text-slate-300' : 'text-emerald-600'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
