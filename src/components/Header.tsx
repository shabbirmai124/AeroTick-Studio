import { BookOpen, Code2, Cpu, FolderGit2, Play } from 'lucide-react';

export type StudioView = 'workbench' | 'simulator' | 'workspace' | 'reference';

interface HeaderProps {
  activeView: StudioView;
  onSelectView: (view: StudioView) => void;
  hasErrors: boolean;
  errorCount: number;
  warningCount: number;
  canSimulate: boolean;
  onLaunchSimulator: () => void;
}

const NAV_ITEMS: Array<{ id: StudioView; label: string; icon: typeof Code2 }> = [
  { id: 'workbench', label: 'Compiler Workbench', icon: Code2 },
  { id: 'simulator', label: 'Tick Simulator', icon: Play },
  { id: 'workspace', label: 'Rust Workspace (M1–M4)', icon: FolderGit2 },
  { id: 'reference', label: 'SI Reference', icon: BookOpen },
];

export function Header({
  activeView,
  onSelectView,
  hasErrors,
  errorCount,
  warningCount,
  canSimulate,
  onLaunchSimulator,
}: HeaderProps) {
  return (
    <header className="h-14 border-b border-slate-200 bg-white px-4 flex items-center justify-between gap-4 shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded bg-slate-900 text-white flex items-center justify-center shrink-0">
          <Cpu className="w-4 h-4" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm tracking-tight text-slate-900 whitespace-nowrap">
              AeroTick Studio
            </span>
            <span className="text-[11px] font-mono text-slate-500 border border-slate-200 bg-slate-50 px-1.5 py-0.5 rounded whitespace-nowrap">
              aeroc v0.1
            </span>
          </div>
          <p className="text-[11px] text-slate-500 truncate">
            Deterministic Real-Time Physics DSL &amp; SoA Compiler
          </p>
        </div>
      </div>

      <nav className="flex items-center gap-1 h-full" aria-label="Primary sections">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = activeView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectView(item.id)}
              className={`h-full px-3 flex items-center gap-1.5 text-xs font-medium border-b-2 whitespace-nowrap cursor-pointer ${
                active
                  ? 'border-slate-900 text-slate-900 bg-slate-50/70'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="flex items-center gap-3 shrink-0">
        <div
          className={`text-xs font-mono px-2.5 py-1 rounded border whitespace-nowrap ${
            hasErrors
              ? 'border-rose-200 bg-rose-50 text-rose-700'
              : 'border-emerald-200 bg-emerald-50 text-emerald-700'
          }`}
        >
          {hasErrors
            ? `${errorCount} error${errorCount === 1 ? '' : 's'}`
            : `0 errors${warningCount > 0 ? ` · ${warningCount} warn` : ''}`}
        </div>

        <button
          type="button"
          disabled={!canSimulate}
          onClick={onLaunchSimulator}
          className="px-3 py-1.5 rounded text-xs font-medium bg-slate-900 text-white hover:bg-slate-800 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
        >
          <Play className="w-3.5 h-3.5" />
          <span>Run Kernel</span>
        </button>
      </div>
    </header>
  );
}
