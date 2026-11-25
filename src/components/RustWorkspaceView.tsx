import { Check, Copy, FileCode, Terminal } from 'lucide-react';
import { useState } from 'react';
import { RUST_WORKSPACE_FILES } from '../rust_workspace/cratesManifest';

export function RustWorkspaceView() {
  const [selectedPath, setSelectedPath] = useState(RUST_WORKSPACE_FILES[1].path);
  const [copied, setCopied] = useState(false);

  const activeFile =
    RUST_WORKSPACE_FILES.find((f) => f.path === selectedPath) ?? RUST_WORKSPACE_FILES[0];

  const handleCopy = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(activeFile.content).catch(() => {});
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="flex-1 flex min-h-0 bg-white">
      <aside className="w-76 border-r border-slate-200 bg-slate-50/70 flex flex-col shrink-0 overflow-y-auto">
        <div className="p-3 border-b border-slate-200">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
            Rust Compiler Crates (`/aerotick`)
          </div>
          <div className="text-[11px] text-slate-600 leading-4">
            10-crate Cargo Edition 2024 workspace implementing Milestones 1–4, CI, and benchmarks.
          </div>
        </div>

        <div className="p-2 space-y-1 flex-1 overflow-y-auto">
          {RUST_WORKSPACE_FILES.map((file) => {
            const active = file.path === activeFile.path;
            return (
              <button
                key={file.path}
                type="button"
                onClick={() => setSelectedPath(file.path)}
                className={`w-full text-left px-2.5 py-2 rounded border cursor-pointer ${
                  active
                    ? 'bg-white border-slate-900 shadow-xs'
                    : 'bg-transparent border-transparent hover:bg-white hover:border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-mono font-medium text-slate-900 truncate">
                    {file.crate}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200 whitespace-nowrap shrink-0">
                    {file.milestone}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-500 truncate mt-0.5">
                  {file.path}
                </div>
              </button>
            );
          })}
        </div>

        <div className="p-3 border-t border-slate-200 bg-white">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-700 mb-1">
            <Terminal className="w-3.5 h-3.5" />
            <span>Local Cargo Commands</span>
          </div>
          <pre className="text-[11px] font-mono text-slate-600 bg-slate-50 border border-slate-200 rounded p-2 overflow-x-auto leading-4">
{`cd aerotick
cargo test --workspace
cargo run -p aeroc -- check examples/reference_body.aero
cargo run --release -p aeroc -- bench examples/reference_body.aero`}
          </pre>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="h-11 px-4 border-b border-slate-200 bg-slate-50/50 flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <FileCode className="w-4 h-4 text-slate-500 shrink-0" />
            <span className="text-xs font-mono font-semibold text-slate-900 truncate">
              {activeFile.path}
            </span>
            <span className="text-xs text-slate-500 truncate hidden md:inline">
              — {activeFile.summary}
            </span>
          </div>
          <button
            type="button"
            onClick={handleCopy}
            className="px-2.5 py-1 rounded border border-slate-200 bg-white text-xs text-slate-700 hover:text-slate-900 flex items-center gap-1 shrink-0 cursor-pointer"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 bg-white">
          <pre className="text-xs font-mono leading-5 text-slate-800 whitespace-pre">
            {activeFile.content}
          </pre>
        </div>
      </div>
    </div>
  );
}
