import { RotateCcw } from 'lucide-react';
import { useRef } from 'react';

interface CodeEditorProps {
  filename: string;
  source: string;
  onChange: (nextSource: string) => void;
  onResetPreset: () => void;
  isModified: boolean;
}

export function CodeEditor({
  filename,
  source,
  onChange,
  onResetPreset,
  isModified,
}: CodeEditorProps) {
  const gutterRef = useRef<HTMLDivElement>(null);
  const lines = source.split('\n');

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    if (gutterRef.current) {
      gutterRef.current.scrollTop = e.currentTarget.scrollTop;
    }
  };

  return (
    <section className="flex flex-col h-full bg-white border-r border-slate-200 min-w-0">
      <div className="h-10 px-3 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-mono font-medium text-slate-800 truncate">
            {filename}
          </span>
          {isModified && (
            <span className="text-[10px] font-mono text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded whitespace-nowrap">
              modified
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-mono text-slate-500">
            {lines.length} lines
          </span>
          {isModified && (
            <button
              type="button"
              onClick={onResetPreset}
              className="text-[11px] font-medium text-slate-600 hover:text-slate-900 border border-slate-200 bg-white px-2 py-0.5 rounded flex items-center gap-1 cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      <div className="relative flex flex-1 min-h-0 overflow-hidden bg-white">
        <div
          ref={gutterRef}
          aria-hidden="true"
          className="w-11 py-3 pr-2.5 text-right text-xs leading-6 font-mono text-slate-400 bg-slate-50/60 border-r border-slate-200 select-none overflow-hidden shrink-0"
        >
          {lines.map((_, idx) => (
            <div key={idx}>{idx + 1}</div>
          ))}
        </div>
        <textarea
          aria-label="AeroTick source editor"
          spellCheck={false}
          wrap="off"
          value={source}
          onChange={(e) => onChange(e.target.value)}
          onScroll={handleScroll}
          className="code-editor flex-1 w-full h-full py-3 px-3.5 text-[13px] leading-6 font-mono text-slate-900 bg-white resize-none focus:outline-none whitespace-pre overflow-auto"
        />
      </div>
    </section>
  );
}
