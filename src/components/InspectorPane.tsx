import { Check, CheckCircle2, Copy, ShieldAlert } from 'lucide-react';
import { useState } from 'react';
import { CompileResult } from '../compiler';
import { formatValueType } from '../compiler/sema';

export type InspectorTab = 'diagnostics' | 'ast' | 'mir' | 'codegen';

interface InspectorPaneProps {
  result: CompileResult;
  activeTab: InspectorTab;
  onSelectTab: (tab: InspectorTab) => void;
}

const TABS: Array<{ id: InspectorTab; label: string }> = [
  { id: 'diagnostics', label: 'Diagnostics & Tokens' },
  { id: 'ast', label: 'AST & SI HIR' },
  { id: 'mir', label: 'SoA MIR & Layout' },
  { id: 'codegen', label: 'LLVM IR & C-ABI' },
];

export function InspectorPane({ result, activeTab, onSelectTab }: InspectorPaneProps) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const copyText = (key: string, text: string) => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  };

  return (
    <section className="flex flex-col h-full bg-slate-50/40 min-w-0">
      <div className="h-10 border-b border-slate-200 bg-white px-2 flex items-center gap-1 shrink-0 overflow-x-auto">
        {TABS.map((t) => {
          const active = activeTab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => onSelectTab(t.id)}
              className={`h-full px-3 text-xs font-medium border-b-2 whitespace-nowrap cursor-pointer ${
                active
                  ? 'border-slate-900 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
        {activeTab === 'diagnostics' && (
          <>
            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between gap-2 mb-2.5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Ariadne Compiler Diagnostics
                </h3>
                <span className="text-xs font-mono text-slate-500">
                  {result.diagnostics.length} item{result.diagnostics.length === 1 ? '' : 's'}
                </span>
              </div>

              {result.diagnostics.length === 0 ? (
                <div className="flex items-center gap-2 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-3 py-2.5">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>
                    Zero dimensional, ownership, or determinism errors. Ready for SoA lowering &amp; LLVM IR emission.
                  </span>
                </div>
              ) : (
                <div className="space-y-3">
                  {result.renderedDiagnostics.map((text, idx) => {
                    const d = result.diagnostics[idx];
                    const isErr = d?.severity === 'error';
                    return (
                      <div
                        key={idx}
                        className={`rounded border p-3 ${
                          isErr
                            ? 'border-rose-200 bg-rose-50/70 text-rose-950'
                            : 'border-amber-200 bg-amber-50/70 text-amber-950'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 text-xs font-semibold mb-1.5">
                          <ShieldAlert
                            className={`w-3.5 h-3.5 shrink-0 ${
                              isErr ? 'text-rose-600' : 'text-amber-600'
                            }`}
                          />
                          <span className="font-mono">{d?.code}</span>
                          <span>— {d?.title}</span>
                        </div>
                        <pre className="text-xs font-mono leading-5 whitespace-pre-wrap break-words text-slate-800">
                          {text}
                        </pre>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between mb-2.5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Phase 1 Token Stream ({result.tokens.length})
                </h3>
                <span className="text-[11px] font-mono text-slate-500">logos mirror</span>
              </div>
              <div className="flex flex-wrap gap-1.5 max-h-60 overflow-y-auto">
                {result.tokens.map((tok, i) => (
                  <div
                    key={i}
                    className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded border border-slate-200 bg-slate-50 text-[11px] font-mono"
                  >
                    <span className="text-slate-400">{tok.type}</span>
                    <span className="text-slate-900 font-medium">{tok.lexeme}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {activeTab === 'ast' && (
          <>
            <div className="bg-white border border-slate-200 rounded p-3.5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2.5">
                Phase 3 SI Dimensional Types &amp; HIR Symbol Table
              </h3>

              <div className="space-y-3">
                {result.hir.consts.length > 0 && (
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 mb-1">Constants</div>
                    <div className="border border-slate-200 rounded divide-y divide-slate-100">
                      {result.hir.consts.map((c) => (
                        <div key={c.name} className="px-3 py-1.5 flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-900 font-medium">const {c.name}</span>
                          <span className="text-sky-700">{formatValueType(c.ty)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.hir.states.length > 0 && (
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 mb-1">Double-Buffered States</div>
                    {result.hir.states.map((st) => (
                      <div key={st.name} className="border border-slate-200 rounded mb-2">
                        <div className="px-3 py-1.5 bg-slate-50 border-b border-slate-200 text-xs font-mono font-semibold text-slate-800">
                          state {st.name}
                        </div>
                        <div className="divide-y divide-slate-100">
                          {st.fields.map((f) => (
                            <div key={f.name} className="px-3 py-1.5 flex items-center justify-between text-xs font-mono">
                              <span className="text-slate-800">{f.name}</span>
                              <span className="text-sky-700">{formatValueType(f.ty)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {result.hir.systems.map((sys) => (
                  <div key={sys.name}>
                    <div className="text-[11px] font-semibold text-slate-500 mb-1">
                      System <code className="font-mono text-slate-700">{sys.name}</code> Kernel Locals
                    </div>
                    <div className="border border-slate-200 rounded divide-y divide-slate-100">
                      {sys.locals.length === 0 ? (
                        <div className="px-3 py-2 text-xs text-slate-500">No local bindings</div>
                      ) : (
                        sys.locals.map((loc, idx) => (
                          <div key={idx} className="px-3 py-1.5 flex items-center justify-between text-xs font-mono">
                            <span className="text-slate-900">let {loc.name}</span>
                            <span className="text-sky-700">{formatValueType(loc.ty)}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                ))}

                {result.hir.topBindings.length > 0 && (
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 mb-1">Top-Level Bindings</div>
                    <div className="border border-slate-200 rounded divide-y divide-slate-100">
                      {result.hir.topBindings.map((b, idx) => (
                        <div key={idx} className="px-3 py-1.5 flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-900">let {b.name}</span>
                          <span className="text-sky-700">{formatValueType(b.ty)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Phase 2 Untyped AST
                </h3>
                <button
                  type="button"
                  onClick={() => copyText('ast', result.astText)}
                  className="text-[11px] text-slate-600 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
                >
                  {copiedKey === 'ast' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'ast' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="text-xs font-mono leading-5 text-slate-800 whitespace-pre-wrap break-words">
                {result.astText}
              </pre>
            </div>
          </>
        )}

        {activeTab === 'mir' && (
          <>
            <div className="bg-white border border-slate-200 rounded p-3.5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2.5">
                Structure-of-Arrays (SoA) Memory Layout Report
              </h3>
              {result.soaLayouts.length === 0 ? (
                <p className="text-xs text-slate-500">
                  No <code className="font-mono">state</code> blocks compiled. Declare a{' '}
                  <code className="font-mono">state</code> block or fix errors to inspect 32-byte aligned SoA buffers.
                </p>
              ) : (
                <div className="space-y-3">
                  {result.soaLayouts.map((layout) => (
                    <div key={layout.stateName} className="border border-slate-200 rounded overflow-hidden">
                      <div className="px-3 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs font-mono">
                        <span className="font-semibold text-slate-900">{layout.stateName}_SoA</span>
                        <span className="text-slate-600">
                          {layout.totalBytesPerEntity} B/entity · {layout.totalPaddingBytesPerEntity} B SIMD pad
                        </span>
                      </div>
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-slate-50/50 text-[11px] text-slate-500 border-b border-slate-200">
                          <tr>
                            <th className="py-1.5 px-3 font-medium">Field</th>
                            <th className="py-1.5 px-3 font-medium">MIR Type</th>
                            <th className="py-1.5 px-3 font-medium">Align</th>
                            <th className="py-1.5 px-3 font-medium">Stride</th>
                            <th className="py-1.5 px-3 font-medium">Pad</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {layout.fields.map((f) => (
                            <tr key={f.name}>
                              <td className="py-1.5 px-3 text-slate-900 font-medium">{f.name}</td>
                              <td className="py-1.5 px-3 text-sky-700">{f.mirType}</td>
                              <td className="py-1.5 px-3 text-slate-600">{f.alignmentBytes}B</td>
                              <td className="py-1.5 px-3 text-slate-800">{f.bytesPerEntity}B</td>
                              <td className="py-1.5 px-3 text-slate-500">{f.paddingBytesPerEntity}B</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Phase 4 Unit-Erased SoA MIR
                </h3>
                <button
                  type="button"
                  onClick={() => copyText('mir', result.mirText)}
                  className="text-[11px] text-slate-600 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
                >
                  {copiedKey === 'mir' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'mir' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="text-xs font-mono leading-5 text-slate-800 whitespace-pre-wrap break-words">
                {result.mirText}
              </pre>
            </div>
          </>
        )}

        {activeTab === 'codegen' && (
          <>
            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                    Phase 5 Deterministic LLVM IR
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    Strict IEEE-754 (<code className="font-mono">fp-contract=off</code>),{' '}
                    <code className="font-mono">noalias readonly %prev</code>,{' '}
                    <code className="font-mono">noalias writeonly %next</code>
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => copyText('llvm', result.llvmIr)}
                  className="text-[11px] text-slate-600 hover:text-slate-900 flex items-center gap-1 cursor-pointer shrink-0"
                >
                  {copiedKey === 'llvm' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'llvm' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="text-xs font-mono leading-5 text-slate-800 whitespace-pre-wrap break-words">
                {result.llvmIr}
              </pre>
            </div>

            <div className="bg-white border border-slate-200 rounded p-3.5">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                    Phase 6 Exported C-ABI Header (<code className="font-mono lowercase">aerotick.h</code>)
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    Godot 4 GDExtension / Unreal Engine / C99 FFI bindings
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => copyText('cheader', result.cHeader)}
                  className="text-[11px] text-slate-600 hover:text-slate-900 flex items-center gap-1 cursor-pointer shrink-0"
                >
                  {copiedKey === 'cheader' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'cheader' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="text-xs font-mono leading-5 text-slate-800 whitespace-pre-wrap break-words">
                {result.cHeader}
              </pre>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
