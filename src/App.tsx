import { useMemo, useState } from 'react';
import { compileAero } from './compiler';
import { AERO_PRESETS, AeroPreset } from './compiler/presets';
import { CodeEditor } from './components/CodeEditor';
import { Header, StudioView } from './components/Header';
import { InspectorPane, InspectorTab } from './components/InspectorPane';
import { RustWorkspaceView } from './components/RustWorkspaceView';
import { Sidebar } from './components/Sidebar';
import { SimulatorView } from './components/SimulatorView';
import { SiReferenceView } from './components/SiReferenceView';

export default function App() {
  const [activeView, setActiveView] = useState<StudioView>('workbench');
  const [activePreset, setActivePreset] = useState<AeroPreset>(AERO_PRESETS[0]);
  const [source, setSource] = useState<string>(AERO_PRESETS[0].source);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('diagnostics');

  const compileResult = useMemo(
    () => compileAero(source, activePreset.filename),
    [source, activePreset.filename],
  );

  const errorCount = compileResult.diagnostics.filter((d) => d.severity === 'error').length;
  const warningCount = compileResult.diagnostics.filter((d) => d.severity === 'warning').length;
  const canSimulate = Boolean(
    !compileResult.hasErrors &&
      compileResult.mir &&
      compileResult.mir.systems.length > 0,
  );

  const handleSelectPreset = (preset: AeroPreset) => {
    setActivePreset(preset);
    setSource(preset.source);
    if (preset.category === 'Compile-Fail') {
      setInspectorTab('diagnostics');
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50 text-slate-900 overflow-hidden">
      <Header
        activeView={activeView}
        onSelectView={setActiveView}
        hasErrors={compileResult.hasErrors}
        errorCount={errorCount}
        warningCount={warningCount}
        canSimulate={canSimulate}
        onLaunchSimulator={() => setActiveView('simulator')}
      />

      {activeView === 'workbench' && (
        <main className="flex-1 flex min-h-0">
          <Sidebar
            presets={AERO_PRESETS}
            activePresetId={activePreset.id}
            onSelectPreset={handleSelectPreset}
            hasErrors={compileResult.hasErrors}
            hasSystem={compileResult.hir.systems.length > 0}
          />
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 min-w-0 min-h-0">
            <CodeEditor
              filename={activePreset.filename}
              source={source}
              onChange={setSource}
              onResetPreset={() => setSource(activePreset.source)}
              isModified={source !== activePreset.source}
            />
            <InspectorPane
              result={compileResult}
              activeTab={inspectorTab}
              onSelectTab={setInspectorTab}
            />
          </div>
        </main>
      )}

      {activeView === 'simulator' && (
        <SimulatorView
          mir={compileResult.mir}
          hasErrors={compileResult.hasErrors}
          onGoToWorkbench={() => setActiveView('workbench')}
        />
      )}

      {activeView === 'workspace' && <RustWorkspaceView />}

      {activeView === 'reference' && <SiReferenceView />}
    </div>
  );
}
