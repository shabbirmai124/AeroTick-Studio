import { getUnitReferenceRows } from '../compiler/units';

const INTRINSICS = [
  { sig: 'dot(Vec3<A>, Vec3<B>) -> Scalar<A*B>', note: '3D inner product propagating SI product units' },
  { sig: 'cross(Vec3<A>, Vec3<B>) -> Vec3<A*B>', note: 'Right-handed cross product (e.g. torque = r x F)' },
  { sig: 'norm(Vec3<A>) -> Scalar<A>', note: 'Euclidean magnitude preserving unit A' },
  { sig: 'normalize(Vec3<A>) -> Vec3<1>', note: 'Dimensionless unit direction vector' },
  { sig: 'sqrt(Scalar<A^2>) -> Scalar<A>', note: 'Halves all 7 SI exponents; rejects odd exponents (E0221)' },
  { sig: 'sin / cos / tan(Scalar<rad | 1>) -> Scalar<1>', note: 'Enforces angular units or dimensionless scalar (E0222)' },
  { sig: 'min / max(T, T) -> T', note: 'Requires dimensionally identical operands' },
  { sig: 'select(Bool, T, T) -> T', note: 'Branchless conditional select for SIMD vectorization' },
];

const DIAGNOSTIC_CODES = [
  { code: 'E0102', title: 'Syntax Error (Recovered)', desc: 'Unexpected token with statement/item synchronization' },
  { code: 'E0201', title: 'Unknown Physical Unit', desc: 'Unrecognized SI unit identifier' },
  { code: 'E0230', title: 'Physical Unit Mismatch', desc: 'Attempted addition/subtraction of incompatible SI dimensions (e.g. 10m + 5kg)' },
  { code: 'E0237', title: 'Division by Zero', desc: 'Compile-time zero scalar divisor inside kernel' },
  { code: 'E0301', title: 'Immutable prev Mutation', desc: 'Attempted write to read-only previous frame state buffer' },
  { code: 'E0302', title: 'Uninitialized next Read', desc: 'Reading a next field before assigning it in the current tick' },
  { code: 'E0303', title: 'Forbidden Heap Allocation', desc: 'Dynamic alloc/malloc/Vec/Box/spawn call inside step' },
  { code: 'E0305', title: 'Unbounded Loop Forbidden', desc: 'while / loop construct inside deterministic hot loop' },
  { code: 'W0301', title: 'Unwritten next Field', desc: 'Warning when a state field is not carried forward into next' },
];

export function SiReferenceView() {
  const units = getUnitReferenceRows();

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="bg-white border border-slate-200 rounded overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              First-Class SI Units &amp; Scale Normalization
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              All units compile to an exact 7-component exponent vector [m, kg, s, A, K, mol, cd] + rad and erase at MIR lowering.
            </p>
          </div>
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2 px-4 font-medium">Symbol</th>
                <th className="py-2 px-4 font-medium">Category</th>
                <th className="py-2 px-4 font-medium">Canonical SI Dim</th>
                <th className="py-2 px-4 font-medium">SI Scale Factor</th>
                <th className="py-2 px-4 font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {units.map((u) => (
                <tr key={u.symbol}>
                  <td className="py-2 px-4 font-semibold text-slate-900">{u.symbol}</td>
                  <td className="py-2 px-4 font-sans text-slate-600">{u.category}</td>
                  <td className="py-2 px-4 text-sky-700">{u.canonicalDim}</td>
                  <td className="py-2 px-4 text-slate-700">{u.scale}</td>
                  <td className="py-2 px-4 font-sans text-slate-600">{u.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Vector &amp; Math Intrinsics
            </h3>
            <div className="divide-y divide-slate-100">
              {INTRINSICS.map((item) => (
                <div key={item.sig} className="py-2">
                  <div className="text-xs font-mono font-medium text-slate-900">{item.sig}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{item.note}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Compile-Time Diagnostic Codes
            </h3>
            <div className="divide-y divide-slate-100">
              {DIAGNOSTIC_CODES.map((d) => (
                <div key={d.code} className="py-2 flex items-start gap-3">
                  <span className="text-xs font-mono font-semibold text-rose-700 shrink-0 w-12">
                    {d.code}
                  </span>
                  <div>
                    <div className="text-xs font-medium text-slate-900">{d.title}</div>
                    <div className="text-xs text-slate-500">{d.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
