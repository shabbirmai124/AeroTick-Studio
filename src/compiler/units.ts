export interface DimVector {
  /** [m, kg, s, A, K, mol, cd] */
  si: [number, number, number, number, number, number, number];
  /** Angular dimension for rad */
  angle: number;
}

export const DIM_DIMENSIONLESS: DimVector = { si: [0, 0, 0, 0, 0, 0, 0], angle: 0 };
export const DIM_METER: DimVector = { si: [1, 0, 0, 0, 0, 0, 0], angle: 0 };
export const DIM_KILOGRAM: DimVector = { si: [0, 1, 0, 0, 0, 0, 0], angle: 0 };
export const DIM_SECOND: DimVector = { si: [0, 0, 1, 0, 0, 0, 0], angle: 0 };
export const DIM_AMPERE: DimVector = { si: [0, 0, 0, 1, 0, 0, 0], angle: 0 };
export const DIM_KELVIN: DimVector = { si: [0, 0, 0, 0, 1, 0, 0], angle: 0 };
export const DIM_MOLE: DimVector = { si: [0, 0, 0, 0, 0, 1, 0], angle: 0 };
export const DIM_CANDELA: DimVector = { si: [0, 0, 0, 0, 0, 0, 1], angle: 0 };
export const DIM_RADIAN: DimVector = { si: [0, 0, 0, 0, 0, 0, 0], angle: 1 };
export const DIM_NEWTON: DimVector = { si: [1, 1, -2, 0, 0, 0, 0], angle: 0 };
export const DIM_JOULE: DimVector = { si: [2, 1, -2, 0, 0, 0, 0], angle: 0 };
export const DIM_WATT: DimVector = { si: [2, 1, -3, 0, 0, 0, 0], angle: 0 };
export const DIM_PASCAL: DimVector = { si: [-1, 1, -2, 0, 0, 0, 0], angle: 0 };
export const DIM_HERTZ: DimVector = { si: [0, 0, -1, 0, 0, 0, 0], angle: 0 };

const UNIT_TABLE: Record<string, { dim: DimVector; scale: number; category: string; description: string }> = {
  m: { dim: DIM_METER, scale: 1, category: 'Base SI', description: 'Meter (length)' },
  km: { dim: DIM_METER, scale: 1000, category: 'Scaled SI', description: 'Kilometer (1,000 m)' },
  cm: { dim: DIM_METER, scale: 0.01, category: 'Scaled SI', description: 'Centimeter (0.01 m)' },
  mm: { dim: DIM_METER, scale: 0.001, category: 'Scaled SI', description: 'Millimeter (0.001 m)' },
  kg: { dim: DIM_KILOGRAM, scale: 1, category: 'Base SI', description: 'Kilogram (mass)' },
  g: { dim: DIM_KILOGRAM, scale: 0.001, category: 'Scaled SI', description: 'Gram (0.001 kg)' },
  s: { dim: DIM_SECOND, scale: 1, category: 'Base SI', description: 'Second (time)' },
  ms: { dim: DIM_SECOND, scale: 0.001, category: 'Scaled SI', description: 'Millisecond (0.001 s)' },
  A: { dim: DIM_AMPERE, scale: 1, category: 'Base SI', description: 'Ampere (electric current)' },
  K: { dim: DIM_KELVIN, scale: 1, category: 'Base SI', description: 'Kelvin (thermodynamic temp)' },
  mol: { dim: DIM_MOLE, scale: 1, category: 'Base SI', description: 'Mole (amount of substance)' },
  cd: { dim: DIM_CANDELA, scale: 1, category: 'Base SI', description: 'Candela (luminous intensity)' },
  rad: { dim: DIM_RADIAN, scale: 1, category: 'Angular', description: 'Radian (plane angle)' },
  deg: { dim: DIM_RADIAN, scale: Math.PI / 180, category: 'Angular', description: 'Degree (pi/180 rad)' },
  N: { dim: DIM_NEWTON, scale: 1, category: 'Derived SI', description: 'Newton (kg*m/s^2)' },
  kN: { dim: DIM_NEWTON, scale: 1000, category: 'Derived SI', description: 'Kilonewton (1,000 N)' },
  J: { dim: DIM_JOULE, scale: 1, category: 'Derived SI', description: 'Joule (kg*m^2/s^2)' },
  W: { dim: DIM_WATT, scale: 1, category: 'Derived SI', description: 'Watt (kg*m^2/s^3)' },
  Pa: { dim: DIM_PASCAL, scale: 1, category: 'Derived SI', description: 'Pascal (kg/(m*s^2))' },
  MPa: { dim: DIM_PASCAL, scale: 1_000_000, category: 'Derived SI', description: 'Megapascal (10^6 Pa)' },
  Hz: { dim: DIM_HERTZ, scale: 1, category: 'Derived SI', description: 'Hertz (1/s)' },
};

export interface UnitSpecRow {
  symbol: string;
  scale: number;
  category: string;
  description: string;
  canonicalDim: string;
}

export function getUnitReferenceRows(): UnitSpecRow[] {
  return Object.entries(UNIT_TABLE).map(([symbol, info]) => ({
    symbol,
    scale: info.scale,
    category: info.category,
    description: info.description,
    canonicalDim: formatDim(info.dim),
  }));
}

export function lookupUnit(symbol: string): { dim: DimVector; scale: number } | null {
  const entry = UNIT_TABLE[symbol];
  return entry ? { dim: entry.dim, scale: entry.scale } : null;
}

export function dimsEqual(a: DimVector, b: DimVector): boolean {
  if (a.angle !== b.angle) return false;
  for (let i = 0; i < 7; i++) {
    if (a.si[i] !== b.si[i]) return false;
  }
  return true;
}

export function mulDim(a: DimVector, b: DimVector): DimVector {
  return {
    si: [
      a.si[0] + b.si[0],
      a.si[1] + b.si[1],
      a.si[2] + b.si[2],
      a.si[3] + b.si[3],
      a.si[4] + b.si[4],
      a.si[5] + b.si[5],
      a.si[6] + b.si[6],
    ],
    angle: a.angle + b.angle,
  };
}

export function divDim(a: DimVector, b: DimVector): DimVector {
  return {
    si: [
      a.si[0] - b.si[0],
      a.si[1] - b.si[1],
      a.si[2] - b.si[2],
      a.si[3] - b.si[3],
      a.si[4] - b.si[4],
      a.si[5] - b.si[5],
      a.si[6] - b.si[6],
    ],
    angle: a.angle - b.angle,
  };
}

export function powDim(a: DimVector, exp: number): DimVector {
  return {
    si: [
      a.si[0] * exp,
      a.si[1] * exp,
      a.si[2] * exp,
      a.si[3] * exp,
      a.si[4] * exp,
      a.si[5] * exp,
      a.si[6] * exp,
    ],
    angle: a.angle * exp,
  };
}

export function sqrtDim(a: DimVector): DimVector | null {
  if (a.angle % 2 !== 0) return null;
  for (let i = 0; i < 7; i++) {
    if (a.si[i] % 2 !== 0) return null;
  }
  return {
    si: [
      a.si[0] / 2,
      a.si[1] / 2,
      a.si[2] / 2,
      a.si[3] / 2,
      a.si[4] / 2,
      a.si[5] / 2,
      a.si[6] / 2,
    ],
    angle: a.angle / 2,
  };
}

export function formatDim(d: DimVector): string {
  if (dimsEqual(d, DIM_DIMENSIONLESS)) return '1';
  if (dimsEqual(d, DIM_NEWTON)) return 'N (kg*m/s^2)';
  if (dimsEqual(d, DIM_JOULE)) return 'J (kg*m^2/s^2)';
  if (dimsEqual(d, DIM_WATT)) return 'W (kg*m^2/s^3)';
  if (dimsEqual(d, DIM_PASCAL)) return 'Pa (kg/(m*s^2))';

  const names = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd'];
  const num: string[] = [];
  const den: string[] = [];

  for (let i = 0; i < 7; i++) {
    const e = d.si[i];
    if (e > 0) num.push(e === 1 ? names[i] : `${names[i]}^${e}`);
    else if (e < 0) den.push(Math.abs(e) === 1 ? names[i] : `${names[i]}^${Math.abs(e)}`);
  }
  if (d.angle > 0) num.push(d.angle === 1 ? 'rad' : `rad^${d.angle}`);
  else if (d.angle < 0) den.push(Math.abs(d.angle) === 1 ? 'rad' : `rad^${Math.abs(d.angle)}`);

  const numStr = num.length === 0 ? '1' : num.join('*');
  if (den.length === 0) return numStr;
  if (den.length === 1) return `${numStr}/${den[0]}`;
  return `${numStr}/(${den.join('*')})`;
}
