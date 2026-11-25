export interface AeroPreset {
  id: string;
  title: string;
  filename: string;
  category: 'Physics Kernel' | 'Scalar Units' | 'Compile-Fail';
  description: string;
  source: string;
}

export const AERO_PRESETS: AeroPreset[] = [
  {
    id: 'reference_body',
    title: 'Reference Rigid Body',
    filename: 'reference_body.aero',
    category: 'Physics Kernel',
    description: 'Thrust-to-mass integration with gravity constant and double-buffered Body SoA.',
    source: `// AeroTick Reference Rigid-Body Physics Kernel (Milestones 2-4)
const GRAVITY: m/s^2 = 9.80665 m/s^2;

state Body {
    pos: Vec3<m>;
    vel: Vec3<m/s>;
    thrust: Vec3<N>;
    mass: kg;
}

system Integrate(dt: s) {
    step(prev: Body) -> next: Body {
        let accel_y: m/s^2 = (prev.thrust.y / prev.mass) - GRAVITY;
        let accel: Vec3<m/s^2> = Vec3(prev.thrust.x / prev.mass, accel_y, prev.thrust.z / prev.mass);
        next.vel = prev.vel + accel * dt;
        next.pos = prev.pos + prev.vel * dt;
        next.thrust = prev.thrust;
        next.mass = prev.mass;
    }
}`,
  },
  {
    id: 'rocket_ascent',
    title: 'Rocket Drag & Staging',
    filename: 'rocket_ascent.aero',
    category: 'Physics Kernel',
    description: 'Variable fuel burn, quadratic aerodynamic drag, and clamped mass floor.',
    source: `// AeroTick Rocket Ascent with Variable Mass, Aerodynamic Drag & Thrust
const GRAVITY: m/s^2 = 9.80665 m/s^2;

state Rocket {
    pos: Vec3<m>;
    vel: Vec3<m/s>;
    thrust: Vec3<N>;
    mass: kg;
    fuel_rate: kg/s;
    drag_k: kg/m;
}

system RocketAscent(dt: s) {
    step(prev: Rocket) -> next: Rocket {
        let speed: m/s = norm(prev.vel);
        let drag_force: Vec3<N> = prev.vel * (-prev.drag_k * speed);
        let gravity_force: Vec3<N> = Vec3(0.0, -prev.mass * GRAVITY, 0.0);
        let net_force: Vec3<N> = prev.thrust + drag_force + gravity_force;
        let accel: Vec3<m/s^2> = net_force / prev.mass;
        let next_vel: Vec3<m/s> = prev.vel + accel * dt;
        next.vel = next_vel;
        next.pos = prev.pos + next_vel * dt;
        next.mass = max(250.0 kg, prev.mass - prev.fuel_rate * dt);
        next.thrust = prev.thrust;
        next.fuel_rate = prev.fuel_rate;
        next.drag_k = prev.drag_k;
    }
}`,
  },
  {
    id: 'orbital_kepler',
    title: 'LEO Orbital Kepler',
    filename: 'orbital_kepler.aero',
    category: 'Physics Kernel',
    description: 'Symplectic Euler orbital propagation using gravitational parameter mu (m^3/s^2).',
    source: `// AeroTick Orbital Kepler Two-Body Symplectic Integrator (LEO 400km)
state OrbitBody {
    pos: Vec3<m>;
    vel: Vec3<m/s>;
    mu: m^3/s^2;
}

system OrbitalStep(dt: s) {
    step(prev: OrbitBody) -> next: OrbitBody {
        let r: m = norm(prev.pos);
        let r2: m^2 = r * r;
        let r_hat: Vec3<1> = normalize(prev.pos);
        let grav_mag: m/s^2 = prev.mu / r2;
        let accel: Vec3<m/s^2> = r_hat * (-grav_mag);
        let next_vel: Vec3<m/s> = prev.vel + accel * dt;
        next.vel = next_vel;
        next.pos = prev.pos + next_vel * dt;
        next.mu = prev.mu;
    }
}`,
  },
  {
    id: 'milestone1',
    title: 'SI Algebra Calculator',
    filename: 'milestone1.aero',
    category: 'Scalar Units',
    description: 'Zero-cost SI dimensional deduction across m, s, kg, and derived Newtons.',
    source: `// Milestone 1: SI Dimensional Algebra Verification
let distance: m = 10.0 m;
let elapsed: s = 2.0 s;
let velocity: m/s = distance / elapsed;
let force: N = 12.0 kg * (velocity / elapsed);`,
  },
  {
    id: 'unit_mismatch',
    title: 'Unit Mismatch Fault',
    filename: 'unit_mismatch.aero',
    category: 'Compile-Fail',
    description: 'Demonstrates compile-time rejection of adding meters to kilograms (E0230).',
    source: `// Milestone 1: Negative compile-fail verification
let speed: m/s = 10.0 m / 2.0 s;
let mismatch = 10.0 m + 5.0 kg;`,
  },
  {
    id: 'deterministic_faults',
    title: 'Step Safety Violations',
    filename: 'deterministic_faults.aero',
    category: 'Compile-Fail',
    description: 'Catches prev mutation (E0301), heap alloc (E0303), unbounded while (E0305), and read-before-write (E0302).',
    source: `// AeroTick Compile-Fail Safety Demo (Immutable prev, Zero Heap, Bounded Control)
state Craft {
    pos: Vec3<m>;
    vel: Vec3<m/s>;
}

system FaultyControl(dt: s) {
    step(prev: Craft) -> next: Craft {
        prev.pos = prev.pos + prev.vel * dt;
        let scratch = alloc(128);
        while dt > 0.0 s {
            next.vel = prev.vel;
        }
        next.pos = next.pos + prev.vel * dt;
    }
}`,
  },
];
