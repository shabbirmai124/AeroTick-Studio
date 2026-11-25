# Godot 4 GDExtension Integration Guide (AeroTick Ultra v2.0)

AeroTick kernels compile directly to cache-aligned C11/C++ ABI libraries with zero runtime dependencies and zero heap allocations, making them ideal as deterministic `_physics_process(delta)` swarm backends inside Godot 4 via GDExtension.

## 1. Export Header & Native Object

Run directly from the repository root:

```bash
cargo run -p aeroc -- emit-header examples/rocket_6dof.aero -o addons/aerotick_gdext/src/aerotick.h
cargo run -p aeroc --features llvm -- emit-ir examples/rocket_6dof.aero --llvm --obj addons/aerotick_gdext/build/rocket_6dof.o
```

## 2. Double-Buffered SoA Binding in C++ (`PhysicsServer3D` / `MultiMeshInstance3D`)

```cpp
#include "aerotick.h"
#include <godot_cpp/classes/multi_mesh.hpp>
#include <utility>

void AeroSwarmNode::step_fixed(double dt, size_t count) {
    // Pass 32-byte aligned prev/next SoA arrays with non-aliasing guarantees
    aerotick_step_batch_rocket6dofstep(&prev_soa, &next_soa, count, dt);
    std::swap(prev_soa, next_soa);
}
```

## 3. Sync Transforms & Quaternions to Godot `MultiMesh`

Because `AeroQuat` (`w, x, y, z`) and `AeroVec3Simd4` (`x, y, z, _pad`) are contiguous 32-byte SIMD buffers verified by compile-time static assertions in `aerotick.h`, copy entity positions and `godot::Quaternion(q.x, q.y, q.z, q.w)` directly into Godot's `Transform3D` buffer in a tight cache-linear pass while preserving bit-exact cross-platform rollback determinism.
