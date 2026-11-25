/*
 * AeroTick Ultra v2.0 — Portable C11 Host Integration Example
 *
 * Build against a compiled AeroTick object or static library (`aeroc emit-ir --llvm --obj body.o`):
 *   cargo run -p aeroc -- emit-header examples/reference_body.aero -o bindings/c_example/aerotick.h
 *   cc -std=c11 -O3 -mavx2 main.c body.o -lm -o aerotick_c_host
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

/* Portable C11 definitions matching `aeroc emit-header examples/reference_body.aero` */
typedef struct {
    alignas(32) double x;
    double y;
    double z;
    double _pad;
} AeroVec3Simd4;

_Static_assert(sizeof(AeroVec3Simd4) == 32, "AeroVec3Simd4 must be 32 bytes");
_Static_assert(alignof(AeroVec3Simd4) == 32, "AeroVec3Simd4 must be 32B aligned");

typedef struct {
    AeroVec3Simd4* pos;
    AeroVec3Simd4* vel;
    AeroVec3Simd4* thrust;
    double* mass;
} Body_SoA;

/* Provided by compiled AeroTick kernel object file: */
extern void aerotick_step_batch_integrate(
    const Body_SoA* restrict prev,
    Body_SoA* restrict next,
    size_t count,
    double dt
);

static void* alloc_aligned32(size_t bytes) {
    size_t rounded = ((bytes + 31u) / 32u) * 32u;
    return aligned_alloc(32, rounded);
}

int main(void) {
    const size_t count = 1024;
    const double dt = 0.01;

    Body_SoA prev = {
        .pos = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .vel = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .thrust = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .mass = (double*)alloc_aligned32(count * sizeof(double)),
    };
    Body_SoA next = {
        .pos = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .vel = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .thrust = (AeroVec3Simd4*)alloc_aligned32(count * sizeof(AeroVec3Simd4)),
        .mass = (double*)alloc_aligned32(count * sizeof(double)),
    };

    if (!prev.pos || !prev.vel || !prev.thrust || !prev.mass ||
        !next.pos || !next.vel || !next.thrust || !next.mass) {
        fprintf(stderr, "aligned_alloc failed\n");
        free(prev.pos); free(prev.vel); free(prev.thrust); free(prev.mass);
        free(next.pos); free(next.vel); free(next.thrust); free(next.mass);
        return 1;
    }

    for (size_t i = 0; i < count; ++i) {
        prev.pos[i] = (AeroVec3Simd4){ 0.0, 120.0, 0.0, 0.0 };
        prev.vel[i] = (AeroVec3Simd4){ 28.0, 42.0, 0.0, 0.0 };
        prev.thrust[i] = (AeroVec3Simd4){ 0.0, 1180.0, 0.0, 0.0 };
        prev.mass[i] = 100.0;
    }

    for (size_t tick = 0; tick < 1000; ++tick) {
        aerotick_step_batch_integrate(&prev, &next, count, dt);
        Body_SoA tmp = prev;
        prev = next;
        next = tmp;
    }

    printf("C11 SoA Host @ 1000 ticks: vel[0].y = %.6f m/s\n", prev.vel[0].y);

    free(prev.pos); free(prev.vel); free(prev.thrust); free(prev.mass);
    free(next.pos); free(next.vel); free(next.thrust); free(next.mass);
    return 0;
}
