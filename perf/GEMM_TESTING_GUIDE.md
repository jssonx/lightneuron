# GEMM Performance Testing Guide

## 🎯 Quick Start

### How to Test Different GEMM Implementations

**Step 1**: Edit `perf/gemm_perf.c` line 72

**Step 2**: Change `gemm_type` to the implementation you want to test

**Step 3**: Run `make gemm_perf`

## 📝 Example Usage

### Test BLAS (Fastest)
```c
// In perf/gemm_perf.c line 72:
GemmType gemm_type = GEMM_BLAS;
```
```bash
make gemm_perf
# Output:
# Testing GEMM implementation: blas
# Performing multiplication for N = 1024...
# 627.77 GFLOPS -- 3.42 ms
```

### Test SIMD (AVX2)
```c
GemmType gemm_type = GEMM_SIMD;
```
```bash
make gemm_perf
# Output:
# Testing GEMM implementation: simd
# 10.35 GFLOPS -- 207.43 ms (on ARM, fallback to scalar)
```

### Test Naive (Baseline)
```c
GemmType gemm_type = GEMM_NAIVE;
```
```bash
make gemm_perf
# Output:
# Testing GEMM implementation: naive
# ~0.5 GFLOPS (slowest)
```

## 📊 Available Implementations

| Enum Value | Name | README Version | Optimization Strategy |
|-----------|------|----------------|----------------------|
| `GEMM_NAIVE` | naive | v1 | Basic ijk loop |
| `GEMM_INTERCHANGE` | interchange_loops | v2 | Loop interchange (ijk→jpi) |
| `GEMM_PARALLEL` | parallel_loops | v4 | OpenMP parallel |
| `GEMM_TILING` | tiling | v5 | Loop tiling/blocking |
| `GEMM_REC_TILING` | rec_tiling | v6 | Recursive divide-and-conquer |
| `GEMM_SIMD` | simd | v7 | AVX2 SIMD (x86 only) |
| `GEMM_BLAS` | blas | v10 | Accelerate/OpenBLAS |

## 🔧 Configuration Location

**File**: `perf/gemm_perf.c`

**Line**: 72

```c
// ============================================================
// CONFIGURATION: Choose GEMM implementation to test
// ============================================================
GemmType gemm_type = GEMM_SIMD;  // 👈 Change this

// Available options:
// GEMM_NAIVE       - v1: Basic ijk loop
// GEMM_INTERCHANGE - v2: Loop interchange (jpi)
// GEMM_PARALLEL    - v4: OpenMP parallel loops
// GEMM_TILING      - v5: Loop tiling/blocking
// GEMM_REC_TILING  - v6: Recursive divide-and-conquer
// GEMM_SIMD        - v7: AVX2 SIMD + recursive
// GEMM_BLAS        - v10: BLAS library
// ============================================================
```

## 💡 Benefits of This System

### Before (Old Way)
```c
// Need to comment/uncomment manually
// gemm_naive(N, N, N, A, N, B_transposed, N, C, N);
// gemm_interchange_loops(N, N, N, A, N, B_transposed, N, C, N);
gemm_simd(N, N, N, A, N, B_transposed, N, C, N);  // Currently testing
// gemm_tiling(N, N, N, A, N, B, N, C, N);
```
❌ Error-prone
❌ Easy to forget which one is active
❌ Hard to compare results

### After (New Way)
```c
GemmType gemm_type = GEMM_SIMD;  // Just change this enum!
```
✅ Clear and concise
✅ Shows implementation name in output
✅ Easy to switch between implementations
✅ Type-safe

## 🚀 Advanced: Batch Testing

To test all implementations at once, you can modify the main loop:

```c
int main() {
    GemmType implementations[] = {
        GEMM_NAIVE, GEMM_INTERCHANGE, GEMM_PARALLEL,
        GEMM_TILING, GEMM_REC_TILING, GEMM_SIMD, GEMM_BLAS
    };

    for (int impl = 0; impl < 7; impl++) {
        GemmType gemm_type = implementations[impl];
        printf("\n=== Testing %s ===\n", gemm_get_name(gemm_type));

        // Run benchmark...
    }
}
```

## 📖 Architecture

### Files Modified
- `kernel/gemm/gemm.h` - Added `GemmType` enum and dispatcher functions
- `kernel/gemm/gemm_dispatch.c` - Implementation of dispatcher
- `perf/gemm_perf.c` - Uses new configuration system
- `Makefile` - Includes `gemm_dispatch.c` in build

### How It Works
```
perf/gemm_perf.c (line 72)
    ↓
GemmType gemm_type = GEMM_SIMD;
    ↓
gemm_dispatch(gemm_type, ...)
    ↓
switch (gemm_type) {
    case GEMM_SIMD: gemm_simd(...); break;
    case GEMM_BLAS: cblas_dgemm(...); break;
    ...
}
```

## ⚠️ Platform Notes

- **x86-64**: All implementations work
- **ARM (Apple Silicon)**:
  - `GEMM_SIMD` falls back to scalar (~10 GFLOPS)
  - `GEMM_BLAS` uses Accelerate framework (~600 GFLOPS) ✅ Recommended
  - Others work normally

## 📚 Related Files

- Implementation registration: `kernel/gemm/gemm.h`
- Dispatcher logic: `kernel/gemm/gemm_dispatch.c`
- Test program: `perf/gemm_perf.c`
- Individual implementations: `kernel/gemm/gemm_*.c`

---

**Quick Reference**: Just change **one line** (line 72 in `perf/gemm_perf.c`) to test different GEMM implementations!
