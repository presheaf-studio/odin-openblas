# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenBLAS Bindings** - Complete Odin bindings for OpenBLAS 0.3.30, providing high-performance linear algebra operations through both low-level C/Fortran bindings and high-level type-safe wrappers.

## Key Architecture Decisions

### Two-Layer Design
1. **Raw Bindings Layer** (`c/` and `f77/` directories) - Direct foreign imports from OpenBLAS
2. **High-Level Layer** - Type-safe Odin wrappers with `Matrix($T)` and `Vector($T)` generic types

### ILP64 Architecture
The entire library is built around 64-bit integer support (ILP64) rather than standard 32-bit (LP32). This affects all integer parameters in LAPACK/BLAS calls.

### Memory Management Pattern
Pre-allocation pattern for all operations:
1. Query workspace sizes with `query_workspace_*` functions
2. Query result sizes with `query_result_sizes_*` functions
3. Pre-allocate all arrays before calling solve procedures
4. No internal memory allocation within procedures

## Build and Test Commands

```bash
# Run tests on Linux
odin test tests/blas -file

# Run tests on Windows
odin test tests/blas

# Build/check the library
odin check .
```

## Critical Implementation Details

### Foreign Import Configuration
The library uses conditional compilation for platform-specific linking:
- **Windows**: Links against local `libopenblas.lib` file
- **Linux**: Uses system-installed openblas package
- **macOS**: Untested, expected to work with brew-installed openblas

### Type System
- **Core Types**: `Matrix($T)` and `Vector($T)` where T ∈ {f32, f64, complex64, complex128}
- **Enums**: Comprehensive enums in `types.odin` for all LAPACK parameters
- **Conversion**: Helper functions for converting between Odin and LAPACK types

### File Organization

#### Matrix Type-Based Structure
The library is organized by matrix type rather than by algorithm, providing clearer boundaries and reducing cross-cutting concerns:

**Base Type Files** (type definitions + conversions TO this type + utilities):
- `bidiagonal.odin` - Bidiagonal type, conversions from banded/tridiagonal
- `tridiagonal.odin` - Tridiagonal type, conversions from general/banded
- `banded.odin` - Banded matrix type, conversions from general/packed
- `packed_symmetric.odin` - Packed symmetric storage, conversions from general
- `packed_hermitian.odin` - Packed Hermitian storage, conversions from general
- `matrix.odin` - General dense matrix type and core utilities

**Operation Files** (algorithms on specific matrix types):
- `bidiagonal_svd.odin` - SVD operations on bidiagonal matrices
- `tridiagonal_eigenvalues.odin` - Eigenvalue problems for tridiagonal
- `banded_linear.odin` - Linear solvers for banded matrices
- `banded_eigenvalues.odin` - Eigenvalue problems for banded matrices
- `packed_symmetric_linear.odin` - Linear solvers for packed symmetric
- `general_svd.odin` - SVD for general dense matrices

**Conversion Rules**:
- Conversions TO a type live with that type (e.g., `banded_to_bidiagonal` in `bidiagonal.odin`)
- Conversions FROM a type live with the destination type (e.g., `bidiagonal_to_tridiagonal` in `tridiagonal.odin`)
- This ensures each type file is self-contained for creating instances of that type

**Other Directories**:
- **Untested Modules**: `_UNTESTED/` contains raw bindgen output needing cleanup
- **Test Suite**: `tests/` contains comprehensive test coverage ported from LAPACK
- **Raw Bindings**: `c/` and `f77/` contain direct foreign imports

## Development Patterns

### Adding New Bindings
1. Start with raw foreign function in `c/` or `f77/`
2. Identify the primary matrix type involved
3. Add to the appropriate matrix type file or operation file
4. Create high-level wrapper following existing patterns
5. Use pre-allocation pattern for workspace
6. Follow conversion rules: TO this type = this file, FROM this type = destination file
7. Add tests using LAPACK reference test data

### Error Handling
- LAPACK info parameters indicate error conditions
- Negative values: invalid input parameters
- Positive values: algorithmic failures
- Zero: successful completion

### Matrix Storage
- Default: Column-major (Fortran) ordering for compatibility
- Row-major support through CBLAS transpose parameters
- Packed storage for symmetric/Hermitian matrices

### Generic Bindings
When possible, all four variants should be under the same generic procedure. Otherwise, the favored approach for combining lapack bindings is to create a `real` variant (f32,f64) and a `complex` variant (complex64,complex128). For the complex variant, the generic terms should be $Cmplx and $Real. since workspaces are preallocated, the user providing the real work array will fulfil the type checkers need to determine what the $Real type is.

## Known Issues and TODOs

### Current Limitations
- Some `_UNTESTED/` files have bindgen syntax errors
- macOS support unverified
- Missing some BLAS Level 2/3 test coverage

### When Working on Untested Files
1. Check for `@(link_name=...)` syntax errors from bindgen
2. Verify parameter types match ILP64 architecture (i64 not i32)
3. Add corresponding tests before marking as stable