# OpenBLAS Bindings - Function Porting Guide

This guide helps agents port LAPACK wrapper functions from old allocating API to the new pre-allocation pattern used throughout the codebase.

## Table of Contents
1. [Core Principles](#core-principles)
2. [API Pattern Overview](#api-pattern-overview)
3. [Step-by-Step Porting Process](#step-by-step-porting-process)
4. [Code Examples](#code-examples)
5. [Common Patterns](#common-patterns)
6. [Checklist](#checklist)

---

## Core Principles

### 1. **Pre-Allocation Pattern**
- **Never allocate memory inside wrapper functions**
- All arrays must be pre-allocated by the caller
- Provide workspace query functions to determine required sizes

### 2. **Type Safety**
- Use generic constraints: `where is_float(T)`, `where is_complex(Cmplx)`
- Real/complex separation with proper type relationships
- When a proc requires the type and rwork real type, the generic names should be Cmplx and Real
    - The constraints should be `where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)`

### 3. **Unified API**
- When all four types are identical, use `T` Generic and keep all 4 together in the single proc
- Split into `_real` and `_complex` variants when behavior differs
- Use proc groups to provide unified interface
- Consistent naming: `operation_name :: proc {operation_name_real, operation_name_complex}`

### 4. **Return Conventions**
- Always return: `(info: Info, ok: bool)` (if info exists)
- Where `ok = (info == 0)`
- Functions that do not use info should not return info or ok if there is no way to know it is indeed ok
- Never return allocated data
- Some extra stack type returns are acceptable such as `rcond` or similar, they must precede info

### 5. **Use Specialized Matrix Types**
- **CRITICAL**: Always use the appropriate specialized matrix type based on the function's matrix group
- **Do NOT use generic `Matrix($T)`** when a specialized type exists
- Matrix type determines which file the function belongs in

**Matrix Type Reference:**
- **Dense/General** → `^Matrix($T)` in dense_*.odin files
- **Banded** → `^BandedMatrix($T)` in banded_*.odin files (defined in banded.odin)
- **Tridiagonal** → `^Tridiagonal($T)` or arrays (DL, D, DU) in tridiagonal_*.odin files (defined in tridiagonal.odin)
- **Bidiagonal** → `^Bidiagonal($T)` in bidiagonal*.odin files (defined in bidiagonal_svd.odin)
- **Packed Symmetric** → `^PackedSymmetric($T)` in packed_symmetric_*.odin files (defined in packed_symmetric.odin)
- **Packed Hermitian** → `^PackedHermitian($T)` in packed_hermitian_*.odin files (defined in packed_hermitian.odin)
- **Packed Triangular** → `^PackedTriangular($T)` in packed_triangular_*.odin files (defined in packed_triangular.odin)
- **Triangular** → `^Triangular($T)` in triangular_*.odin files (defined in triangular.odin)
- **RFP** → `^RFP($T)` in rfp.odin (defined in rfp.odin)

**Examples:**
```odin
// ❌ WRONG - Using Matrix for banded function
banded_solve :: proc(A: ^Matrix($T), ...) { }  // BAD!

// ✅ CORRECT - Using BandedMatrix
banded_solve :: proc(A: ^BandedMatrix($T), ...) { }  // GOOD!

// ❌ WRONG - Using Matrix for RFP function
rfp_cholesky :: proc(A: ^Matrix($T), ...) { }  // BAD!

// ✅ CORRECT - Using RFP struct
rfp_cholesky :: proc(A: ^RFP($T), ...) { }  // GOOD!
```

---

## API Pattern Overview

### Real vs Complex Variants

**When to split:**
- Complex variants require additional real workspace (`rwork: []Real`)
- Complex variants return real types for certain outputs (norms, singular values)
- Different LAPACK functions are called (e.g., `sgesvd` vs `cgesvd`)

**Example from dense_svd.odin:**

```odin
// Proc group - unified interface
svd :: proc {
    svd_real,
    svd_complex,
}

// Real variant
svd_real :: proc(
    A: ^Matrix($T),
    S: []T,
    U: ^Matrix(T),
    VT: ^Matrix(T),
    work: []T,
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (info: Info, ok: bool)
where is_float(T) {
    // Implementation...
    when T == f32 {
        lapack.sgesvd_(...)
    } else when T == f64 {
        lapack.dgesvd_(...)
    }
    return info, info == 0
}

// Complex variant
svd_complex :: proc(
    A: ^Matrix($Cmplx),
    S: []$Real,              // Note: Real type for singular values!
    U: ^Matrix(Cmplx),
    VT: ^Matrix(Cmplx),
    work: []Cmplx,
    rwork: []Real,           // Additional real workspace
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (info: Info, ok: bool)
where is_complex(Cmplx), Real == real_type_of(Cmplx) {
    // Implementation...
    when Cmplx == complex64 {
        lapack.cgesvd_(...)
    } else when Cmplx == complex128 {
        lapack.zgesvd_(...)
    }
    return info, info == 0
}
```

---

## Step-by-Step Porting Process

### Step 1: Analyze the Old Function

**Old allocating API pattern:**
```odin
m_svd_real :: proc(
    A: ^Matrix($T),
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
    allocator := context.allocator,
) -> (S: []T, U: ^Matrix(T), VT: ^Matrix(T), info: Info) {
    // Queries workspace size
    lwork := query_workspace(...)
    work := make([]T, lwork, allocator)  // ❌ Internal allocation
    defer delete(work)

    S = make([]T, min_mn, allocator)     // ❌ Returns allocated data
    U = make_matrix(...)                  // ❌ Returns allocated matrix

    // Calls LAPACK...
    return S, U, VT, info
}
```

### Step 2: Create Workspace Query Function

```odin
// Query workspace size before allocation
query_workspace_svd :: proc(
    A: ^Matrix($T),
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (lwork: int)
where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    jobu_c := cast(char)jobu
    jobvt_c := cast(char)jobvt

    // Dummy parameters for workspace query
    info: Info
    lwork_query: Blas_Int = -1
    work_query: T

    when T == f32 {
        lapack.sgesvd_(&jobu_c, &jobvt_c, &m, &n, nil, &lda, nil, nil, &m, nil, &n, &work_query, &lwork_query, &info)
    } else when T == f64 {
        lapack.dgesvd_(&jobu_c, &jobvt_c, &m, &n, nil, &lda, nil, nil, &m, nil, &n, &work_query, &lwork_query, &info)
    } else when T == complex64 {
        rwork_dummy: f32
        lapack.cgesvd_(&jobu_c, &jobvt_c, &m, &n, nil, &lda, nil, nil, &m, nil, &n, &work_query, &lwork_query, &rwork_dummy, &info)
    } else when T == complex128 {
        rwork_dummy: f64
        lapack.zgesvd_(&jobu_c, &jobvt_c, &m, &n, nil, &lda, nil, nil, &m, nil, &n, &work_query, &lwork_query, &rwork_dummy, &info)
    }

    return int(real(work_query))
}
```

### Step 3: Create Result Size Query Function

```odin
// Query result sizes for pre-allocation
query_result_sizes_svd :: proc(
    m: int,
    n: int,
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (
    S_size: int,
    U_rows: int, U_cols: int,
    VT_rows: int, VT_cols: int,
) {
    min_mn := min(m, n)

    S_size = min_mn

    if jobu != .None {
        U_rows = m
        U_cols = jobu == .All ? m : min_mn
    }

    if jobvt != .None {
        VT_rows = jobvt == .All ? n : min_mn
        VT_cols = n
    }

    return
}
```

### Step 4: Create New Pre-Allocation Wrapper

```odin
svd_real :: proc(
    A: ^Matrix($T),           // Input/output - overwritten
    S: []T,                   // Pre-allocated by caller
    U: ^Matrix(T),            // Pre-allocated by caller (optional)
    VT: ^Matrix(T),           // Pre-allocated by caller (optional)
    work: []T,                // Pre-allocated workspace
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (info: Info, ok: bool)
where is_float(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    // Validate pre-allocated arrays
    assert(len(S) >= int(min_mn), "S array too small")
    assert(len(work) > 0, "work array must be provided")

    // Prepare LAPACK parameters
    jobu_c := cast(char)jobu
    jobvt_c := cast(char)jobvt

    // Handle optional U matrix
    ldu: Blas_Int = 1
    ptr_u: ^T = nil
    if jobu != .None && U != nil {
        ldu = U.ld
        u_cols := jobu == .All ? int(m) : int(min_mn)
        assert(U.rows == m && U.cols >= u_cols, "U matrix dimensions incorrect")
        ptr_u = raw_data(U.data)
    }

    // Handle optional VT matrix
    ldvt: Blas_Int = 1
    ptr_vt: ^T = nil
    if jobvt != .None && VT != nil {
        ldvt = VT.ld
        vt_rows := jobvt == .All ? int(n) : int(min_mn)
        assert(VT.rows >= vt_rows && VT.cols == n, "VT matrix dimensions incorrect")
        ptr_vt = raw_data(VT.data)
    }

    lwork := Blas_Int(len(work))

    // Call LAPACK
    when T == f32 {
        lapack.sgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, &info)
    }

    return info, info == 0
}
```

### Step 5: Create Complex Variant (if needed)

```odin
svd_complex :: proc(
    A: ^Matrix($Cmplx),
    S: []$Real,               // Note: Real type!
    U: ^Matrix(Cmplx),
    VT: ^Matrix(Cmplx),
    work: []Cmplx,
    rwork: []Real,            // Additional real workspace
    jobu: SVD_Job = .Some,
    jobvt: SVD_Job = .Some,
) -> (info: Info, ok: bool)
where is_complex(Cmplx), Real == real_type_of(Cmplx) {
    // Similar to real variant but with rwork parameter

    assert(len(rwork) >= 5 * int(min(m, n)), "rwork array too small")

    when Cmplx == complex64 {
        lapack.cgesvd_(..., raw_data(rwork), ...)
    } else when Cmplx == complex128 {
        lapack.zgesvd_(..., raw_data(rwork), ...)
    }

    return info, info == 0
}
```

### Step 6: Create Proc Group

```odin
// Unified interface
svd :: proc {
    svd_real,
    svd_complex,
}
```

---

## Common Patterns

### Pattern 1: Matrix Type Inference

```odin
// Use ^Matrix($T) for automatic type inference
proc_name :: proc(A: ^Matrix($T), ...) where is_float(T) {
    // T is automatically inferred from A
}
```

### Pattern 2: Optional Output Parameters

```odin
// Use pointers for optional outputs, check for nil
U: ^Matrix(T),  // Can be nil

if jobu != .None && U != nil {
    // Use U
    ptr_u = raw_data(U.data)
} else {
    // Provide dummy
    ptr_u = nil
}
```

### Pattern 3: Enum to LAPACK Character

```odin
// Always cast enums to u8 for LAPACK
uplo_c := cast(u8)uplo  // Some use u8 instead of u8
```
Note: the lapack wrappers automatically assign c-string lengths to 1 for you.

### Pattern 4: Real Type from Complex

```odin
// Complex variant needs real workspace
svd_complex :: proc(
    A: ^Matrix($Cmplx),
    S: []$Real,              // Real singular values
    work: []Cmplx,
    rwork: []Real,           // Real workspace
) -> (info: Info, ok: bool)
where is_complex(Cmplx), Real == real_type_of(Cmplx) {
    // Real is automatically f32 for complex64, f64 for complex128
}
```

### Pattern 5: Workspace Query with lwork = -1

```odin
query_workspace_xxx :: proc(A: ^Matrix($T), ...) -> (lwork: int) {
    info: Info
    lwork_query: Blas_Int = -1
    work_query: T

    when T == f32 {
        lapack.sxxxxx_(..., &work_query, &lwork_query, &info)
    }

    return int(real(work_query))  // Use real() for complex types
}
```

### Pattern 6: Type-Specific LAPACK Calls

```odin
when T == f32 {
    lapack.sgesvd_(...)
} else when T == f64 {
    lapack.dgesvd_(...)
}

// For complex
when Cmplx == complex64 {
    lapack.cgesvd_(...)
} else when Cmplx == complex128 {
    lapack.zgesvd_(...)
}
```

### Pattern 7: Specialized Matrix Types

```odin
// Banded matrix - extract parameters from BandedMatrix struct
banded_solve :: proc(A: ^BandedMatrix($T), ...) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    kl := A.lower_bw  // Lower bandwidth
    ku := A.upper_bw  // Upper bandwidth
    ldab := A.ld      // Leading dimension of banded storage

    when T == f32 {
        lapack.sgbsv_(&n, &kl, &ku, &nrhs, raw_data(A.data), &ldab, ...)
    }
    // ...
}

// RFP matrix - extract parameters from RFP struct
rfp_cholesky :: proc(A: ^RFP($T), ...) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    transr := cast(u8)A.trans_state  // RFP transpose state
    uplo := cast(u8)A.uplo           // Upper/Lower
    n := A.n                         // Matrix dimension

    when T == f32 {
        lapack.spftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
    }
    // ...
}

// Tridiagonal - uses Tridiagonal struct
tridiagonal_solve :: proc(
    A: ^Tridiagonal($T),  // Tridiagonal matrix struct
    B: ^Matrix(T),        // Right-hand side
    ...,
) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    n := A.n
    nrhs := B.cols
    ldb := B.ld

    when T == f32 {
        lapack.sgtsv_(&n, &nrhs, raw_data(A.DL), raw_data(A.D), raw_data(A.DU), raw_data(B.data), &ldb, &info)
    }
    // Note: A.DL, A.D, A.DU are the diagonals stored in the Tridiagonal struct
}

// Packed Symmetric - uses PackedSymmetric struct
packed_symmetric_solve :: proc(
    A: ^PackedSymmetric($T),  // Packed symmetric matrix struct
    B: ^Matrix(T),            // Right-hand side
    ...,
) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    uplo_c := cast(u8)A.uplo
    n := A.n
    nrhs := B.cols
    ldb := B.ld

    when T == f32 {
        lapack.sspsv_(&uplo_c, &n, &nrhs, raw_data(A.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
    }
    // Note: A.data contains the packed elements, A.uplo is Upper/Lower
}

// Triangular - uses Triangular struct
triangular_solve :: proc(
    A: ^Triangular($T),  // Triangular matrix struct
    B: ^Matrix(T),       // Right-hand side
    ...,
) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    uplo_c := cast(u8)A.uplo
    trans_c := cast(u8)trans
    diag_c := cast(u8)A.diag
    n := A.n
    nrhs := B.cols
    ldb := B.ld

    when T == f32 {
        lapack.strtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &A.ld, raw_data(B.data), &ldb, &info)
    }
    // Note: A.uplo, A.diag contain triangle info
}
```

---

## Code Examples

### Example 1: Simple Function (No Complex Workspace)

```odin
// Function that works the same for real and complex
matrix_copy :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
    uplo: MatrixTriangle = .Upper,
) -> (info: Info, ok: bool)
where is_float(T) || is_complex(T) {
    assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

    uplo_c := cast(u8)uplo
    m := A.rows
    n := A.cols

    when T == f32 {
        lapack.slacpy_(&uplo_c, &m, &n, raw_data(A.data), &A.ld, raw_data(B.data), &B.ld)
    } else when T == f64 {
        lapack.dlacpy_(&uplo_c, &m, &n, raw_data(A.data), &A.ld, raw_data(B.data), &B.ld)
    } else when T == complex64 {
        lapack.clacpy_(&uplo_c, &m, &n, raw_data(A.data), &A.ld, raw_data(B.data), &B.ld)
    } else when T == complex128 {
        lapack.zlacpy_(&uplo_c, &m, &n, raw_data(A.data), &A.ld, raw_data(B.data), &B.ld)
    }

    info = 0  // lacpy doesn't return info
    return info, info == 0
}
```

### Example 2: Real Returns Real, Complex Returns Real

```odin
// Norm functions: always return real type
matrix_norm_real :: proc(
    A: ^Matrix($T),
    norm_type: MatrixNorm = .FrobeniusNorm,
) -> (norm: T)
where is_float(T) {
    norm_c := cast(char)norm_type
    m := A.rows
    n := A.cols

    when T == f32 {
        norm = lapack.slange_(&norm_c, &m, &n, raw_data(A.data), &A.ld, nil)
    } else when T == f64 {
        norm = lapack.dlange_(&norm_c, &m, &n, raw_data(A.data), &A.ld, nil)
    }

    return norm
}

matrix_norm_complex :: proc(
    A: ^Matrix($Cmplx),
    work: []$Real,
    norm_type: MatrixNorm = .FrobeniusNorm,
) -> (norm: Real)
where is_complex(Cmplx), Real == real_type_of(Cmplx) {
    norm_c := cast(char)norm_type
    m := A.rows
    n := A.cols

    // Frobenius norm needs workspace
    ptr_work: ^Real = nil
    if norm_type == .FrobeniusNorm {
        assert(len(work) >= int(m), "work array too small for Frobenius norm")
        ptr_work = raw_data(work)
    }

    when Cmplx == complex64 {
        norm = lapack.clange_(&norm_c, &m, &n, raw_data(A.data), &A.ld, ptr_work)
    } else when Cmplx == complex128 {
        norm = lapack.zlange_(&norm_c, &m, &n, raw_data(A.data), &A.ld, ptr_work)
    }

    return norm
}

matrix_norm :: proc {
    matrix_norm_real,
    matrix_norm_complex,
}
```

### Example 3: Different Behavior for Real vs Complex

```odin
// Symmetric (real) vs Hermitian (complex) factorization
factorize_symmetric_real :: proc(
    A: ^Matrix($T),
    ipiv: []Blas_Int,
    work: []T,
    uplo: MatrixTriangle = .Upper,
) -> (info: Info, ok: bool)
where is_float(T) {
    n := A.rows
    lda := A.ld
    uplo_c := cast(u8)uplo
    lwork := Blas_Int(len(work))

    assert(len(ipiv) >= int(n), "ipiv array too small")

    when T == f32 {
        lapack.ssytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dsytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

factorize_hermitian_complex :: proc(
    A: ^Matrix($Cmplx),
    ipiv: []Blas_Int,
    work: []Cmplx,
    uplo: MatrixTriangle = .Upper,
) -> (info: Info, ok: bool)
where is_complex(Cmplx) {
    n := A.rows
    lda := A.ld
    uplo_c := cast(u8)uplo
    lwork := Blas_Int(len(work))

    assert(len(ipiv) >= int(n), "ipiv array too small")

    when Cmplx == complex64 {
        lapack.chetrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
    } else when Cmplx == complex128 {
        lapack.zhetrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// Different proc groups for different operations
factorize_symmetric :: proc {
    factorize_symmetric_real,
}

factorize_hermitian :: proc {
    factorize_hermitian_complex,
}
```

---

## Checklist

### Before Starting
- [ ] Read LAPACK_FUNCTION_TABLE.md to identify target file and matrix type
- [ ] **CRITICAL**: Identify the specialized matrix type from the Matrix Type column
- [ ] Check if function already exists in target file
- [ ] Locate old implementation in needs_revisited/ or old_allocating_api/
- [ ] Identify if real and complex variants behave differently

### During Porting
- [ ] **CRITICAL**: Use the correct specialized matrix type (e.g., `^BandedMatrix($T)`, `^RFP($T)`, `^Triangular($T)`)
- [ ] **DO NOT** use generic `^Matrix($T)` unless the function is in dense_*.odin files
- [ ] Create workspace query function (`query_workspace_xxx`)
- [ ] Create result size query function (`query_result_sizes_xxx`) if needed
- [ ] Create `_real` variant with `where is_float(T)` constraint
- [ ] Create `_complex` variant if needed with `where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)`
- [ ] Add proc group to unify variants
- [ ] Use correct matrix type for type inference (e.g., `A: ^BandedMatrix($T)`)
- [ ] Extract parameters from specialized struct (e.g., `A.lower_bw`, `A.upper_bw` for banded)
- [ ] Validate all pre-allocated arrays with assertions
- [ ] Handle optional parameters with nil checks
- [ ] Use `when T == f32` pattern for LAPACK calls
- [ ] Return `(info: Info, ok: bool)` where `ok = (info == 0)`
- [ ] Add comprehensive documentation comments

### After Porting
- [ ] Verify all LAPACK bindings exist in f77/ directory
- [ ] Add proc group to appropriate section in target file
- [ ] Test compilation with `odin check .`
- [ ] Mark or delete old implementation
- [ ] Update MISSING_WRAPPERS.md if needed
- [ ] Document any special considerations

---

## Common Pitfalls to Avoid

### ❌ Don't Do This:

```odin
// Don't allocate inside wrapper
proc_name :: proc(A: ^Matrix($T)) -> (result: []T) {
    result = make([]T, n)  // ❌ BAD
    return result
}

// Don't use wrong return type
proc_name :: proc(...) -> (success: bool, info: Info) {  // ❌ Wrong order
    return success, info
}

// Don't forget real workspace for complex
svd_complex :: proc(
    A: ^Matrix($Cmplx),
    work: []Cmplx,
    // Missing: rwork: []Real  // ❌ Complex SVD needs this
) -> (info: Info, ok: bool) {
    // ...
}

// Don't use generic T for complex-specific
proc_complex :: proc(A: ^Matrix($T)) where T == complex64 || T == complex128 {  // ❌ Use is_complex(T)
    // ...
}
```

### ✅ Do This Instead:

```odin
// Pre-allocated by caller
proc_name :: proc(A: ^Matrix($T), result: []T) -> (info: Info, ok: bool) {
    assert(len(result) >= n, "result array too small")
    // Fill result...
    return info, info == 0
}

// Correct return type and order
proc_name :: proc(...) -> (info: Info, ok: bool) {
    return info, info == 0
}

// Include real workspace for complex
svd_complex :: proc(
    A: ^Matrix($Cmplx),
    work: []Cmplx,
    rwork: []Real,  // ✅ Proper real workspace
) -> (info: Info, ok: bool)
where is_complex(Cmplx), Real == real_type_of(Cmplx) {
    // ...
}

// Use proper type helpers
proc_complex :: proc(A: ^Matrix($Cmplx))
where is_complex(Cmplx) {  // ✅ Use is_complex helper
    // ...
}
```

---

## Reference Files

For more examples of the established patterns, refer to:
- **dense_svd.odin** - SVD operations with complex workspace handling
- **dense_linear.odin** - Linear system solvers with expert drivers
- **dense_factorization.odin** - Matrix factorizations (LU, QR, Cholesky)
- **banded_linear.odin** - Banded matrix operations
- **rfp.odin** - RFP format operations
- **matrix_utilities.odin** - Simple utility functions

---

## Questions?

When in doubt:
1. Look for similar functions in existing files
2. Follow the patterns in dense_svd.odin for complex workspace handling
3. Check LAPACK_FUNCTION_TABLE.md for correct target file
4. Ensure all arrays are pre-allocated (never use `make()`)
5. Always provide workspace query functions
6. Use consistent naming: `operation_name_real` and `operation_name_complex`
