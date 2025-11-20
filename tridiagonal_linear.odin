package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// GENERAL TRIDIAGONAL LINEAR SOLVERS (GT PREFIX)
// ===================================================================================
// Linear system solvers for general tridiagonal matrices A*X = B
// Non-allocating API with pre-allocated arrays

// ===================================================================================
// TRIDIAGONAL FACTORIZATION (GTTRF)
// ===================================================================================

// Factorize general tridiagonal matrix using Gaussian elimination with partial pivoting
// A = P*L*U where P is permutation, L is unit lower triangular, U is upper triangular
trid_factorize :: proc(
    dl: []$T, // Subdiagonal (size n-1, modified)
    d: []T, // Diagonal (size n, modified)
    du: []T, // Superdiagonal (size n-1, modified)
    du2: []T, // Second superdiagonal (size n-2, output from factorization)
    ipiv: []Blas_Int, // Pivot indices (size n)
) -> (
    info: Info,
    ok: bool,
) where is_numeric_type(T) {
    n := len(d)
    assert(len(dl) >= n - 1 || n <= 1, "Subdiagonal array too small")
    assert(len(du) >= n - 1 || n <= 1, "Superdiagonal array too small")
    assert(len(du2) >= n - 2 || n <= 2, "Second superdiagonal array too small")
    assert(len(ipiv) >= n, "Pivot array too small")

    n_int := Blas_Int(n)

    when T == f32 {
        lapack.sgttrf_(&n_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(du2), raw_data(ipiv), &info)
    } else when T == f64 {
        lapack.dgttrf_(&n_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(du2), raw_data(ipiv), &info)
    } else when T == complex64 {
        lapack.cgttrf_(&n_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(du2), raw_data(ipiv), &info)
    } else when T == complex128 {
        lapack.zgttrf_(&n_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(du2), raw_data(ipiv), &info)
    }

    return info, info == 0
}


// ===================================================================================
// TRIDIAGONAL SOLVE USING FACTORIZATION (GTTRS)
// ===================================================================================

// Solve tridiagonal system using LU factorization from GTTRF
trid_solve_factorized :: proc(
    dl: []$T, // Factorized subdiagonal from GTTRF
    d: []T, // Factorized diagonal from GTTRF
    du: []T, // Factorized superdiagonal from GTTRF
    du2: []T, // Second superdiagonal from GTTRF
    ipiv: []Blas_Int, // Pivot indices from GTTRF
    B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
    trans := TransposeMode.None,
) -> (
    info: Info,
    ok: bool,
) where is_numeric_type(T) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n, "B matrix too small")
    assert(len(dl) >= n - 1 || n <= 1, "Subdiagonal array too small")
    assert(len(du) >= n - 1 || n <= 1, "Superdiagonal array too small")
    assert(len(du2) >= n - 2 || n <= 2, "Second superdiagonal array too small")
    assert(len(ipiv) >= n, "Pivot array too small")

    trans_c := cast(u8)trans
    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld

    when T == f32 {
        lapack.sgttrs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            &info,
        )
    } else when T == f64 {
        lapack.dgttrs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            &info,
        )
    } else when T == complex64 {
        lapack.cgttrs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            &info,
        )
    } else when T == complex128 {
        lapack.zgttrs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            &info,
        )
    }

    return info, info == 0
}


// ===================================================================================
// TRIDIAGONAL DIRECT SOLVE (GTSV)
// ===================================================================================

// Solve tridiagonal system directly (factorization + solve in one call)
trid_solve :: proc(
    dl: []$T, // Subdiagonal (size n-1, modified during factorization)
    d: []T, // Diagonal (size n, modified during factorization)
    du: []T, // Superdiagonal (size n-1, modified during factorization)
    B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
) -> (
    info: Info,
    ok: bool,
) where is_numeric_type(T) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n, "B matrix too small")
    assert(len(dl) >= n - 1 || n <= 1, "Subdiagonal array too small")
    assert(len(du) >= n - 1 || n <= 1, "Superdiagonal array too small")

    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld

    when T == f32 {
        lapack.sgtsv_(&n_int, &nrhs_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(B.data), &ldb, &info)
    } else when T == f64 {
        lapack.dgtsv_(&n_int, &nrhs_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(B.data), &ldb, &info)
    } else when T == complex64 {
        lapack.cgtsv_(&n_int, &nrhs_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(B.data), &ldb, &info)
    } else when T == complex128 {
        lapack.zgtsv_(&n_int, &nrhs_int, raw_data(dl), raw_data(d), raw_data(du), raw_data(B.data), &ldb, &info)
    }

    return info, info == 0
}


// ===================================================================================
// CONDITION NUMBER ESTIMATION (GTCON)
// ===================================================================================

// Query workspace for condition number estimation
query_workspace_trid_condition :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
    when is_float(T) {
        // Real types need both real and integer workspace
        return n, n
    } else when is_complex(T) {
        // Complex types only need complex workspace
        return n, 0
    }
    return 0, 0 // fallback
}

// Estimate condition number of factorized tridiagonal matrix
trid_condition :: proc {
    trid_condition_real,
    trid_condition_complex,
}

// Estimate condition number for real matrices
trid_condition_real :: proc(
    dl: []$T, // Factorized subdiagonal from GTTRF
    d: []T, // Factorized diagonal from GTTRF
    du: []T, // Factorized superdiagonal from GTTRF
    du2: []T, // Second superdiagonal from GTTRF
    ipiv: []Blas_Int, // Pivot indices from GTTRF
    anorm: T, // 1-norm or infinity-norm of original matrix
    work: []T, // Pre-allocated workspace (size n)
    iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
    norm := MatrixNorm.OneNorm,
) -> (
    rcond: T,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := len(d)
    assert(len(dl) >= n - 1 || n <= 1, "Subdiagonal array too small")
    assert(len(du) >= n - 1 || n <= 1, "Superdiagonal array too small")
    assert(len(du2) >= n - 2 || n <= 2, "Second superdiagonal array too small")
    assert(len(ipiv) >= n, "Pivot array too small")
    assert(len(work) >= n, "Work array too small")
    assert(len(iwork) >= n, "Integer work array too small")

    norm_c := cast(u8)norm
    n_int := Blas_Int(n)
    anorm_val := anorm

    when T == f32 {
        lapack.sgtcon_(
            &norm_c,
            &n_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            &anorm_val,
            &rcond,
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgtcon_(
            &norm_c,
            &n_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            &anorm_val,
            &rcond,
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    }

    return rcond, info, info == 0
}

// Estimate condition number for complex matrices
trid_condition_complex :: proc(
    dl: []$Cmplx, // Factorized subdiagonal from GTTRF
    d: []Cmplx, // Factorized diagonal from GTTRF
    du: []Cmplx, // Factorized superdiagonal from GTTRF
    du2: []Cmplx, // Second superdiagonal from GTTRF
    ipiv: []Blas_Int, // Pivot indices from GTTRF
    anorm: $Real, // 1-norm or infinity-norm of original matrix
    work: []Cmplx, // Pre-allocated workspace (size n)
    norm := MatrixNorm.OneNorm,
) -> (
    rcond: Real,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := len(d)
    assert(len(dl) >= n - 1 || n <= 1, "Subdiagonal array too small")
    assert(len(du) >= n - 1 || n <= 1, "Superdiagonal array too small")
    assert(len(du2) >= n - 2 || n <= 2, "Second superdiagonal array too small")
    assert(len(ipiv) >= n, "Pivot array too small")
    assert(len(work) >= n, "Work array too small")

    norm_c := cast(u8)norm
    n_int := Blas_Int(n)
    anorm_val := anorm

    when Cmplx == complex64 {
        lapack.cgtcon_(
            &norm_c,
            &n_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            &anorm_val,
            &rcond,
            raw_data(work),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgtcon_(
            &norm_c,
            &n_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(du2),
            raw_data(ipiv),
            &anorm_val,
            &rcond,
            raw_data(work),
            &info,
        )
    }

    return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (GTRFS)
// ===================================================================================

// Query workspace for iterative refinement
query_workspace_trid_refine :: proc(
    $T: typeid,
    nrhs: int,
) -> (
    work_size: int,
    rwork_size: int,
    iwork_size: int,
    ferr_size: int,
    berr_size: int,
) {
    when is_float(T) {
        // Real types need work and iwork
        return 3 * nrhs, 0, nrhs, nrhs, nrhs
    } else when is_complex(T) {
        // Complex types need work and rwork
        return 2 * nrhs, nrhs, 0, nrhs, nrhs
    }
    return 0, 0, 0, 0, 0 // fallback
}

// Perform iterative refinement for tridiagonal system
trid_refine :: proc {
    trid_refine_real,
    trid_refine_complex,
}

// Iterative refinement for real matrices
trid_refine_real :: proc(
    dl: []$T, // Original subdiagonal
    d: []T, // Original diagonal
    du: []T, // Original superdiagonal
    dlf: []T, // Factorized subdiagonal from GTTRF
    df: []T, // Factorized diagonal from GTTRF
    duf: []T, // Factorized superdiagonal from GTTRF
    du2: []T, // Second superdiagonal from GTTRF
    ipiv: []Blas_Int, // Pivot indices from GTTRF
    B: ^Matrix(T), // Original right-hand side
    X: ^Matrix(T), // Solution (input initial guess, output refined solution)
    ferr: []T, // Forward error bounds (size nrhs)
    berr: []T, // Backward error bounds (size nrhs)
    work: []T, // Pre-allocated workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    trans := TransposeMode.None,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
    assert(B.cols == X.cols, "Number of right-hand sides must match")
    assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

    trans_c := cast(u8)trans
    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld
    ldx := X.ld

    when T == f32 {
        lapack.sgtrfs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgtrfs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    }

    return info, info == 0
}

// Iterative refinement for complex matrices
trid_refine_complex :: proc(
    dl: []$Cmplx, // Original subdiagonal
    d: []Cmplx, // Original diagonal
    du: []Cmplx, // Original superdiagonal
    dlf: []Cmplx, // Factorized subdiagonal from GTTRF
    df: []Cmplx, // Factorized diagonal from GTTRF
    duf: []Cmplx, // Factorized superdiagonal from GTTRF
    du2: []Cmplx, // Second superdiagonal from GTTRF
    ipiv: []Blas_Int, // Pivot indices from GTTRF
    B: ^Matrix(Cmplx), // Original right-hand side
    X: ^Matrix(Cmplx), // Solution (input initial guess, output refined solution)
    ferr: []$Real, // Forward error bounds (size nrhs)
    berr: []Real, // Backward error bounds (size nrhs)
    work: []Cmplx, // Pre-allocated workspace
    rwork: []Real, // Pre-allocated real workspace
    trans := TransposeMode.None,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
    assert(B.cols == X.cols, "Number of right-hand sides must match")
    assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

    trans_c := cast(u8)trans
    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld
    ldx := X.ld

    when Cmplx == complex64 {
        lapack.cgtrfs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgtrfs_(
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// EXPERT DRIVER (GTSVX)
// ===================================================================================

// Expert driver for solving tridiagonal systems with optional equilibration,
// factorization, condition estimation, and error bounds
trid_solve_expert :: proc {
    trid_solve_expert_real,
    trid_solve_expert_complex,
}

// Expert driver for real matrices
trid_solve_expert_real :: proc(
    dl: []$T, // Subdiagonal (size n-1)
    d: []T, // Diagonal (size n)
    du: []T, // Superdiagonal (size n-1)
    dlf: []T, // Factorized subdiagonal (output if fact='N', input if fact='F')
    df: []T, // Factorized diagonal (output if fact='N', input if fact='F')
    duf: []T, // Factorized superdiagonal (output if fact='N', input if fact='F')
    du2: []T, // Second superdiagonal (output if fact='N', input if fact='F')
    ipiv: []Blas_Int, // Pivot indices (output if fact='N', input if fact='F')
    B: ^Matrix(T), // Right-hand side matrix
    X: ^Matrix(T), // Solution matrix (output)
    ferr: []T, // Forward error bounds (output, size nrhs)
    berr: []T, // Backward error bounds (output, size nrhs)
    work: []T, // Pre-allocated workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    fact := FactorizationOption.Factor,
    trans := TransposeMode.None,
) -> (
    rcond: T,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
    assert(B.cols == X.cols, "Number of right-hand sides must match")
    assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

    fact_c := cast(u8)fact
    trans_c := cast(u8)trans
    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld
    ldx := X.ld

    when T == f32 {
        lapack.sgtsvx_(
            &fact_c,
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            &rcond,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgtsvx_(
            &fact_c,
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            &rcond,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    }

    return rcond, info, info == 0
}

// Expert driver for complex matrices
trid_solve_expert_complex :: proc(
    dl: []$Cmplx, // Subdiagonal (size n-1)
    d: []Cmplx, // Diagonal (size n)
    du: []Cmplx, // Superdiagonal (size n-1)
    dlf: []Cmplx, // Factorized subdiagonal (output if fact='N', input if fact='F')
    df: []Cmplx, // Factorized diagonal (output if fact='N', input if fact='F')
    duf: []Cmplx, // Factorized superdiagonal (output if fact='N', input if fact='F')
    du2: []Cmplx, // Second superdiagonal (output if fact='N', input if fact='F')
    ipiv: []Blas_Int, // Pivot indices (output if fact='N', input if fact='F')
    B: ^Matrix(Cmplx), // Right-hand side matrix
    X: ^Matrix(Cmplx), // Solution matrix (output)
    ferr: []$Real, // Forward error bounds (output, size nrhs)
    berr: []Real, // Backward error bounds (output, size nrhs)
    work: []Cmplx, // Pre-allocated workspace
    rwork: []Real, // Pre-allocated real workspace
    fact := FactorizationOption.Factor,
    trans := TransposeMode.None,
) -> (
    rcond: Real,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := len(d)
    nrhs := B.cols
    assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
    assert(B.cols == X.cols, "Number of right-hand sides must match")
    assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

    fact_c := cast(u8)fact
    trans_c := cast(u8)trans
    n_int := Blas_Int(n)
    nrhs_int := Blas_Int(nrhs)
    ldb := B.ld
    ldx := X.ld

    when Cmplx == complex64 {
        lapack.cgtsvx_(
            &fact_c,
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            &rcond,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgtsvx_(
            &fact_c,
            &trans_c,
            &n_int,
            &nrhs_int,
            raw_data(dl),
            raw_data(d),
            raw_data(du),
            raw_data(dlf),
            raw_data(df),
            raw_data(duf),
            raw_data(du2),
            raw_data(ipiv),
            raw_data(B.data),
            &ldb,
            raw_data(X.data),
            &ldx,
            &rcond,
            raw_data(ferr),
            raw_data(berr),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    }

    return rcond, info, info == 0
}
