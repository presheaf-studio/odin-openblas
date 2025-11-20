package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// TRIANGULAR MATRIX OPERATIONS AND CONDITION NUMBER ESTIMATION
// ===================================================================================
//
// This file provides condition number estimation and other operations for
// full-storage triangular matrices:
// - Condition number estimation (TRCON)
// - Matrix norm computations
// - Well-conditioning checks
//
// All functions use the non-allocating API pattern with pre-allocated arrays.

// ===================================================================================
// CONDITION NUMBER ESTIMATION (TRCON)
// ===================================================================================

// Estimate condition number of triangular matrix
estimate_condition_triangular :: proc {
    tri_condition_estimate_real,
    tri_condition_estimate_complex,
}

// Real triangular condition number estimation (f32/f64)
tri_condition_estimate_real :: proc(
    A: Triangular($T), // Triangular matrix [n×n]
    work: []T, // Pre-allocated workspace (size 3*n)
    iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
    norm: MatrixNorm = .OneNorm, // Norm type
    uplo: MatrixRegion = .Upper, // Upper or lower triangular
    diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
    rcond: T,
    info: Info,
    ok: bool,
) where is_float(T) {
    assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
    assert(len(work) >= 3 * n, "Workspace too small")
    assert(len(iwork) >= n, "Integer workspace too small")

    norm_c := u8(norm)
    uplo_c := u8(uplo)
    diag_c := u8(diag)
    n_blas := A.n
    lda_blas := A.lda

    when T == f32 {
        lapack.strcon_(
            &norm_c,
            &uplo_c,
            &diag_c,
            &n_blas,
            raw_data(A),
            &lda_blas,
            &rcond,
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dtrcon_(
            &norm_c,
            &uplo_c,
            &diag_c,
            &n_blas,
            raw_data(A),
            &lda_blas,
            &rcond,
            raw_data(work),
            raw_data(iwork),
            &info,
        )
    }

    return rcond, info, info == 0
}

// Complex triangular condition number estimation (complex64/complex128)
tri_condition_estimate_complex :: proc(
    A: Triangular($Cmplx), // Triangular matrix [n×n]
    work: []Cmplx, // Pre-allocated workspace (size 2*n)
    rwork: []$Real, // Pre-allocated real workspace (size n)
    norm: MatrixNorm = .OneNorm, // Norm type
    uplo: MatrixRegion = .Upper, // Upper or lower triangular
    diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
    rcond: Real,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
    assert(len(work) >= 2 * n, "Workspace too small")
    assert(len(rwork) >= n, "Real workspace too small")

    norm_c := u8(norm)
    uplo_c := u8(uplo)
    diag_c := u8(diag)
    n_blas := Blas_Int(A.n)
    lda_blas := Blas_Int(A.lda)

    when Cmplx == complex64 {
        lapack.ctrcon_(
            &norm_c,
            &uplo_c,
            &diag_c,
            &n_blas,
            raw_data(A),
            &lda_blas,
            &rcond,
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztrcon_(
            &norm_c,
            &uplo_c,
            &diag_c,
            &n_blas,
            raw_data(A),
            &lda_blas,
            &rcond,
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    }

    return rcond, info, info == 0
}

// ===================================================================================
// MATRIX NORM COMPUTATIONS
// ===================================================================================

// Compute 1-norm of triangular matrix
one_norm_triangular :: proc(
    A: Triangular($T),
    res_out: ^$R,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
) where (is_float(T) && R == T) ||
    (is_complex(T) && ((T == complex64 && R == f32) || (T == complex128 && R == f64))) {
    max_col_sum := R(0)

    switch uplo {
    case .Upper:
        for j in 0 ..< A.n {
            col_sum := R(0)
            for i in 0 ..= j {
                if diag == .Unit && i == j {
                    col_sum += R(1)
                } else {
                    col_sum += abs(A.data[i + j * A.lda])
                }
            }
            max_col_sum = max(max_col_sum, col_sum)
        }
    case .Lower:
        for j in 0 ..< n {
            col_sum := R(0)
            for i in j ..< n {
                if diag == .Unit && i == j {
                    col_sum += R(1)
                } else {
                    col_sum += abs(A.data[i + j * A.lda])
                }
            }
            max_col_sum = max(max_col_sum, col_sum)
        }
    case .Full:
        panic("Full storage not supported for triangular matrices")
    }

    res_out^ = max_col_sum
}

// Compute infinity-norm of triangular matrix
infinity_norm_triangular :: proc(
    A: Triangular($T),
    res_out: ^$R,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
) where (is_float(T) && R == T) ||
    (is_complex(T) && ((T == complex64 && R == f32) || (T == complex128 && R == f64))) {
    max_row_sum := R(0)

    switch uplo {
    case .Upper:
        for i in 0 ..< A.n {
            row_sum := R(0)
            for j in i ..< A.n {
                if diag == .Unit && i == j {
                    row_sum += R(1)
                } else {
                    row_sum += abs(A.data[i + j * A.lda])
                }
            }
            max_row_sum = max(max_row_sum, row_sum)
        }
    case .Lower:
        for i in 0 ..< A.n {
            row_sum := R(0)
            for j in 0 ..= i {
                if diag == .Unit && i == j {
                    row_sum += R(1)
                } else {
                    row_sum += abs(A.data[i + j * A.lda])
                }
            }
            max_row_sum = max(max_row_sum, row_sum)
        }
    case .Full:
        panic("Full storage not supported for triangular matrices")
    }

    res_out^ = max_row_sum
}

// Compute max-norm (largest absolute value) of triangular matrix
max_norm_triangular :: proc(
    A: Triangular($T),
    res_out: ^$R,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
) where (is_float(T) && R == T) ||
    (is_complex(T) && ((T == complex64 && R == f32) || (T == complex128 && R == f64))) {
    max_val := R(0)

    switch uplo {
    case .Upper:
        for j in 0 ..< A.n {
            for i in 0 ..= j {
                if diag == .Unit && i == j {
                    max_val = max(max_val, R(1))
                } else {
                    max_val = max(max_val, abs(A.data[i + j * A.lda]))
                }
            }
        }
    case .Lower:
        for j in 0 ..< A.n {
            for i in j ..< A.n {
                if diag == .Unit && i == j {
                    max_val = max(max_val, R(1))
                } else {
                    max_val = max(max_val, abs(A.data[i + j * lda]))
                }
            }
        }
    case .Full:
        panic("Full storage not supported for triangular matrices")
    }

    res_out^ = max_val
}

// Compute specified norm of triangular matrix
norm_triangular :: proc(
    A: Triangular($T),
    norm_type: MatrixNorm,
    res_out: ^$R,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
) where (is_float(T) && R == T) ||
    (is_complex(T) && ((T == complex64 && R == f32) || (T == complex128 && R == f64))) {
    switch norm_type {
    case .OneNorm:
        one_norm_triangular(A, res_out, uplo, diag)
    case .InfinityNorm:
        infinity_norm_triangular(A, res_out, uplo, diag)
    case .MaxNorm:
        max_norm_triangular(A, res_out, uplo, diag)
    case .FrobeniusNorm:
        res_out^ = frobenius_norm_triangular(&A)
    }
}

// ===================================================================================
// WELL-CONDITIONING CHECKS
// ===================================================================================

// Note: is_well_conditioned_triangular is defined in packed_triangular.odin

// Estimate condition number and return both reciprocal and actual condition number
estimate_condition_with_actual :: proc {
    tri_condition_with_actual_real,
    tri_condition_with_actual_complex,
}

tri_condition_with_actual_real :: proc(
    A: Triangular($T),
    work: []T,
    iwork: []Blas_Int,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
    norm_type: MatrixNorm = .OneNorm,
    allocator := context.allocator,
) -> (
    rcond: T,
    cond: T,
    ok: bool,
) where is_float(T) {
    rcond_val, info, success := tri_condition_estimate_real(A, work, iwork, norm_type, uplo, diag)
    if !success || rcond_val <= 0 {
        return 0, max(T), false
    }
    return rcond_val, 1.0 / rcond_val, true
}

tri_condition_with_actual_complex :: proc(
    A: Triangular($Cmplx),
    work: Cmplx,
    rwork: $Real,
    rcond_out: ^Real,
    cond_out: ^Real,
    uplo: MatrixRegion = .Upper,
    diag: DiagonalType = .NonUnit,
    norm_type: MatrixNorm = .OneNorm,
    allocator := context.allocator,
) -> (
    ok: bool,
) where is_complex(Cmplx) &&
    ((Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)) {
    rcond_val, info, success := tri_condition_estimate_complex(A, work, rwork, n, lda, norm_type, uplo, diag)
    if !success || rcond_val <= 0 {
        rcond_out^ = 0
        cond_out^ = max(Real)
        return false
    }
    rcond_out^ = rcond_val
    cond_out^ = 1.0 / rcond_val
    return true
}

// ===================================================================================
// SYLVESTER EQUATION SOLVER (TRSYL)
// ===================================================================================

// Solve the triangular Sylvester equation: op(A)*X ± X*op(B) = scale*C
// where op(M) can be M, M^T, or M^H
solve_sylvester :: proc {
    tri_solve_sylvester_real,
    tri_solve_sylvester_complex,
}

// Real Sylvester equation solver (f32/f64)
tri_solve_sylvester_real :: proc(
    A: []$T, // Triangular matrix A [m×m]
    B: []T, // Triangular matrix B [n×n]
    C: []T, // Right-hand side matrix C [m×n] (overwritten with solution X)
    m, n: int, // Matrix dimensions
    lda, ldb, ldc: int, // Leading dimensions
    trana: TransposeMode = .None, // Transpose operation for A
    tranb: TransposeMode = .None, // Transpose operation for B
    isgn: int = 1, // Sign in equation: +1 for plus, -1 for minus
) -> (
    scale: T,
    info: Info,
    ok: bool, // Scaling factor applied to C
) where is_float(T) {
    assert(len(A) >= m * lda, "A array too small")
    assert(len(B) >= n * ldb, "B array too small")
    assert(len(C) >= m * ldc, "C array too small")
    assert(lda >= max(1, m), "Leading dimension lda too small")
    assert(ldb >= max(1, n), "Leading dimension ldb too small")
    assert(ldc >= max(1, m), "Leading dimension ldc too small")
    assert(isgn == 1 || isgn == -1, "isgn must be 1 or -1")

    trana_c := u8(trana)
    tranb_c := u8(tranb)
    m_blas := Blas_Int(m)
    n_blas := Blas_Int(n)
    lda_blas := Blas_Int(lda)
    ldb_blas := Blas_Int(ldb)
    ldc_blas := Blas_Int(ldc)
    isgn_blas := Blas_Int(isgn)

    when T == f32 {
        lapack.strsyl_(
            &trana_c,
            &tranb_c,
            &isgn_blas,
            &m_blas,
            &n_blas,
            raw_data(A),
            &lda_blas,
            raw_data(B),
            &ldb_blas,
            raw_data(C),
            &ldc_blas,
            &scale,
            &info,
        )
    } else when T == f64 {
        lapack.dtrsyl_(
            &trana_c,
            &tranb_c,
            &isgn_blas,
            &m_blas,
            &n_blas,
            raw_data(A),
            &lda_blas,
            raw_data(B),
            &ldb_blas,
            raw_data(C),
            &ldc_blas,
            &scale,
            &info,
        )
    }

    return scale, info, info == 0
}

// Complex Sylvester equation solver (complex64/complex128)
tri_solve_sylvester_complex :: proc(
    A: []$Cmplx, // Triangular matrix A [m×m]
    B: []Cmplx, // Triangular matrix B [n×n]
    C: []Cmplx, // Right-hand side matrix C [m×n] (overwritten with solution X)
    m, n: int, // Matrix dimensions
    lda, ldb, ldc: int, // Leading dimensions
    scale_out: ^$Real, // Output: scaling factor applied to C (real type)
    trana: TransposeMode = .None, // Transpose operation for A
    tranb: TransposeMode = .None, // Transpose operation for B
    isgn: int = 1, // Sign in equation: +1 for plus, -1 for minus
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) && ((Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)) {
    assert(len(A) >= m * lda, "A array too small")
    assert(len(B) >= n * ldb, "B array too small")
    assert(len(C) >= m * ldc, "C array too small")
    assert(lda >= max(1, m), "Leading dimension lda too small")
    assert(ldb >= max(1, n), "Leading dimension ldb too small")
    assert(ldc >= max(1, m), "Leading dimension ldc too small")
    assert(isgn == 1 || isgn == -1, "isgn must be 1 or -1")

    trana_c := u8(trana)
    tranb_c := u8(tranb)
    m_blas := Blas_Int(m)
    n_blas := Blas_Int(n)
    lda_blas := Blas_Int(lda)
    ldb_blas := Blas_Int(ldb)
    ldc_blas := Blas_Int(ldc)
    isgn_blas := Blas_Int(isgn)

    scale: Real

    when Cmplx == complex64 {
        lapack.ctrsyl_(
            &trana_c,
            &tranb_c,
            &isgn_blas,
            &m_blas,
            &n_blas,
            raw_data(A),
            &lda_blas,
            raw_data(B),
            &ldb_blas,
            raw_data(C),
            &ldc_blas,
            &scale,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztrsyl_(
            &trana_c,
            &tranb_c,
            &isgn_blas,
            &m_blas,
            &n_blas,
            raw_data(A),
            &lda_blas,
            raw_data(B),
            &ldb_blas,
            raw_data(C),
            &ldc_blas,
            &scale,
            &info,
        )
    }

    scale_out^ = scale
    return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace size for condition number estimation
query_workspace_condition_triangular :: proc(
    A: Triangular($T),
) -> (
    work_size: int,
    iwork_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.n
    when is_float(T) {
        return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork (per n)
    } else when is_complex(T) {
        return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork (per n)
    }
}
