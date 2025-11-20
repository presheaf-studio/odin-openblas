package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// HERMITIAN EIGENVALUE SOLVERS - NON-ALLOCATING API
// ============================================================================
// Complex Hermitian matrices: all eigenvalues are real
// Using standard QR algorithm and more advanced methods

// Query workspace for Hermitian eigenvalue computation (QR algorithm)
// Returns both work and rwork sizes (rwork has fixed size max(1, 3*n-2))
query_workspace_dns_eigen_hermitian :: proc(
    A: ^Matrix($Cmplx),
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    rwork_size: int,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    lwork := QUERY_WORKSPACE
    info: Info
    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.cheev_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheev_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = max(1, 3 * int(n) - 2)
    return work_size, rwork_size
}

// Compute eigenvalues and eigenvectors of Hermitian matrix using QR algorithm
dns_eigen_hermitian :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
    W: []$Real, // Output eigenvalues (length n, real values)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")

    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cheev_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheev_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ============================================================================
// DIVIDE AND CONQUER ALGORITHM - More efficient for large matrices
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (Divide and Conquer)
query_workspace_dns_eigen_hermitian_dc :: proc(
    A: ^Matrix($Cmplx),
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    rwork_size: int,
    iwork_size: int,
) where is_complex(Cmplx) {
    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    lwork := QUERY_WORKSPACE
    lrwork := QUERY_WORKSPACE
    liwork := QUERY_WORKSPACE
    info: Info

    work_query: Cmplx
    rwork_query: Real
    iwork_query: Blas_Int

    when Cmplx == complex64 {
        lapack.cheevd_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevd_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = int(rwork_query)
    iwork_size = int(iwork_query)
    return work_size, rwork_size, iwork_size
}

// Compute eigenvalues and eigenvectors using Divide and Conquer algorithm
dns_eigen_hermitian_dc :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
    W: []$Real, // Output eigenvalues (length n, real values)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) > 0, "Real workspace required")
    assert(len(iwork) > 0, "Integer workspace required")

    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    lda := A.ld
    lwork := Blas_Int(len(work))
    lrwork := Blas_Int(len(rwork))
    liwork := Blas_Int(len(iwork))

    when Cmplx == complex64 {
        lapack.cheevd_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevd_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    }

    return info, info == 0
}

// ============================================================================
// RELATIVELY ROBUST REPRESENTATIONS (RRR) - Most advanced algorithm
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (RRR algorithm)
query_workspace_dns_eigen_hermitian_rrr :: proc(
    A: ^Matrix($Cmplx),
    Z: ^Matrix(Cmplx),
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size, rwork_size, iwork_size: int,
) where is_complex(Cmplx) {
    jobz_c := cast(u8)jobz
    range_c := cast(u8)range
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    ldz := Z.ld
    lwork := QUERY_WORKSPACE
    lrwork := QUERY_WORKSPACE
    liwork := QUERY_WORKSPACE
    info: Info

    // Dummy values for query
    vl: Real = 0
    vu: Real = 0
    il: Blas_Int = 1
    iu: Blas_Int = n
    abstol: Real = 0
    m: Blas_Int = 0

    work_query: Cmplx
    rwork_query: Real
    iwork_query: Blas_Int

    when Cmplx == complex64 {
        lapack.cheevr_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz,
            nil, // W, Z, ISUPPZ
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevr_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz,
            nil, // W, Z, ISUPPZ
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = int(rwork_query)
    iwork_size = int(iwork_query)
    return work_size, rwork_size, iwork_size
}

// Compute eigenvalues and eigenvectors using RRR algorithm (most robust)
dns_eigen_hermitian_rrr :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (preserved)
    W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
    Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
    ISUPPZ: []Blas_Int, // Support indices (length 2*max(1,m))
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    vl: Real, // Value range (if range == 'V')
    vu: Real,
    il: int, // Index range (if range == 'I'), 1-based
    iu: int,
    abstol: Real, // Absolute tolerance for eigenvalues
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    m: int,
    info: Info,
    ok: bool, // Number of eigenvalues found
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) > 0, "Real workspace required")
    assert(len(iwork) > 0, "Integer workspace required")

    jobz_c := cast(u8)jobz
    range_c := cast(u8)range
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldz := Z.ld
    lwork := Blas_Int(len(work))
    lrwork := Blas_Int(len(rwork))
    liwork := Blas_Int(len(iwork))
    il_int := Blas_Int(il)
    iu_int := Blas_Int(iu)
    m_int: Blas_Int

    when Cmplx == complex64 {
        lapack.cheevr_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(ISUPPZ),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevr_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(ISUPPZ),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    }

    return int(m_int), info, info == 0
}

// ============================================================================
// SELECTED EIGENVALUES - Legacy Expert Driver (HEEVX)
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (Expert driver)
// NOTE: Must Pre-allocate `rwork` for query: `make([]Real, 7 * n)`
query_workspace_dns_eigen_hermitian_expert :: proc(
    A: ^Matrix($Cmplx),
    Z: ^Matrix(Cmplx),
    rwork: []$Real,
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    iwork_size: int,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    jobz_c := cast(u8)jobz
    range_c := cast(u8)range
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    ldz := Z.ld
    lwork := QUERY_WORKSPACE
    info: Info

    // Dummy values for query
    vl: Real = 0
    vu: Real = 0
    il: Blas_Int = 1
    iu: Blas_Int = n
    abstol: Real = 0
    m: Blas_Int = 0

    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.cheevx_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz, // W, Z
            &work_query,
            &lwork,
            raw_data(rwork),
            nil,
            nil, // rwork, iwork, IFAIL
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevx_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz, // W, Z
            &work_query,
            &lwork,
            raw_data(rwork),
            nil,
            nil, // rwork, iwork, IFAIL
            &info,
        )
    }

    work_size = int(real(work_query))
    iwork_size = 5 * n // Fixed size for HEEVX
    return work_size, iwork_size
}

// Compute selected eigenvalues and eigenvectors using expert driver
dns_eigen_hermitian_expert :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed)
    W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
    Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    IFAIL: []Blas_Int, // Failed convergence indices (length n)
    vl: Real, // Value range (if range == 'V')
    vu: Real,
    il: int, // Index range (if range == 'I'), 1-based
    iu: int,
    abstol: Real, // Absolute tolerance for eigenvalues
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    m: int,
    info: Info,
    ok: bool, // Number of eigenvalues found
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) >= 7 * n, "Real workspace too small")
    assert(len(iwork) >= 5 * n, "Integer workspace too small")
    assert(len(IFAIL) >= n, "IFAIL array too small")

    jobz_c := cast(u8)jobz
    range_c := cast(u8)range
    uplo_c := cast(u8)uplo
    n := n
    lda := A.ld
    ldz := Z.ld
    lwork := Blas_Int(len(work))
    il_int := Blas_Int(il)
    iu_int := Blas_Int(iu)
    m_int: Blas_Int

    when Cmplx == complex64 {
        lapack.cheevx_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(IFAIL),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheevx_(
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(IFAIL),
            &info,
        )
    }

    return int(m_int), info, info == 0
}

// ============================================================================
// TWO-STAGE ALGORITHMS - For very large matrices
// ============================================================================

// Query workspace for two-stage Hermitian eigenvalue computation
// Returns both work and rwork sizes (rwork has fixed size max(1, 3*n-2))
query_workspace_dns_eigen_hermitian_2stage :: proc(
    A: ^Matrix($Cmplx),
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    rwork_size: int,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    lwork := QUERY_WORKSPACE
    info: Info

    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.cheev_2stage_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheev_2stage_(
            &jobz_c,
            &uplo_c,
            &n,
            nil, // a
            &lda,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = max(1, 3 * int(n) - 2)
    return work_size, rwork_size
}

// Compute eigenvalues and eigenvectors using two-stage algorithm
dns_eigen_hermitian_2stage_complex :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
    W: []$Real, // Output eigenvalues (length n, real values)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")

    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    lda := A.ld
    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cheev_2stage_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zheev_2stage_(
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ============================================================================
// GENERALIZED HERMITIAN EIGENVALUE PROBLEMS
// ============================================================================
// Solve generalized eigenvalue problems of the form:
//   itype=1: A*x = lambda*B*x
//   itype=2: A*B*x = lambda*x
//   itype=3: B*A*x = lambda*x
// where A is Hermitian and B is Hermitian positive definite

// Query workspace for generalized Hermitian eigenvalue computation (QR algorithm)
// Returns both work and rwork sizes (rwork has fixed size max(1, 3*n-2))
query_workspace_dns_eigen_hermitian_generalized :: proc(
    A: ^Matrix($Cmplx),
    B: ^Matrix(Cmplx),
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    rwork_size: int,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    ldb := B.ld
    lwork := QUERY_WORKSPACE
    info: Info
    itype := Blas_Int(1)

    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.chegv_(
            &itype,
            &jobz_c,
            &uplo_c,
            &n,
            nil, // A
            &lda,
            nil, // B
            &ldb,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegv_(
            &itype,
            &jobz_c,
            &uplo_c,
            &n,
            nil, // A
            &lda,
            nil, // B
            &ldb,
            nil, // w
            &work_query,
            &lwork,
            nil,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = max(1, 3 * int(n) - 2)
    return work_size, rwork_size
}

// Compute generalized eigenvalues and eigenvectors using QR algorithm
dns_eigen_hermitian_generalized :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
    B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
    W: []$Real, // Output eigenvalues (length n, real values)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")
    assert(itype >= 1 && itype <= 3, "Invalid problem type")

    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldb := B.ld
    lwork := Blas_Int(len(work))
    itype_blas := Blas_Int(itype)

    when Cmplx == complex64 {
        lapack.chegv_(
            &itype_blas,
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegv_(
            &itype_blas,
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ============================================================================
// GENERALIZED DIVIDE AND CONQUER ALGORITHM
// ============================================================================

// Query workspace for generalized Hermitian eigenvalue computation (Divide and Conquer)
query_workspace_dns_eigen_hermitian_generalized_dc :: proc(
    A: ^Matrix($Cmplx),
    B: ^Matrix(Cmplx),
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    rwork_size: int,
    iwork_size: int,
) where is_complex(Cmplx) {
    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldb := B.ld
    lwork := QUERY_WORKSPACE
    lrwork := QUERY_WORKSPACE
    liwork := QUERY_WORKSPACE
    info: Info
    itype := Blas_Int(1)

    work_query: Cmplx
    rwork_query: Real
    iwork_query: Blas_Int

    when Cmplx == complex64 {
        lapack.chegvd_(
            &itype,
            &jobz_c,
            &uplo_c,
            &n,
            nil, // A
            &lda,
            nil, // B
            &ldb,
            nil, // w
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegvd_(
            &itype,
            &jobz_c,
            &uplo_c,
            &n,
            nil, // A
            &lda,
            nil, // B
            &ldb,
            nil, // w
            &work_query,
            &lwork,
            &rwork_query,
            &lrwork,
            &iwork_query,
            &liwork,
            &info,
        )
    }

    work_size = int(real(work_query))
    rwork_size = int(rwork_query)
    iwork_size = int(iwork_query)
    return work_size, rwork_size, iwork_size
}

// Compute generalized eigenvalues and eigenvectors using Divide and Conquer algorithm
dns_eigen_hermitian_generalized_dc :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
    B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
    W: []$Real, // Output eigenvalues (length n, real values)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) > 0, "Real workspace required")
    assert(len(iwork) > 0, "Integer workspace required")
    assert(itype >= 1 && itype <= 3, "Invalid problem type")

    jobz_c := cast(u8)jobz
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldb := B.ld
    lwork := Blas_Int(len(work))
    lrwork := Blas_Int(len(rwork))
    liwork := Blas_Int(len(iwork))
    itype_blas := Blas_Int(itype)

    when Cmplx == complex64 {
        lapack.chegvd_(
            &itype_blas,
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegvd_(
            &itype_blas,
            &jobz_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(W),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    }

    return info, info == 0
}

// ============================================================================
// GENERALIZED EXPERT DRIVER (HEGVX)
// ============================================================================

// Query workspace for generalized Hermitian eigenvalue computation (Expert driver)
// NOTE: Requires rwork allocated: `make([]Real, 7 * n)`
query_workspace_dns_eigen_hermitian_generalized_expert :: proc(
    A: ^Matrix($Cmplx),
    B: ^Matrix(Cmplx),
    Z: ^Matrix(Cmplx),
    rwork: $Real,
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
    iwork_size: int,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    jobz_c := cast(u8)jobz
    range_c: u8 = cast(u8)range
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldb := B.ld
    ldz := Z.ld
    lwork := QUERY_WORKSPACE
    info: Info
    itype := Blas_Int(1)

    // Dummy values for query
    vl: Real = 0
    vu: Real = 0
    il: Blas_Int = 1
    iu: Blas_Int = n
    abstol: Real = 0
    m: Blas_Int = 0

    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.chegvx_(
            &itype,
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            nil,
            &ldb, // B
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz, // W, Z
            &work_query,
            &lwork,
            raw_data(rwork),
            nil,
            nil, // rwork, iwork, IFAIL
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegvx_(
            &itype,
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            nil,
            &lda, // A
            nil,
            &ldb, // B
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &m,
            nil,
            nil,
            &ldz, // W, Z
            &work_query,
            &lwork,
            raw_data(rwork),
            nil,
            nil, // rwork, iwork, IFAIL
            &info,
        )
    }

    work_size = int(real(work_query))
    iwork_size = 5 * n // Fixed size for HEGVX
    return work_size, iwork_size
}

// Compute selected generalized eigenvalues and eigenvectors using expert driver
dns_eigen_hermitian_generalized_expert :: proc(
    A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed)
    B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
    W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
    Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
    work: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    IFAIL: []Blas_Int, // Failed convergence indices (length n)
    vl: Real, // Value range lower bound (if range == 'V')
    vu: Real, // Value range upper bound (if range == 'V')
    il: int = 1, // Index range lower bound (if range == 'I'), 1-based
    iu: int = 1, // Index range upper bound (if range == 'I'), 1-based
    abstol: Real, // Absolute tolerance for eigenvalues
    itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
    range: EigenRangeOption,
    jobz: EigenJobOption,
    uplo := MatrixRegion.Upper,
) -> (
    m: int,
    info: Info,
    ok: bool, // Number of eigenvalues found
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(W) >= n, "Eigenvalue array too small")
    assert(len(work) > 0, "Complex workspace required")
    assert(len(rwork) >= 7 * n, "Real workspace too small")
    assert(len(iwork) >= 5 * n, "Integer workspace too small")
    assert(len(IFAIL) >= n, "IFAIL array too small")
    assert(itype >= 1 && itype <= 3, "Invalid problem type")

    jobz_c := cast(u8)jobz
    range_c := cast(u8)range
    uplo_c := cast(u8)uplo
    lda := A.ld
    ldb := B.ld
    ldz := Z.ld
    lwork := Blas_Int(len(work))
    il_int := Blas_Int(il)
    iu_int := Blas_Int(iu)
    itype_blas := Blas_Int(itype)
    m_int: Blas_Int

    when Cmplx == complex64 {
        lapack.chegvx_(
            &itype_blas,
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(IFAIL),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhegvx_(
            &itype_blas,
            &jobz_c,
            &range_c,
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &vl,
            &vu,
            &il_int,
            &iu_int,
            &abstol,
            &m_int,
            raw_data(W),
            raw_data(Z.data),
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(IFAIL),
            &info,
        )
    }

    return int(m_int), info, info == 0
}
// ============================================================================
// HERMITIAN GENERALIZED EIGENVALUE REDUCTION (HEGST)
// ============================================================================
// Reduce generalized Hermitian eigenvalue problem to standard form
// Transforms A*x = lambda*B*x into C*y = lambda*y where C = inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H)

// Reduce generalized Hermitian eigenvalue problem to standard form (complex)
dns_hermitian_reduce_generalized :: proc(
    A: ^Matrix($Cmplx), // Input/output: Hermitian matrix A, overwritten with transformed matrix
    B: ^Matrix(Cmplx), // Input: Cholesky factor of B (from potrf)
    itype: int = 1, // Problem type: 1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x // FIXME: ENUM
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimension")
    assert(itype >= 1 && itype <= 3, "Invalid problem type (must be 1, 2, or 3)")

    uplo_c := cast(u8)uplo
    itype_blas := Blas_Int(itype)
    lda := A.ld
    ldb := B.ld

    when Cmplx == complex64 {
        lapack.chegst_(&itype_blas, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
    } else when Cmplx == complex128 {
        lapack.zhegst_(&itype_blas, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
    }

    return info, info == 0
}

// ============================================================================
// HERMITIAN TRIDIAGONAL REDUCTION (HETRD)
// ============================================================================
// Reduce Hermitian matrix to tridiagonal form using unitary similarity transformations

// Query workspace for Hermitian tridiagonal reduction
query_workspace_dns_hermitian_reduce_tridiagonal :: proc(
    A: ^Matrix($Cmplx),
    uplo := MatrixRegion.Upper,
) -> (
    work_size: int,
) where is_complex(Cmplx) {
    uplo_c := cast(u8)uplo
    n := A.cols
    lda := A.ld
    lwork := QUERY_WORKSPACE
    info: Info

    work_query: Cmplx

    when Cmplx == complex64 {
        lapack.chetrd_(&uplo_c, &n, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
    } else when Cmplx == complex128 {
        lapack.zhetrd_(&uplo_c, &n, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
    }

    work_size = int(real(work_query))
    return work_size
}

// Reduce Hermitian matrix to tridiagonal form (complex)
// Output tridiagonal matrix has real diagonal and real off-diagonal (symmetric tridiagonal)
dns_hermitian_reduce_tridiagonal :: proc(
    A: ^Matrix($Cmplx), // Input/output: Hermitian matrix, overwritten with Q
    T: ^Tridiagonal($Real), // Output: Real symmetric tridiagonal matrix (d and dl/du are real)
    tau: []Cmplx, // Output: Scalar factors of elementary reflectors (length n-1)
    work: []Cmplx, // Pre-allocated workspace
    uplo := MatrixRegion.Upper,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    assert(A.rows == A.cols, "Matrix must be square")
    assert(T.n == n, "Tridiagonal matrix dimension mismatch")
    assert(len(T.d) >= int(n), "Tridiagonal diagonal array too small")
    assert(len(T.dl) >= int(n) - 1, "Tridiagonal off-diagonal array too small")
    assert(len(tau) >= int(n) - 1, "Tau array too small")
    assert(len(work) > 0, "Workspace required")

    uplo_c := cast(u8)uplo
    lda := A.ld
    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.chetrd_(
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(T.d),
            raw_data(T.dl),
            raw_data(tau),
            raw_data(work),
            &lwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhetrd_(
            &uplo_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(T.d),
            raw_data(T.dl),
            raw_data(tau),
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}
