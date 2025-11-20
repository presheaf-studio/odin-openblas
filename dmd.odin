package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// DYNAMIC MODE DECOMPOSITION (DMD)
// Data-driven method for analyzing dynamical systems
// ===================================================================================

DMD_Rank_Method :: enum u8 {
    Fixed     = 'F',
    Tolerance = 'T',
    Economy   = 'E',
}

// DMD computation options
DMDOptions :: struct {
    // Data selection
    compute_modes:         bool, // Compute DMD modes
    compute_eigenvalues:   bool, // Compute eigenvalues
    compute_residuals:     bool, // Compute residuals
    compute_reconstructed: bool, // Compute reconstructed dynamics

    // SVD method selection
    svd_method:            Blas_Int, // 1 = gesdd, 2 = gesvd, 3 = gesvdq, 4 = gesdd with compensation

    // Rank selection
    rank_method:           DMD_Rank_Method, // Fixed, Tolerance, or Economy
}

// Query workspace for DMD computation
query_workspace_dmd :: proc(
    X: ^Matrix($T),
    Y: ^Matrix(T),
    k: int,
    options: DMDOptions = {compute_modes = true, compute_eigenvalues = true, svd_method = 1, rank_method = .Tolerance},
) -> (
    work_size: int,
    iwork_size: int,
    rwork_size: int, // Real workspace for complex types
) where is_float(T) || is_complex(T) {
    m := X.rows
    n := X.cols

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    ldx := X.ld
    ldy := Y.ld
    k_int := Blas_Int(k)

    // Set job parameters
    jobs_c: u8 = options.compute_modes ? 'V' : 'N'
    jobz_c: u8 = options.compute_eigenvalues ? 'V' : 'N'
    jobr_c: u8 = u8(options.rank_method)
    jobf_c: u8 = options.compute_residuals ? 'V' : 'N'

    ldz := Blas_Int(m)
    ldb := k_int
    ldw := m_int
    lds := k_int
    lwork := QUERY_WORKSPACE
    liwork := QUERY_WORKSPACE
    nrnk_int: Blas_Int
    tol: T = 0
    info: Info
    iwork_query: Blas_Int
    work_query: T
    rwork_size = 0

    when T == f32 {
        lapack.sgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            nil, // eigenvalues_real
            nil, // eigenvalues_imag
            nil, // modes
            &ldz,
            nil, // residuals
            nil, // B
            &ldb,
            nil, // W
            &ldw,
            nil, // S
            &lds,
            &work_query,
            &lwork,
            &iwork_query,
            &liwork,
            &info,
        )
        work_size = max(1, int(work_query))
        iwork_size = max(1, int(iwork_query))
    } else when T == f64 {
        lapack.dgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            nil, // eigenvalues_real
            nil, // eigenvalues_imag
            nil, // modes
            &ldz,
            nil, // residuals
            nil, // B
            &ldb,
            nil, // W
            &ldw,
            nil, // S
            &lds,
            &work_query,
            &lwork,
            &iwork_query,
            &liwork,
            &info,
        )
        work_size = max(1, int(work_query))
        iwork_size = max(1, int(iwork_query))
    } else when T == complex64 {
        rwork_query: f32
        lapack.cgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            nil, // eigenvalues
            nil, // modes
            &ldz,
            nil, // residuals
            nil, // B
            &ldb,
            nil, // W
            &ldw,
            nil, // S
            &lds,
            &work_query,
            &lwork,
            &rwork_query,
            &lwork,
            &iwork_query,
            &liwork,
            &info,
        )
        work_size = int(real(work_query))
        iwork_size = int(iwork_query)
        rwork_size = int(rwork_query)
    } else when T == complex128 {
        rwork_query: f64
        lapack.zgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            nil, // eigenvalues
            nil, // modes
            &ldz,
            nil, // residuals
            nil, // B
            &ldb,
            nil, // W
            &ldw,
            nil, // S
            &lds,
            &work_query,
            &lwork,
            &rwork_query,
            &lwork,
            &iwork_query,
            &liwork,
            &info,
        )
        work_size = max(1, int(real(work_query)))
        iwork_size = max(1, int(iwork_query))
        rwork_size = max(1, int(rwork_query))
    }

    // Ensure minimum workspace
    if rwork_size < 1 && is_complex(T) {
        rwork_size = 1
    }

    return work_size, iwork_size, rwork_size
}

// Compute Dynamic Mode Decomposition (non-allocating API)
// Analyzes time series data to extract dynamical modes
dmd :: proc {
    dmd_real,
    dmd_complex,
}

// DMD for f32/f64
dmd_real :: proc(
    X: ^Matrix($T), // Snapshot matrix X = [x0, x1, ..., xn-1]
    Y: ^Matrix(T), // Shifted snapshots Y = [x1, x2, ..., xn]
    eigenvalues_real: []T, // Pre-allocated real part of eigenvalues (size k)
    eigenvalues_imag: []T, // Pre-allocated imaginary part of eigenvalues (size k)
    modes: ^Matrix(T), // Pre-allocated DMD modes matrix (m x k) - optional
    residuals: []T, // Pre-allocated residuals (size k) - optional
    B: ^Matrix(T), // Pre-allocated DMD approximation matrix (k x n)
    W: ^Matrix(T), // Pre-allocated modes in full space (m x k)
    S: ^Matrix(T), // Pre-allocated low-rank operator (k x k)
    work: []T, // Pre-allocated workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    k: int, // Target rank
    tol: T, // Tolerance for rank truncation
    options: DMDOptions = {compute_modes = true, compute_eigenvalues = true, svd_method = 1, rank_method = .Tolerance},
) -> (
    nrnk: int,
    info: Info,
    ok: bool, // Effective rank (output)
) where is_float(T) {
    m := X.rows
    n := X.cols

    assert(len(eigenvalues_real) >= k, "Real eigenvalues array too small")
    assert(len(eigenvalues_imag) >= k, "Imaginary eigenvalues array too small")
    assert(int(B.rows) >= k && B.cols >= n, "B matrix too small")
    assert(W.rows >= m && int(W.cols) >= k, "W matrix too small")
    assert(int(S.rows) >= k && int(S.cols) >= k, "S matrix too small")

    if options.compute_modes {
        assert(modes != nil && modes.rows >= m && int(modes.cols) >= k, "Modes matrix too small")
    }
    if options.compute_residuals {
        assert(len(residuals) >= k, "Residuals array too small")
    }

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    ldx := X.ld
    ldy := Y.ld
    k_int := Blas_Int(k)

    // Set job parameters
    jobs_c: u8 = options.compute_modes ? 'V' : 'N'
    jobz_c: u8 = options.compute_eigenvalues ? 'V' : 'N'
    jobr_c: u8 = u8(options.rank_method)
    jobf_c: u8 = options.compute_residuals ? 'V' : 'N'

    ldz := Blas_Int(modes.ld if modes != nil else m)
    ldb := Blas_Int(B.ld)
    ldw := Blas_Int(W.ld)
    lds := Blas_Int(S.ld)
    lwork := Blas_Int(len(work))
    liwork := Blas_Int(len(iwork))
    nrnk_int: Blas_Int
    tol := tol

    when T == f32 {
        lapack.sgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            raw_data(eigenvalues_real),
            raw_data(eigenvalues_imag),
            options.compute_modes ? raw_data(modes.data) : nil,
            &ldz,
            options.compute_residuals ? raw_data(residuals) : nil,
            raw_data(B.data),
            &ldb,
            raw_data(W.data),
            &ldw,
            raw_data(S.data),
            &lds,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            &info,
            1,
            1,
            1,
            1,
        )
    } else when T == f64 {
        lapack.dgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &options.svd_method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            raw_data(eigenvalues_real),
            raw_data(eigenvalues_imag),
            options.compute_modes ? raw_data(modes.data) : nil,
            &ldz,
            options.compute_residuals ? raw_data(residuals) : nil,
            raw_data(B.data),
            &ldb,
            raw_data(W.data),
            &ldw,
            raw_data(S.data),
            &lds,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            &info,
            1,
            1,
            1,
            1,
        )
    }

    nrnk = int(nrnk_int)
    ok = info == 0
    return nrnk, info, ok
}

// DMD for complex64/complex128
dmd_complex :: proc(
    X: ^Matrix($Cmplx), // Snapshot matrix
    Y: ^Matrix(Cmplx), // Shifted snapshots
    eigenvalues: []Cmplx, // Pre-allocated eigenvalues (size k)
    modes: ^Matrix(Cmplx), // Pre-allocated DMD modes matrix (m x k) - optional
    residuals: []$Real, // Pre-allocated residuals (size k) - optional
    B: ^Matrix(Cmplx), // Pre-allocated DMD approximation matrix (k x n)
    W: ^Matrix(Cmplx), // Pre-allocated modes in full space (m x k)
    S: ^Matrix(Cmplx), // Pre-allocated low-rank operator (k x k)
    zwork: []Cmplx, // Pre-allocated complex workspace
    rwork: []Real, // Pre-allocated real workspace
    iwork: []Blas_Int, // Pre-allocated integer workspace
    k: int, // Target rank
    tol: Real, // Tolerance for rank truncation
    options: DMDOptions = {compute_modes = true, compute_eigenvalues = true, svd_method = 1, rank_method = .Tolerance},
) -> (
    nrnk: int,
    info: Info,
    ok: bool, // Effective rank (output)
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    m := X.rows
    n := X.cols

    assert(len(eigenvalues) >= k, "Eigenvalues array too small")
    assert(int(B.rows) >= k && B.cols >= n, "B matrix too small")
    assert(W.rows >= m && int(W.cols) >= k, "W matrix too small")
    assert(int(S.rows) >= k && int(S.cols) >= k, "S matrix too small")

    if options.compute_modes {
        assert(modes != nil && modes.rows >= m && int(modes.cols) >= k, "Modes matrix too small")
    }
    if options.compute_residuals {
        assert(len(residuals) >= k, "Residuals array too small")
    }

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    ldx := X.ld
    ldy := Y.ld
    k_int := Blas_Int(k)
    method := options.svd_method

    // Set job parameters
    jobs_c: u8 = options.compute_modes ? 'V' : 'N'
    jobz_c: u8 = options.compute_eigenvalues ? 'V' : 'N'
    jobr_c: u8 = u8(options.rank_method)
    jobf_c: u8 = options.compute_residuals ? 'V' : 'N'

    ldz := Blas_Int(modes.ld if modes != nil else m)
    ldb := Blas_Int(B.ld)
    ldw := Blas_Int(W.ld)
    lds := Blas_Int(S.ld)
    lzwork := Blas_Int(len(zwork))
    lrwork := Blas_Int(len(rwork))
    liwork := Blas_Int(len(iwork))
    nrnk_int: Blas_Int
    tol := tol

    when T == complex64 {
        lapack.cgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            raw_data(eigenvalues),
            options.compute_modes ? raw_data(modes.data) : nil,
            &ldz,
            options.compute_residuals ? raw_data(residuals) : nil,
            raw_data(B.data),
            &ldb,
            raw_data(W.data),
            &ldw,
            raw_data(S.data),
            &lds,
            raw_data(zwork),
            &lzwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
            1,
            1,
            1,
            1,
        )
    } else when T == complex128 {
        lapack.zgedmd_(
            &jobs_c,
            &jobz_c,
            &jobr_c,
            &jobf_c,
            &method,
            &m_int,
            &n_int,
            raw_data(X.data),
            &ldx,
            raw_data(Y.data),
            &ldy,
            &nrnk_int,
            &tol,
            &k_int,
            raw_data(eigenvalues),
            options.compute_modes ? raw_data(modes.data) : nil,
            &ldz,
            options.compute_residuals ? raw_data(residuals) : nil,
            raw_data(B.data),
            &ldb,
            raw_data(W.data),
            &ldw,
            raw_data(S.data),
            &lds,
            raw_data(zwork),
            &lzwork,
            raw_data(rwork),
            &lrwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    }

    return int(nrnk_int), info, info == 0
}
