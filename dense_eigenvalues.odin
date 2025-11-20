package openblas

import lapack "./f77"
import "base:intrinsics"

// Job options for computing left eigenvectors
Eigen_Job_Left :: enum u8 {
    None    = 'N', // Left eigenvectors are not computed
    Compute = 'V', // Left eigenvectors are computed
}

// Job options for computing right eigenvectors
Eigen_Job_Right :: enum u8 {
    None    = 'N', // Right eigenvectors are not computed
    Compute = 'V', // Right eigenvectors are computed
}

// Job options for Schur vectors
Schur_Job :: enum u8 {
    None    = 'N', // Schur vectors are not computed
    Compute = 'V', // Schur vectors are computed
}

// Sort options for Schur form
Schur_Sort :: enum u8 {
    None = 'N', // Eigenvalues are not ordered
    Sort = 'S', // Eigenvalues are ordered (requires select function)
}

// Balance job for preprocessing matrices
Balance_Job :: enum u8 {
    None    = 'N', // No balancing
    Permute = 'P', // Permute only
    Scale   = 'S', // Scale only
    Both    = 'B', // Both permute and scale
}

// Sense options for condition numbers in expert drivers
Sense_Job :: enum u8 {
    None         = 'N', // No condition numbers computed
    Eigenvalues  = 'E', // Condition numbers for eigenvalues only
    Eigenvectors = 'V', // Condition numbers for eigenvectors only
    Both         = 'B', // Condition numbers for both eigenvalues and eigenvectors
}

// ===================================================================================
// STANDARD EIGENVALUE PROBLEMS (GEEV family)
// ===================================================================================

dns_eigen_general :: proc {
    dns_eigen_general_real,
    dns_eigen_general_complex,
}

dns_eigen_general_expert :: proc {
    dns_eigen_general_expert_real,
    dns_eigen_general_expert_complex,
}

// Query workspace size for eigenvalue computation
query_workspace_dns_eigen_general :: proc(
    A: ^Matrix($T),
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    work_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    lwork: Blas_Int = QUERY_WORKSPACE
    work_query: T
    info: Info
    rwork_size = 0

    when T == f32 {
        lapack.sgeev_(&jobvl_c, &jobvr_c, &n, nil, &lda, nil, nil, nil, &n, nil, &n, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        lapack.dgeev_(&jobvl_c, &jobvr_c, &n, nil, &lda, nil, nil, nil, &n, nil, &n, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        lapack.cgeev_(&jobvl_c, &jobvr_c, &n, nil, &lda, nil, nil, &n, nil, &n, &work_query, &lwork, nil, &info)
        work_size = int(real(work_query))
        rwork_size = int(2 * n)
    } else when T == complex128 {
        lapack.zgeev_(&jobvl_c, &jobvr_c, &n, nil, &lda, nil, nil, &n, nil, &n, &work_query, &lwork, nil, &info)
        work_size = int(real(work_query))
        rwork_size = int(2 * n)
    }

    return work_size, rwork_size
}

// Compute eigenvalues and eigenvectors of a real general matrix
dns_eigen_general_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(work) > 0, "work array must be provided")

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeev_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dgeev_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// Compute eigenvalues and eigenvectors of a complex general matrix
dns_eigen_general_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(2 * n), "rwork array too small")

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgeev_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgeev_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// EXPERT EIGENVALUE DRIVERS (GEEVX family)
// ===================================================================================

// Query workspace size for expert eigenvalue computation
query_workspace_dns_eigen_general_expert :: proc(
    A: ^Matrix($T),
    sense: Sense_Job = .None,
) -> (
    work_size: int,
    rwork_size: int,
    iwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows

    when is_float(T) {
        work_size = int(n * (n + 6))
        if sense == .Eigenvalues || sense == .Both {
            work_size = max(work_size, int(n * (n + 6)))
        }
        if sense == .Eigenvectors || sense == .Both {
            work_size = max(work_size, int(n * (n + 6)))
        }
        iwork_size = int(2 * n - 2)
        rwork_size = 0
    } else when is_complex(T) {
        work_size = int(2 * n)
        rwork_size = int(2 * n)
        iwork_size = 0
    }

    return work_size, rwork_size, iwork_size
}

// Expert eigenvalue driver with balancing and condition number estimation
// ilo:  lowest index of balanced matrix
// ihi: highest index of balanced matrix
// scale: Scaling factors (pre-allocated, size n)
// abnrm: 1-norm of balanced matrix
// rconde: Condition numbers of eigenvalues (pre-allocated, size n)
// rcondv: Condition numbers of eigenvectors (pre-allocated, size n)
dns_eigen_general_expert_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    balanc: Balance_Job = .Both,
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
    sense: Sense_Job = .None,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    scale: []T,
    abnrm: T,
    rconde: []T,
    rcondv: []T,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(scale) >= int(n), "scale array too small")
    assert(len(rconde) >= int(n), "rconde array too small")
    assert(len(rcondv) >= int(n), "rcondv array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) >= int(2 * n - 2), "iwork array too small")

    balanc_c := cast(u8)balanc
    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    sense_c := cast(u8)sense

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(scale),
            &abnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgeevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(scale),
            &abnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    }

    return ilo, ihi, scale, abnrm, rconde, rcondv, info, info == 0
}

dns_eigen_general_expert_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated)
    balanc: Balance_Job = .Both,
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
    sense: Sense_Job = .None,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    scale: []Real,
    abnrm: Real,
    rconde: []Real,
    rcondv: []Real,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(scale) >= int(n), "scale array too small")
    assert(len(rconde) >= int(n), "rconde array too small")
    assert(len(rcondv) >= int(n), "rcondv array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(2 * n), "rwork array too small")

    balanc_c := cast(u8)balanc
    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    sense_c := cast(u8)sense

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgeevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(scale),
            &abnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgeevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(scale),
            &abnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return ilo, ihi, scale, abnrm, rconde, rcondv, info, info == 0
}

// ===================================================================================
// SCHUR DECOMPOSITION (GEES family)
// ===================================================================================

dns_schur :: proc {
    dns_schur_real,
    dns_schur_complex,
}

dns_schur_expert :: proc {
    dns_schur_expert_real,
    dns_schur_expert_complex,
}

// Query workspace size for Schur decomposition
query_workspace_dns_schur :: proc(
    A: ^Matrix($T),
    jobvs: Schur_Job = .Compute,
) -> (
    work_size: int,
    rwork_size: int,
    bwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows

    when is_float(T) {
        work_size = max(1, int(3 * n))
        bwork_size = int(n)
        rwork_size = 0
    } else when is_complex(T) {
        work_size = max(1, int(2 * n))
        rwork_size = int(n)
        bwork_size = int(n)
    }

    return work_size, rwork_size, bwork_size
}

// Compute Schur form: A = Q * T * Q^T where T is upper quasi-triangular
// sdim: Number of selected eigenvalues
dns_schur_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Schur form)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    VS: ^Matrix(T), // Schur vectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvs: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    select: rawptr = nil, // Function pointer for sorting (optional)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(bwork) >= int(n), "bwork array too small")

    jobvs_c := cast(u8)jobvs
    sort_c := cast(u8)sort

    ldvs: Blas_Int = 1
    vs_ptr: ^T = nil
    if jobvs != .None && VS != nil {
        ldvs = VS.ld
        assert(VS.rows == n && VS.cols == n, "VS matrix dimensions incorrect")
        vs_ptr = raw_data(VS.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgees_(
            &jobvs_c,
            &sort_c,
            select,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(WR),
            raw_data(WI),
            vs_ptr,
            &ldvs,
            raw_data(work),
            &lwork,
            raw_data(bwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgees_(
            &jobvs_c,
            &sort_c,
            select,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(WR),
            raw_data(WI),
            vs_ptr,
            &ldvs,
            raw_data(work),
            &lwork,
            raw_data(bwork),
            &info,
        )
    }

    return sdim, info, info == 0
}

dns_schur_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten with Schur form)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    VS: ^Matrix(Cmplx), // Schur vectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvs: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    select: rawptr = nil, // Function pointer for sorting (optional)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(n), "rwork array too small")
    assert(len(bwork) >= int(n), "bwork array too small")

    jobvs_c := cast(u8)jobvs
    sort_c := cast(u8)sort

    ldvs: Blas_Int = 1
    vs_ptr: ^Cmplx = nil
    if jobvs != .None && VS != nil {
        ldvs = VS.ld
        assert(VS.rows == n && VS.cols == n, "VS matrix dimensions incorrect")
        vs_ptr = raw_data(VS.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgees_(
            &jobvs_c,
            &sort_c,
            select,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(W),
            vs_ptr,
            &ldvs,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(bwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgees_(
            &jobvs_c,
            &sort_c,
            select,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(W),
            vs_ptr,
            &ldvs,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(bwork),
            &info,
        )
    }

    return sdim, info, info == 0
}

// Query workspace size for expert Schur decomposition
query_workspace_dns_schur_expert :: proc(
    A: ^Matrix($T),
    jobvs: Schur_Job = .Compute,
) -> (
    work: int,
    iwork: int,
    rwork: int,
    bwork: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld
    lwork := QUERY_WORKSPACE
    liwork := QUERY_WORKSPACE
    info: Info
    work_query: T
    iwork_query: Blas_Int

    jobvs_c := cast(u8)jobvs
    sort_c := u8('N')
    sense_c := u8('B') // Both condition numbers

    when is_float(T) {
        when T == f32 {
            lapack.sgeesx_(
                &jobvs_c,
                &sort_c,
                nil,
                &sense_c,
                &n,
                nil,
                &lda,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                &work_query,
                &lwork,
                &iwork_query,
                &liwork,
                nil,
                &info,
            )
        } else when T == f64 {
            lapack.dgeesx_(
                &jobvs_c,
                &sort_c,
                nil,
                &sense_c,
                &n,
                nil,
                &lda,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                &work_query,
                &lwork,
                &iwork_query,
                &liwork,
                nil,
                &info,
            )
        }
        return int(work_query), int(iwork_query), 0, int(n)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.cgeesx_(
                &jobvs_c,
                &sort_c,
                nil,
                &sense_c,
                &n,
                nil,
                &lda,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                &work_query,
                &lwork,
                nil,
                nil,
                &info,
            )
        } else when T == complex128 {
            lapack.zgeesx_(
                &jobvs_c,
                &sort_c,
                nil,
                &sense_c,
                &n,
                nil,
                &lda,
                nil,
                nil,
                nil,
                nil,
                nil,
                nil,
                &work_query,
                &lwork,
                nil,
                nil,
                &info,
            )
        }
        return int(real(work_query)), 0, int(n), int(n)
    }
}

// Expert Schur decomposition with condition number estimates
dns_schur_expert_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Schur form)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    VS: ^Matrix(T), // Schur vectors (pre-allocated, optional)
    rconde: ^T, // Reciprocal condition number for eigenvalues (output)
    rcondv: ^T, // Reciprocal condition number for eigenvectors (output)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvs: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    sense: Sense_Job = .Both,
    select: rawptr = nil, // Function pointer for sorting (optional)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) > 0, "iwork array must be provided")
    assert(len(bwork) >= int(n), "bwork array too small")

    jobvs_c := cast(u8)jobvs
    sort_c := cast(u8)sort
    sense_c := cast(u8)sense

    ldvs: Blas_Int = 1
    vs_ptr: ^T = nil
    if jobvs != .None && VS != nil {
        ldvs = VS.ld
        assert(VS.rows == n && VS.cols == n, "VS matrix dimensions incorrect")
        vs_ptr = raw_data(VS.data)
    }

    lwork := Blas_Int(len(work))
    liwork := Blas_Int(len(iwork))

    when T == f32 {
        lapack.sgeesx_(
            &jobvs_c,
            &sort_c,
            select,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(WR),
            raw_data(WI),
            vs_ptr,
            &ldvs,
            rconde,
            rcondv,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            raw_data(bwork),
            &info,
        )
    } else when T == f64 {
        lapack.dgeesx_(
            &jobvs_c,
            &sort_c,
            select,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(WR),
            raw_data(WI),
            vs_ptr,
            &ldvs,
            rconde,
            rcondv,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            raw_data(bwork),
            &info,
        )
    }

    return sdim, info, info == 0
}

dns_schur_expert_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten with Schur form)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    VS: ^Matrix(Cmplx), // Schur vectors (pre-allocated, optional)
    rconde: ^$Real, // Reciprocal condition number for eigenvalues (output)
    rcondv: ^Real, // Reciprocal condition number for eigenvectors (output)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []Real, // Real workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvs: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    sense: Sense_Job = .Both,
    select: rawptr = nil, // Function pointer for sorting (optional)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(n), "rwork array too small")
    assert(len(bwork) >= int(n), "bwork array too small")

    jobvs_c := cast(u8)jobvs
    sort_c := cast(u8)sort
    sense_c := cast(u8)sense

    ldvs: Blas_Int = 1
    vs_ptr: ^Cmplx = nil
    if jobvs != .None && VS != nil {
        ldvs = VS.ld
        assert(VS.rows == n && VS.cols == n, "VS matrix dimensions incorrect")
        vs_ptr = raw_data(VS.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgeesx_(
            &jobvs_c,
            &sort_c,
            select,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(W),
            vs_ptr,
            &ldvs,
            rconde,
            rcondv,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(bwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgeesx_(
            &jobvs_c,
            &sort_c,
            select,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            &sdim,
            raw_data(W),
            vs_ptr,
            &ldvs,
            rconde,
            rcondv,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(bwork),
            &info,
        )
    }

    return sdim, info, info == 0
}

// ===================================================================================
// MATRIX BALANCING (GEBAL/GEBAK)
// ===================================================================================

// Balance a general matrix to improve eigenvalue computation
dns_balance :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with balanced matrix)
    scale: []T, // Scaling factors (pre-allocated, size n)
    job: Balance_Job = .Both,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(scale) >= int(n), "scale array too small")

    job_c := cast(u8)job

    when T == f32 {
        lapack.sgebal_(&job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info)
    } else when T == f64 {
        lapack.dgebal_(&job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info)
    } else when T == complex64 {
        lapack.cgebal_(&job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info)
    } else when T == complex128 {
        lapack.zgebal_(&job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info)
    }

    return ilo, ihi, info, info == 0
}

// Back-transform eigenvectors of a balanced matrix
dns_back_transform :: proc(
    V: ^Matrix($T), // Eigenvectors (overwritten with back-transformed vectors)
    scale: []T, // From balance
    ilo: Blas_Int, // From balance
    ihi: Blas_Int, // From balance
    job: Balance_Job, // Must match job used in balance
    side: Side, // Left or right eigenvectors
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    n := V.rows
    m := V.cols
    ldv := V.ld

    assert(len(scale) >= int(n), "scale array too small")

    job_c := cast(u8)job
    side_c := cast(u8)side

    when T == f32 {
        lapack.sgebak_(&job_c, &side_c, &n, &ilo, &ihi, raw_data(scale), &m, raw_data(V.data), &ldv, &info)
    } else when T == f64 {
        lapack.dgebak_(&job_c, &side_c, &n, &ilo, &ihi, raw_data(scale), &m, raw_data(V.data), &ldv, &info)
    } else when T == complex64 {
        lapack.cgebak_(&job_c, &side_c, &n, &ilo, &ihi, raw_data(scale), &m, raw_data(V.data), &ldv, &info)
    } else when T == complex128 {
        lapack.zgebak_(&job_c, &side_c, &n, &ilo, &ihi, raw_data(scale), &m, raw_data(V.data), &ldv, &info)
    }

    return info, info == 0
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS
// ===================================================================================

// Selection function types for sorting generalized eigenvalues
LAPACK_S_SELECT3 :: proc "c" (alphar: ^f32, alphai: ^f32, beta: ^f32) -> Blas_Int
LAPACK_D_SELECT3 :: proc "c" (alphar: ^f64, alphai: ^f64, beta: ^f64) -> Blas_Int
LAPACK_C_SELECT2 :: proc "c" (alpha: ^complex64, beta: ^complex64) -> Blas_Int
LAPACK_Z_SELECT2 :: proc "c" (alpha: ^complex128, beta: ^complex128) -> Blas_Int

// ===================================================================================
// Generalized Schur Decomposition (GGES family)
// ===================================================================================

dns_schur_generalized :: proc {
    dns_schur_generalized_real,
    dns_schur_generalized_complex,
}

// Query workspace size for generalized Schur decomposition
query_workspace_dns_schur_generalized :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
) -> (
    work_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort
    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        sdim: Blas_Int
        lapack.sgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            &info,
        )
        work_size = int(work_query)
        rwork_size = 0
    } else when T == f64 {
        work_query: f64
        info: Info
        sdim: Blas_Int
        lapack.dgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            &info,
        )
        work_size = int(work_query)
        rwork_size = 0
    } else when T == complex64 {
        work_query: complex64
        info: Info
        sdim: Blas_Int
        lapack.cgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        rwork_size = int(8 * n)
    } else when T == complex128 {
        work_query: complex128
        info: Info
        sdim: Blas_Int
        lapack.zgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        rwork_size = int(8 * n)
    }

    return work_size, rwork_size
}

// Compute generalized Schur decomposition (real matrices)
// A and B are overwritten with Schur forms S and T
// VSL and VSR contain the left and right Schur vectors if requested
// For real matrices, eigenvalues are (alphar + i*alphai)/beta
// sdim: Number of eigenvalues selected (if sorting)
dns_schur_generalized_real :: proc(
    A: ^Matrix($T), // Input matrix A (overwritten with S)
    B: ^Matrix(T), // Input matrix B (overwritten with T)
    alphar: []T, // Real parts of alpha (pre-allocated, size n)
    alphai: []T, // Imaginary parts of alpha (pre-allocated, size n)
    beta: []T, // Beta values (pre-allocated, size n)
    VSL: ^Matrix(T), // Left Schur vectors (pre-allocated, optional)
    VSR: ^Matrix(T), // Right Schur vectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean work array for sorting (pre-allocated if sorting, size n)
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    select_fn: rawptr = nil, // Selection function for sorting (LAPACK_S_SELECT3 or LAPACK_D_SELECT3)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alphar) >= int(n), "alphar array too small")
    assert(len(alphai) >= int(n), "alphai array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    if sort == .Sort {
        assert(len(bwork) >= int(n), "bwork array too small for sorting")
        assert(select_fn != nil, "select_fn must be provided when sorting")
    }

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort

    ldvsl: Blas_Int = 1
    vsl_ptr: ^T = nil
    if jobvsl != .None && VSL != nil {
        ldvsl = VSL.ld
        assert(VSL.rows == n && VSL.cols == n, "VSL matrix dimensions incorrect")
        vsl_ptr = raw_data(VSL.data)
    }

    ldvsr: Blas_Int = 1
    vsr_ptr: ^T = nil
    if jobvsr != .None && VSR != nil {
        ldvsr = VSR.ld
        assert(VSR.rows == n && VSR.cols == n, "VSR matrix dimensions incorrect")
        vsr_ptr = raw_data(VSR.data)
    }

    bwork_ptr: ^Blas_Int = nil
    if sort == .Sort {
        bwork_ptr = raw_data(bwork)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_S_SELECT3)select_fn,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            raw_data(work),
            &lwork,
            bwork_ptr,
            &info,
        )
    } else when T == f64 {
        lapack.dgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_D_SELECT3)select_fn,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            raw_data(work),
            &lwork,
            bwork_ptr,
            &info,
        )
    }

    return sdim, info, info == 0
}

// Compute generalized Schur decomposition (complex matrices)
// A and B are overwritten with Schur forms S and T
// VSL and VSR contain the left and right Schur vectors if requested
// For complex matrices, eigenvalues are alpha/beta
dns_schur_generalized_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix A (overwritten with S)
    B: ^Matrix(Cmplx), // Input matrix B (overwritten with T)
    alpha: []Cmplx, // Alpha values (pre-allocated, size n)
    beta: []Cmplx, // Beta values (pre-allocated, size n)
    VSL: ^Matrix(Cmplx), // Left Schur vectors (pre-allocated, optional)
    VSR: ^Matrix(Cmplx), // Right Schur vectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated, size 8*n)
    bwork: []Blas_Int, // Boolean work array for sorting (pre-allocated if sorting, size n)
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    select_fn: rawptr = nil, // Selection function for sorting (LAPACK_C_SELECT2 or LAPACK_Z_SELECT2)
) -> (
    sdim: Blas_Int,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alpha) >= int(n), "alpha array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(8 * n), "rwork array too small")
    if sort == .Sort {
        assert(len(bwork) >= int(n), "bwork array too small for sorting")
        assert(select_fn != nil, "select_fn must be provided when sorting")
    }

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort

    ldvsl: Blas_Int = 1
    vsl_ptr: ^Cmplx = nil
    if jobvsl != .None && VSL != nil {
        ldvsl = VSL.ld
        assert(VSL.rows == n && VSL.cols == n, "VSL matrix dimensions incorrect")
        vsl_ptr = raw_data(VSL.data)
    }

    ldvsr: Blas_Int = 1
    vsr_ptr: ^Cmplx = nil
    if jobvsr != .None && VSR != nil {
        ldvsr = VSR.ld
        assert(VSR.rows == n && VSR.cols == n, "VSR matrix dimensions incorrect")
        vsr_ptr = raw_data(VSR.data)
    }

    bwork_ptr: ^Blas_Int = nil
    if sort == .Sort {
        bwork_ptr = raw_data(bwork)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_C_SELECT2)select_fn,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alpha),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            bwork_ptr,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgges3_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_Z_SELECT2)select_fn,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alpha),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            bwork_ptr,
            &info,
        )
    }

    return sdim, info, info == 0
}

// ===================================================================================
// Generalized Schur Decomposition Expert Driver (GGESX family)
// ===================================================================================

dns_schur_generalized_expert :: proc {
    dns_schur_generalized_expert_real,
    dns_schur_generalized_expert_complex,
}

// Query workspace size for generalized Schur decomposition expert driver
query_workspace_generalized_schur_expert :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    sense: Sense_Job = .None,
) -> (
    work_size: int,
    iwork_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort
    sense_c := cast(u8)sense
    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        sdim: Blas_Int
        liwork: Blas_Int = 1
        lapack.sggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            &liwork,
            nil,
            &info,
        )
        work_size = int(work_query)
        iwork_size = int(liwork)
        rwork_size = 0
    } else when T == f64 {
        work_query: f64
        info: Info
        sdim: Blas_Int
        liwork: Blas_Int = 1
        lapack.dggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            &liwork,
            nil,
            &info,
        )
        work_size = int(work_query)
        iwork_size = int(liwork)
        rwork_size = 0
    } else when T == complex64 {
        work_query: complex64
        info: Info
        sdim: Blas_Int
        liwork: Blas_Int = 1
        lapack.cggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            &liwork,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        iwork_size = int(liwork)
        rwork_size = int(n)
    } else when T == complex128 {
        work_query: complex128
        info: Info
        sdim: Blas_Int
        liwork: Blas_Int = 1
        lapack.zggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            nil,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            &sdim,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            &liwork,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        iwork_size = int(liwork)
        rwork_size = int(n)
    }

    return work_size, iwork_size, rwork_size
}

// Compute generalized Schur decomposition with expert driver (real matrices)
// Provides condition number estimates for eigenvalues and subspaces
dns_schur_generalized_expert_real :: proc(
    A: ^Matrix($T), // Input matrix A (overwritten with S)
    B: ^Matrix(T), // Input matrix B (overwritten with T)
    alphar: []T, // Real parts of alpha (pre-allocated, size n)
    alphai: []T, // Imaginary parts of alpha (pre-allocated, size n)
    beta: []T, // Beta values (pre-allocated, size n)
    VSL: ^Matrix(T), // Left Schur vectors (pre-allocated, optional)
    VSR: ^Matrix(T), // Right Schur vectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean work array for sorting (pre-allocated if sorting, size n)
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    sense: Sense_Job = .None,
    select_fn: rawptr = nil, // Selection function for sorting
) -> (
    sdim: Blas_Int,
    rconde: [2]T,
    rcondv: [2]T,
    info: Info,
    ok: bool, // Number of eigenvalues selected (if sorting)// Reciprocal condition numbers for eigenvalues// Reciprocal condition numbers for eigenvectors
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alphar) >= int(n), "alphar array too small")
    assert(len(alphai) >= int(n), "alphai array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) > 0, "iwork array must be provided")
    if sort == .Sort {
        assert(len(bwork) >= int(n), "bwork array too small for sorting")
        assert(select_fn != nil, "select_fn must be provided when sorting")
    }

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort
    sense_c := cast(u8)sense

    ldvsl: Blas_Int = 1
    vsl_ptr: ^T = nil
    if jobvsl != .None && VSL != nil {
        ldvsl = VSL.ld
        assert(VSL.rows == n && VSL.cols == n, "VSL matrix dimensions incorrect")
        vsl_ptr = raw_data(VSL.data)
    }

    ldvsr: Blas_Int = 1
    vsr_ptr: ^T = nil
    if jobvsr != .None && VSR != nil {
        ldvsr = VSR.ld
        assert(VSR.rows == n && VSR.cols == n, "VSR matrix dimensions incorrect")
        vsr_ptr = raw_data(VSR.data)
    }

    bwork_ptr: ^Blas_Int = nil
    if sort == .Sort {
        bwork_ptr = raw_data(bwork)
    }

    lwork := Blas_Int(len(work))
    liwork := Blas_Int(len(iwork))

    when T == f32 {
        lapack.sggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_S_SELECT3)select_fn,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            &rconde[0],
            &rcondv[0],
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            bwork_ptr,
            &info,
        )
    } else when T == f64 {
        lapack.dggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_D_SELECT3)select_fn,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            &rconde[0],
            &rcondv[0],
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            bwork_ptr,
            &info,
        )
    }

    return sdim, rconde, rcondv, info, info == 0
}

// Compute generalized Schur decomposition with expert driver (complex matrices)
dns_schur_generalized_expert_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix A (overwritten with S)
    B: ^Matrix(Cmplx), // Input matrix B (overwritten with T)
    alpha: []Cmplx, // Alpha values (pre-allocated, size n)
    beta: []Cmplx, // Beta values (pre-allocated, size n)
    VSL: ^Matrix(Cmplx), // Left Schur vectors (pre-allocated, optional)
    VSR: ^Matrix(Cmplx), // Right Schur vectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated, size n)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    bwork: []Blas_Int, // Boolean work array for sorting (pre-allocated if sorting, size n)
    jobvsl: Schur_Job = .Compute,
    jobvsr: Schur_Job = .Compute,
    sort: Schur_Sort = .None,
    sense: Sense_Job = .None,
    select_fn: rawptr = nil, // Selection function for sorting
) -> (
    sdim: Blas_Int,
    rconde: [2]Real,
    rcondv: [2]Real,
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alpha) >= int(n), "alpha array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(n), "rwork array too small")
    assert(len(iwork) > 0, "iwork array must be provided")
    if sort == .Sort {
        assert(len(bwork) >= int(n), "bwork array too small for sorting")
        assert(select_fn != nil, "select_fn must be provided when sorting")
    }

    jobvsl_c := cast(u8)jobvsl
    jobvsr_c := cast(u8)jobvsr
    sort_c := cast(u8)sort
    sense_c := cast(u8)sense

    ldvsl: Blas_Int = 1
    vsl_ptr: ^Cmplx = nil
    if jobvsl != .None && VSL != nil {
        ldvsl = VSL.ld
        assert(VSL.rows == n && VSL.cols == n, "VSL matrix dimensions incorrect")
        vsl_ptr = raw_data(VSL.data)
    }

    ldvsr: Blas_Int = 1
    vsr_ptr: ^Cmplx = nil
    if jobvsr != .None && VSR != nil {
        ldvsr = VSR.ld
        assert(VSR.rows == n && VSR.cols == n, "VSR matrix dimensions incorrect")
        vsr_ptr = raw_data(VSR.data)
    }

    bwork_ptr: ^Blas_Int = nil
    if sort == .Sort {
        bwork_ptr = raw_data(bwork)
    }

    lwork := Blas_Int(len(work))
    liwork := Blas_Int(len(iwork))

    when Cmplx == complex64 {
        lapack.cggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_C_SELECT2)select_fn,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alpha),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            &rconde[0],
            &rcondv[0],
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            &liwork,
            bwork_ptr,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zggesx_(
            &jobvsl_c,
            &jobvsr_c,
            &sort_c,
            cast(LAPACK_Z_SELECT2)select_fn,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &sdim,
            raw_data(alpha),
            raw_data(beta),
            vsl_ptr,
            &ldvsl,
            vsr_ptr,
            &ldvsr,
            &rconde[0],
            &rcondv[0],
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            &liwork,
            bwork_ptr,
            &info,
        )
    }

    return sdim, rconde, rcondv, info, info == 0
}

// ===================================================================================
// Generalized Eigenvalue Problem (GGEV family)
// ===================================================================================

dns_eigen_generalized :: proc {
    dns_eigen_generalized_real,
    dns_eigen_generalized_complex,
}

// Query workspace size for generalized eigenvalue problem
query_workspace_dns_eigen_generalized :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    work_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            &info,
        )
        work_size = int(work_query)
        rwork_size = 0
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            &info,
        )
        work_size = int(work_query)
        rwork_size = 0
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        rwork_size = int(8 * n)
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            &work_query,
            &lwork,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        rwork_size = int(8 * n)
    }

    return work_size, rwork_size
}

// Compute generalized eigenvalues and eigenvectors (real matrices)
// Solves the generalized eigenvalue problem: A*x = lambda*B*x
// For real matrices, eigenvalues are (alphar + i*alphai)/beta
dns_eigen_generalized_real :: proc(
    A: ^Matrix($T), // Input matrix A (overwritten)
    B: ^Matrix(T), // Input matrix B (overwritten)
    alphar: []T, // Real parts of alpha (pre-allocated, size n)
    alphai: []T, // Imaginary parts of alpha (pre-allocated, size n)
    beta: []T, // Beta values (pre-allocated, size n)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    work: []T, // Workspace (pre-allocated)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alphar) >= int(n), "alphar array too small")
    assert(len(alphai) >= int(n), "alphai array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// Compute generalized eigenvalues and eigenvectors (complex matrices)
// Solves the generalized eigenvalue problem: A*x = lambda*B*x
// For complex matrices, eigenvalues are alpha/beta
dns_eigen_generalized_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix A (overwritten)
    B: ^Matrix(Cmplx), // Input matrix B (overwritten)
    alpha: []Cmplx, // Alpha values (pre-allocated, size n)
    beta: []Cmplx, // Beta values (pre-allocated, size n)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated, size 8*n)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alpha) >= int(n), "alpha array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(8 * n), "rwork array too small")

    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alpha),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zggev3_(
            &jobvl_c,
            &jobvr_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alpha),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// Generalized Eigenvalue Problem Expert Driver (GGEVX family)
// ===================================================================================

dns_eigen_generalized_expert :: proc {
    dns_eigen_generalized_expert_real,
    dns_eigen_generalized_expert_complex,
}

// Query workspace size for generalized eigenvalue expert driver
query_workspace_generalized_eigenvalues_expert :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
    balanc: Balance_Job = .Both,
    sense: Sense_Job = .None,
) -> (
    work_size: int,
    iwork_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    balanc_c := cast(u8)balanc
    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    sense_c := cast(u8)sense
    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            &info,
        )
        work_size = int(work_query)
        iwork_size = int(n + 6)
        rwork_size = 0
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            &info,
        )
        work_size = int(work_query)
        iwork_size = int(n + 6)
        rwork_size = 0
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        iwork_size = int(n + 6)
        rwork_size = int(6 * n)
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            nil,
            &lda,
            nil,
            &ldb,
            nil,
            nil,
            nil,
            &n,
            nil,
            &n,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            &work_query,
            &lwork,
            nil,
            nil,
            nil,
            &info,
        )
        work_size = int(real(work_query))
        iwork_size = int(n + 6)
        rwork_size = int(6 * n)
    }

    return work_size, iwork_size, rwork_size
}

// Compute generalized eigenvalues with expert driver (real matrices)
// Provides balancing, condition number estimates
dns_eigen_generalized_expert_real :: proc(
    A: ^Matrix($T), // Input matrix A (overwritten)
    B: ^Matrix(T), // Input matrix B (overwritten)
    alphar: []T, // Real parts of alpha (pre-allocated, size n)
    alphai: []T, // Imaginary parts of alpha (pre-allocated, size n)
    beta: []T, // Beta values (pre-allocated, size n)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    lscale: []T, // Left scale factors (pre-allocated, size n)
    rscale: []T, // Right scale factors (pre-allocated, size n)
    rconde: []T, // Condition numbers for eigenvalues (pre-allocated, size n)
    rcondv: []T, // Condition numbers for eigenvectors (pre-allocated, size n)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated, size n+6)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
    balanc: Balance_Job = .Both,
    sense: Sense_Job = .None,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    abnrm: T,
    bbnrm: T,
    info: Info,
    ok: bool, // Balancing info// Balancing info// Norm of balanced A// Norm of balanced B
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alphar) >= int(n), "alphar array too small")
    assert(len(alphai) >= int(n), "alphai array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(lscale) >= int(n), "lscale array too small")
    assert(len(rscale) >= int(n), "rscale array too small")
    assert(len(rconde) >= int(n), "rconde array too small")
    assert(len(rcondv) >= int(n), "rcondv array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) >= int(n + 6), "iwork array too small")
    assert(len(bwork) >= int(n), "bwork array too small")

    balanc_c := cast(u8)balanc
    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    sense_c := cast(u8)sense

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &abnrm,
            &bbnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(iwork),
            raw_data(bwork),
            &info,
        )
    } else when T == f64 {
        lapack.dggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &abnrm,
            &bbnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(iwork),
            raw_data(bwork),
            &info,
        )
    }

    return ilo, ihi, abnrm, bbnrm, info, info == 0
}

// Compute generalized eigenvalues with expert driver (complex matrices)
dns_eigen_generalized_expert_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix A (overwritten)
    B: ^Matrix(Cmplx), // Input matrix B (overwritten)
    alpha: []Cmplx, // Alpha values (pre-allocated, size n)
    beta: []Cmplx, // Beta values (pre-allocated, size n)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    lscale: []$Real, // Left scale factors (pre-allocated, size n)
    rscale: []Real, // Right scale factors (pre-allocated, size n)
    rconde: []Real, // Condition numbers for eigenvalues (pre-allocated, size n)
    rcondv: []Real, // Condition numbers for eigenvectors (pre-allocated, size n)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []Real, // Real workspace (pre-allocated, size 6*n)
    iwork: []Blas_Int, // Integer workspace (pre-allocated, size n+6)
    bwork: []Blas_Int, // Boolean workspace (pre-allocated, size n)
    jobvl: Eigen_Job_Left = .None,
    jobvr: Eigen_Job_Right = .Compute,
    balanc: Balance_Job = .Both,
    sense: Sense_Job = .None,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    abnrm: Real,
    bbnrm: Real,
    info: Info,
    ok: bool, // Balancing info// Balancing info// Norm of balanced A// Norm of balanced B
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(alpha) >= int(n), "alpha array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(lscale) >= int(n), "lscale array too small")
    assert(len(rscale) >= int(n), "rscale array too small")
    assert(len(rconde) >= int(n), "rconde array too small")
    assert(len(rcondv) >= int(n), "rcondv array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(6 * n), "rwork array too small")
    assert(len(iwork) >= int(n + 6), "iwork array too small")
    assert(len(bwork) >= int(n), "bwork array too small")

    balanc_c := cast(u8)balanc
    jobvl_c := cast(u8)jobvl
    jobvr_c := cast(u8)jobvr
    sense_c := cast(u8)sense

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if jobvl != .None && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols == n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if jobvr != .None && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols == n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alpha),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &abnrm,
            &bbnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(bwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zggevx_(
            &balanc_c,
            &jobvl_c,
            &jobvr_c,
            &sense_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(alpha),
            raw_data(beta),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &abnrm,
            &bbnrm,
            raw_data(rconde),
            raw_data(rcondv),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            raw_data(iwork),
            raw_data(bwork),
            &info,
        )
    }

    return ilo, ihi, abnrm, bbnrm, info, info == 0
}

// ===================================================================================
// Triangular Eigenvector Computation (TREVC family)
// ===================================================================================

// Side options for eigenvector computation
Eigenvector_Side :: enum u8 {
    Left  = 'L', // Compute left eigenvectors
    Right = 'R', // Compute right eigenvectors
    Both  = 'B', // Compute both left and right eigenvectors
}

// Selection options for eigenvectors
Eigenvector_Selection :: enum u8 {
    All           = 'A', // Compute all eigenvectors
    Backtransform = 'B', // Backtransform selected eigenvectors
    Selected      = 'S', // Compute selected eigenvectors
}

triangular_eigenvectors :: proc {
    triangular_eigenvectors_real,
    triangular_eigenvectors_complex,
}

// Query workspace size for triangular eigenvector computation
query_workspace_triangular_eigenvectors :: proc(
    schur: ^Matrix($T),
    side: Eigenvector_Side = .Right,
) -> (
    work_size: int,
    rwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := schur.rows

    when is_float(T) {
        work_size = int(3 * n)
        rwork_size = 0
    } else when is_complex(T) {
        // Query with lwork = -1
        lwork: Blas_Int = QUERY_WORKSPACE
        work_query: T
        side_c := cast(u8)side
        howmny_c := cast(u8)Eigenvector_Selection.All
        mm: Blas_Int = n
        m: Blas_Int
        info: Info

        when T == complex64 {
            lrwork: Blas_Int = QUERY_WORKSPACE
            rwork_query: f32
            lapack.ctrevc3_(
                &side_c,
                &howmny_c,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &mm,
                &m,
                &work_query,
                &lwork,
                &rwork_query,
                &lrwork,
                &info,
            )
            work_size = int(real(work_query))
            rwork_size = int(rwork_query)
        } else when T == complex128 {
            lrwork: Blas_Int = QUERY_WORKSPACE
            rwork_query: f64
            lapack.ztrevc3_(
                &side_c,
                &howmny_c,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &mm,
                &m,
                &work_query,
                &lwork,
                &rwork_query,
                &lrwork,
                &info,
            )
            work_size = int(real(work_query))
            rwork_size = int(rwork_query)
        }
    }

    return work_size, rwork_size
}

// Compute eigenvectors of a triangular matrix (real)
// For real matrices, complex conjugate pairs are returned in consecutive columns
triangular_eigenvectors_real :: proc(
    schur: ^Matrix($T), // Input triangular matrix (not modified)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (pre-allocated if howmny == .Selected, size n)
    work: []T, // Workspace (pre-allocated)
    side: Eigenvector_Side = .Right,
    howmny: Eigenvector_Selection = .All,
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where is_float(T) {
    n := T.rows
    ldt := T.ld

    assert(T.rows == T.cols, "Matrix T must be square")
    assert(len(work) > 0, "work array must be provided")
    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    side_c := cast(u8)side
    howmny_c := cast(u8)howmny
    mm: Blas_Int = n

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    select_ptr: ^Blas_Int = nil
    if howmny == .Selected {
        select_ptr = raw_data(select)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.strevc3_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(T.data),
            &ldt,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dtrevc3_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(T.data),
            &ldt,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return m, info, info == 0
}

// Compute eigenvectors of a triangular matrix (complex)
triangular_eigenvectors_complex :: proc(
    T: ^Matrix($Cmplx), // Input triangular matrix (not modified)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (pre-allocated if howmny == .Selected, size n)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated)
    side: Eigenvector_Side = .Right,
    howmny: Eigenvector_Selection = .All,
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := T.rows
    ldt := T.ld

    assert(T.rows == T.cols, "Matrix T must be square")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) > 0, "rwork array must be provided")
    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    side_c := cast(u8)side
    howmny_c := cast(u8)howmny
    mm: Blas_Int = n

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    select_ptr: ^Blas_Int = nil
    if howmny == .Selected {
        select_ptr = raw_data(select)
    }

    lwork := Blas_Int(len(work))
    lrwork := Blas_Int(len(rwork))

    when Cmplx == complex64 {
        lapack.ctrevc3_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(T.data),
            &ldt,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztrevc3_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(T.data),
            &ldt,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &lrwork,
            &info,
        )
    }

    return m, info, info == 0
}

// ===================================================================================
// Generalized Triangular Eigenvector Computation (TGEVC family)
// ===================================================================================

generalized_triangular_eigenvectors :: proc {
    generalized_triangular_eigenvectors_real,
    generalized_triangular_eigenvectors_complex,
}

// Query workspace size for generalized triangular eigenvector computation
query_workspace_generalized_triangular_eigenvectors :: proc(
    S: ^Matrix($T),
) -> (
    work: int,
    rwork: int,
) where is_float(T) ||
    is_complex(T) {
    n := int(S.rows)

    when is_float(T) {
        work = 6 * n
        rwork = 0
    } else when is_complex(T) {
        work = 2 * n
        rwork = 2 * n
    }

    return
}

// Compute eigenvectors of a generalized eigenvalue problem from generalized Schur form (real)
// Given the generalized Schur form (S, P), compute left and/or right eigenvectors
generalized_triangular_eigenvectors_real :: proc(
    S: ^Matrix($T), // First matrix in generalized Schur form (not modified)
    P: ^Matrix(T), // Second matrix in generalized Schur form (not modified)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (pre-allocated if howmny == .Selected, size n)
    work: []T, // Workspace (pre-allocated)
    side: Eigenvector_Side = .Right,
    howmny: Eigenvector_Selection = .All,
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where is_float(T) {
    n := S.rows
    lds := S.ld
    ldp := P.ld

    assert(S.rows == S.cols, "Matrix S must be square")
    assert(P.rows == P.cols, "Matrix P must be square")
    assert(S.rows == P.rows, "Matrices S and P must have same dimensions")
    assert(len(work) > 0, "work array must be provided")
    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    side_c := cast(u8)side
    howmny_c := cast(u8)howmny
    mm: Blas_Int = n

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    select_ptr: ^Blas_Int = nil
    if howmny == .Selected {
        select_ptr = raw_data(select)
    }

    when T == f32 {
        lapack.stgevc_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(S.data),
            &lds,
            raw_data(P.data),
            &ldp,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &info,
        )
    } else when T == f64 {
        lapack.dtgevc_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(S.data),
            &lds,
            raw_data(P.data),
            &ldp,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            &info,
        )
    }

    return m, info, info == 0
}

// Compute eigenvectors of a generalized eigenvalue problem from generalized Schur form (complex)
// Given the generalized Schur form (S, P), compute left and/or right eigenvectors
generalized_triangular_eigenvectors_complex :: proc(
    S: ^Matrix($Cmplx), // First matrix in generalized Schur form (not modified)
    P: ^Matrix(Cmplx), // Second matrix in generalized Schur form (not modified)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (pre-allocated if howmny == .Selected, size n)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated)
    side: Eigenvector_Side = .Right,
    howmny: Eigenvector_Selection = .All,
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := S.rows
    lds := S.ld
    ldp := P.ld

    assert(S.rows == S.cols, "Matrix S must be square")
    assert(P.rows == P.cols, "Matrix P must be square")
    assert(S.rows == P.rows, "Matrices S and P must have same dimensions")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) > 0, "rwork array must be provided")
    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    side_c := cast(u8)side
    howmny_c := cast(u8)howmny
    mm: Blas_Int = n

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= n, "VL matrix dimensions incorrect")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= n, "VR matrix dimensions incorrect")
        vr_ptr = raw_data(VR.data)
    }

    select_ptr: ^Blas_Int = nil
    if howmny == .Selected {
        select_ptr = raw_data(select)
    }

    when Cmplx == complex64 {
        lapack.ctgevc_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(S.data),
            &lds,
            raw_data(P.data),
            &ldp,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztgevc_(
            &side_c,
            &howmny_c,
            select_ptr,
            &n,
            raw_data(S.data),
            &lds,
            raw_data(P.data),
            &ldp,
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm,
            &m,
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    }

    return m, info, info == 0
}

// ===================================================================================
// Schur Form Reordering (TREXC)
// ===================================================================================

// Computation mode for Schur vectors
Schur_Computation :: enum u8 {
    None   = 'N', // Do not update Schur vectors
    Update = 'V', // Update Schur vectors
}

reorder_schur :: proc {
    reorder_schur_real,
    reorder_schur_complex,
}

// Reorder eigenvalues in Schur form (real)
// Exchanges two adjacent blocks in the Schur factorization
reorder_schur_real :: proc(
    schur: ^Matrix($T), // Schur form matrix (overwritten with reordered form)
    Q: ^Matrix(T), // Schur vectors (updated if compq == .Update)
    ifst: ^Blas_Int, // Position of first block (updated on exit)
    ilst: ^Blas_Int, // Desired position (updated on exit)
    work: []T, // Workspace (pre-allocated, size n)
    compq: Schur_Computation = .Update,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := schur.rows
    ldt := schur.ld

    assert(schur.rows == schur.cols, "Matrix T must be square")
    assert(len(work) >= int(n), "work array too small")
    assert(ifst^ >= 1 && ifst^ <= n, "ifst out of range")
    assert(ilst^ >= 1 && ilst^ <= n, "ilst out of range")

    compq_c := cast(u8)compq

    ldq: Blas_Int = 1
    q_ptr: ^T = nil
    if compq == .Update && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    when T == f32 {
        lapack.strexc_(&compq_c, &n, raw_data(schur.data), &ldt, q_ptr, &ldq, ifst, ilst, raw_data(work), &info)
    } else when T == f64 {
        lapack.dtrexc_(&compq_c, &n, raw_data(schur.data), &ldt, q_ptr, &ldq, ifst, ilst, raw_data(work), &info)
    }

    return info, info == 0
}

// Reorder eigenvalues in Schur form (complex)
// For complex matrices, no workspace is needed
reorder_schur_complex :: proc(
    T: ^Matrix($Cmplx), // Schur form matrix (overwritten with reordered form)
    Q: ^Matrix(Cmplx), // Schur vectors (updated if compq == .Update)
    ifst: Blas_Int, // Position of eigenvalue to move
    ilst: Blas_Int, // Desired position
    compq: Schur_Computation = .Update,
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) {
    n := T.rows
    ldt := T.ld

    assert(T.rows == T.cols, "Matrix T must be square")
    assert(ifst >= 1 && ifst <= n, "ifst out of range")
    assert(ilst >= 1 && ilst <= n, "ilst out of range")

    compq_c := cast(u8)compq

    ldq: Blas_Int = 1
    q_ptr: ^Cmplx = nil
    if compq == .Update && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ifst_copy := ifst
    ilst_copy := ilst

    when Cmplx == complex64 {
        lapack.ctrexc_(&compq_c, &n, raw_data(T.data), &ldt, q_ptr, &ldq, &ifst_copy, &ilst_copy, &info)
    } else when Cmplx == complex128 {
        lapack.ztrexc_(&compq_c, &n, raw_data(T.data), &ldt, q_ptr, &ldq, &ifst_copy, &ilst_copy, &info)
    }

    return info, info == 0
}

// ===================================================================================
// SCHUR FORM CONDITION NUMBERS AND REORDERING WITH CONDITIONING
// ===================================================================================

// Job options for computing condition numbers
Condition_Job :: enum u8 {
    None        = 'N', // No condition numbers computed
    Eigenvalues = 'E', // Condition numbers for eigenvalues only
    Subspace    = 'V', // Condition numbers for invariant subspace only
    Both        = 'B', // Both eigenvalue and subspace condition numbers
}

reorder_schur_with_condition :: proc {
    reorder_schur_with_condition_real,
    reorder_schur_with_condition_complex,
}

// Query workspace sizes for reordering Schur form with condition numbers
query_workspace_reorder_schur_condition :: proc(
    schur: ^Matrix($T),
    job: Condition_Job = .Both,
) -> (
    work_size: int,
    iwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := schur.rows
    ldt := schur.ld

    job_c := cast(u8)job
    compq_c := cast(u8)Schur_Computation.Update
    lwork: Blas_Int = QUERY_WORKSPACE
    liwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    m: Blas_Int
    work_query: T
    iwork_query: Blas_Int

    when is_float(T) {
        when T == f32 {
            lapack.strsen_(
                &job_c,
                &compq_c,
                nil,
                &n,
                nil,
                &ldt,
                nil,
                &ldt,
                nil,
                nil,
                &m,
                nil,
                nil,
                &work_query,
                &lwork,
                &iwork_query,
                &liwork,
                &info,
            )
        } else when T == f64 {
            lapack.dtrsen_(
                &job_c,
                &compq_c,
                nil,
                &n,
                nil,
                &ldt,
                nil,
                &ldt,
                nil,
                nil,
                &m,
                nil,
                nil,
                &work_query,
                &lwork,
                &iwork_query,
                &liwork,
                &info,
            )
        }
        work_size = int(work_query)
        iwork_size = int(iwork_query)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.ctrsen_(
                &job_c,
                &compq_c,
                nil,
                &n,
                nil,
                &ldt,
                nil,
                &ldt,
                nil,
                &m,
                nil,
                nil,
                &work_query,
                &lwork,
                &info,
            )
        } else when T == complex128 {
            lapack.ztrsen_(
                &job_c,
                &compq_c,
                nil,
                &n,
                nil,
                &ldt,
                nil,
                &ldt,
                nil,
                &m,
                nil,
                nil,
                &work_query,
                &lwork,
                &info,
            )
        }
        work_size = int(real(work_query))
        iwork_size = 0
    }

    return work_size, iwork_size
}

// Reorder Schur form with condition number estimates (real)
// Reorders the real Schur factorization and computes condition numbers
reorder_schur_with_condition_real :: proc(
    schur: ^Matrix($T), // Schur form matrix (overwritten with reordered form)
    Q: ^Matrix(T), // Schur vectors (updated if compq == .Update)
    select: []Blas_Int, // Logical selection array (size n, 1 = select eigenvalue)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated for real types)
    job: Condition_Job = .Both,
    compq: Schur_Computation = .Update,
) -> (
    m: Blas_Int,
    s: T,
    sep: T,
    info: Info,
    ok: bool, // Number of selected eigenvalues// Reciprocal condition number for eigenvalue cluster// Estimated reciprocal condition number for invariant subspace
) where is_float(T) {
    n := schur.rows
    ldt := schur.ld
    ldq: Blas_Int = 1
    ptr_q: ^T = nil

    assert(schur.rows == schur.cols, "Matrix T must be square")
    assert(len(select) >= int(n), "select array too small")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(work) > 0, "work array must be provided")

    if compq == .Update && Q != nil {
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        ldq = Q.ld
        ptr_q = raw_data(Q.data)
    }

    job_c := cast(u8)job
    compq_c := cast(u8)compq
    lwork := Blas_Int(len(work))
    liwork := Blas_Int(len(iwork))

    when T == f32 {
        lapack.strsen_(
            &job_c,
            &compq_c,
            raw_data(select),
            &n,
            raw_data(schur.data),
            &ldt,
            ptr_q,
            &ldq,
            raw_data(WR),
            raw_data(WI),
            &m,
            &s,
            &sep,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    } else when T == f64 {
        lapack.dtrsen_(
            &job_c,
            &compq_c,
            raw_data(select),
            &n,
            raw_data(schur.data),
            &ldt,
            ptr_q,
            &ldq,
            raw_data(WR),
            raw_data(WI),
            &m,
            &s,
            &sep,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &liwork,
            &info,
        )
    }

    return m, s, sep, info, info == 0
}

// Reorder Schur form with condition number estimates (complex)
// Reorders the complex Schur factorization and computes condition numbers
reorder_schur_with_condition_complex :: proc(
    T: ^Matrix($Cmplx), // Schur form matrix (overwritten with reordered form)
    Q: ^Matrix(Cmplx), // Schur vectors (updated if compq == .Update)
    select: []Blas_Int, // Logical selection array (size n, 1 = select eigenvalue)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    work: []Cmplx, // Workspace (pre-allocated)
    s_out: ^$Real, // Output: reciprocal condition number for average of selected eigenvalues
    sep_out: ^Real, // Output: reciprocal condition number for right invariant subspace
    job: Condition_Job = .Both,
    compq: Schur_Computation = .Update,
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool,
) where is_complex(Cmplx) && ((Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)) {
    n := T.rows
    ldt := T.ld
    ldq: Blas_Int = 1
    ptr_q: ^Cmplx = nil

    assert(T.rows == T.cols, "Matrix T must be square")
    assert(len(select) >= int(n), "select array too small")
    assert(len(W) >= int(n), "W array too small")
    assert(len(work) > 0, "work array must be provided")

    if compq == .Update && Q != nil {
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        ldq = Q.ld
        ptr_q = raw_data(Q.data)
    }

    job_c := cast(u8)job
    compq_c := cast(u8)compq
    lwork := Blas_Int(len(work))

    s: Real
    sep: Real

    when Cmplx == complex64 {
        lapack.ctrsen_(
            &job_c,
            &compq_c,
            raw_data(select),
            &n,
            raw_data(T.data),
            &ldt,
            ptr_q,
            &ldq,
            raw_data(W),
            &m,
            &s,
            &sep,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztrsen_(
            &job_c,
            &compq_c,
            raw_data(select),
            &n,
            raw_data(T.data),
            &ldt,
            ptr_q,
            &ldq,
            raw_data(W),
            &m,
            &s,
            &sep,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    s_out^ = s
    sep_out^ = sep

    return m, info, info == 0
}

// ===================================================================================
// CONDITION NUMBERS FOR EIGENVALUES AND EIGENVECTORS
// ===================================================================================

// Job options for condition number computation
Sensitivity_Job :: enum u8 {
    Eigenvalues  = 'E', // Compute eigenvalue condition numbers only
    Eigenvectors = 'V', // Compute eigenvector condition numbers only
    Both         = 'B', // Compute both
}

// Options for selecting which eigenvalues/eigenvectors to analyze
Howmny :: enum u8 {
    All      = 'A', // Condition numbers for all eigenvalues/eigenvectors
    Selected = 'S', // Condition numbers for selected eigenvalues/eigenvectors
}

condition_numbers_eigenvalues :: proc {
    condition_numbers_eigenvalues_real,
    condition_numbers_eigenvalues_complex,
}

// Query workspace sizes for condition number computation
query_workspace_condition_numbers :: proc(
    schur: ^Matrix($T),
) -> (
    work: int,
    iwork: int,
) where is_float(T) ||
    is_complex(T) {
    n := int(schur.rows)

    when is_float(T) {
        work = 3 * n
        iwork = 2 * (n - 1)
    } else when is_complex(T) {
        work = 2 * n * n
        iwork = 0
    }

    return
}

// Compute condition numbers for eigenvalues and eigenvectors (real)
// Estimates reciprocal condition numbers for specified eigenvalues/eigenvectors
condition_numbers_eigenvalues_real :: proc(
    schur: ^Matrix($T), // Triangular matrix from Schur decomposition (not modified)
    VL: ^Matrix(T), // Left eigenvectors (optional, not modified)
    VR: ^Matrix(T), // Right eigenvectors (optional, not modified)
    select: []Blas_Int, // Selection array if howmny == .Selected (size n)
    S: []T, // Reciprocal condition numbers for eigenvalues (pre-allocated)
    SEP: []T, // Reciprocal condition numbers for eigenvectors (pre-allocated)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated for real types)
    job: Sensitivity_Job = .Both,
    howmny: Howmny = .All,
    mm: Blas_Int = 0, // Number of elements in S and SEP (0 = auto-detect from n)
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of condition numbers computed
) where is_float(T) {
    n := schur.rows
    ldt := schur.ld
    ldvl: Blas_Int = 1
    ldvr: Blas_Int = 1
    ptr_vl: ^T = nil
    ptr_vr: ^T = nil

    assert(schur.rows == schur.cols, "Matrix T must be square")

    if VL != nil {
        assert(VL.rows == n, "VL matrix dimensions incorrect")
        ldvl = VL.ld
        ptr_vl = raw_data(VL.data)
    }

    if VR != nil {
        assert(VR.rows == n, "VR matrix dimensions incorrect")
        ldvr = VR.ld
        ptr_vr = raw_data(VR.data)
    }

    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    mm_val := mm
    if mm_val == 0 {
        mm_val = n
    }

    assert(len(S) >= int(mm_val), "S array too small")
    assert(len(SEP) >= int(mm_val), "SEP array too small")
    assert(len(work) >= int(3 * n), "work array too small")
    assert(len(iwork) >= int(2 * (n - 1)), "iwork array too small")

    job_c := cast(u8)job
    howmny_c := cast(u8)howmny
    ldwork := n

    when T == f32 {
        lapack.strsna_(
            &job_c,
            &howmny_c,
            raw_data(select),
            &n,
            raw_data(schur.data),
            &ldt,
            ptr_vl,
            &ldvl,
            ptr_vr,
            &ldvr,
            raw_data(S),
            raw_data(SEP),
            &mm_val,
            &m,
            raw_data(work),
            &ldwork,
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dtrsna_(
            &job_c,
            &howmny_c,
            raw_data(select),
            &n,
            raw_data(schur.data),
            &ldt,
            ptr_vl,
            &ldvl,
            ptr_vr,
            &ldvr,
            raw_data(S),
            raw_data(SEP),
            &mm_val,
            &m,
            raw_data(work),
            &ldwork,
            raw_data(iwork),
            &info,
        )
    }

    return m, info, info == 0
}

// Compute condition numbers for eigenvalues and eigenvectors (complex)
// Estimates reciprocal condition numbers for specified eigenvalues/eigenvectors
condition_numbers_eigenvalues_complex :: proc(
    T: ^Matrix($Cmplx), // Triangular matrix from Schur decomposition (not modified)
    VL: ^Matrix(Cmplx), // Left eigenvectors (optional, not modified)
    VR: ^Matrix(Cmplx), // Right eigenvectors (optional, not modified)
    select: []Blas_Int, // Selection array if howmny == .Selected (size n)
    S: []$Real, // Reciprocal condition numbers for eigenvalues (pre-allocated)
    SEP: []Real, // Reciprocal condition numbers for eigenvectors (pre-allocated)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []Real, // Real workspace (pre-allocated)
    job: Sensitivity_Job = .Both,
    howmny: Howmny = .All,
    mm: Blas_Int = 0, // Number of elements in S and SEP (0 = auto-detect from n)
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of condition numbers computed
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := T.rows
    ldt := T.ld
    ldvl: Blas_Int = 1
    ldvr: Blas_Int = 1
    ptr_vl: ^Cmplx = nil
    ptr_vr: ^Cmplx = nil

    assert(T.rows == T.cols, "Matrix T must be square")

    if VL != nil {
        assert(VL.rows == n, "VL matrix dimensions incorrect")
        ldvl = VL.ld
        ptr_vl = raw_data(VL.data)
    }

    if VR != nil {
        assert(VR.rows == n, "VR matrix dimensions incorrect")
        ldvr = VR.ld
        ptr_vr = raw_data(VR.data)
    }

    if howmny == .Selected {
        assert(len(select) >= int(n), "select array too small")
    }

    mm_val := mm
    if mm_val == 0 {
        mm_val = n
    }

    assert(len(S) >= int(mm_val), "S array too small")
    assert(len(SEP) >= int(mm_val), "SEP array too small")
    assert(len(work) >= int(2 * n * n), "work array too small")
    assert(len(rwork) >= int(n), "rwork array too small")

    job_c := cast(u8)job
    howmny_c := cast(u8)howmny
    ldwork := n

    when Cmplx == complex64 {
        lapack.ctrsna_(
            &job_c,
            &howmny_c,
            raw_data(select),
            &n,
            raw_data(T.data),
            &ldt,
            ptr_vl,
            &ldvl,
            ptr_vr,
            &ldvr,
            raw_data(S),
            raw_data(SEP),
            &mm_val,
            &m,
            raw_data(work),
            &ldwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztrsna_(
            &job_c,
            &howmny_c,
            raw_data(select),
            &n,
            raw_data(T.data),
            &ldt,
            ptr_vl,
            &ldvl,
            ptr_vr,
            &ldvr,
            raw_data(S),
            raw_data(SEP),
            &mm_val,
            &m,
            raw_data(work),
            &ldwork,
            raw_data(rwork),
            &info,
        )
    }

    return m, info, info == 0
}

// ===================================================================================
// AUXILIARY EIGENVALUE FUNCTIONS
// ===================================================================================

// Job options for Hessenberg eigenvalue computation
Hessenberg_Job :: enum u8 {
    Eigenvalues = 'E', // Compute eigenvalues only
    Schur       = 'S', // Compute Schur form (T and eigenvalues)
}

// Compz options for Hessenberg routines
Compz :: enum u8 {
    None    = 'N', // No Schur vectors computed
    Init    = 'I', // Z is initialized to identity, then computed
    Vectors = 'V', // Z must contain an orthogonal matrix Q on entry, updated on exit
}

// Eigsrc options for hsein (eigenvalue source)
Eigsrc :: enum u8 {
    Qr     = 'Q', // Eigenvalues from QR algorithm
    Direct = 'N', // Eigenvalues from direct input
}

// Initv options for hsein (initial vectors)
Initv :: enum u8 {
    None = 'N', // No initial vectors
    Unit = 'U', // Use unit vectors as initial vectors
}

// ===================================================================================
// HGEQZ - Generalized Schur form using QZ algorithm
// ===================================================================================

hessenberg_qz_iteration :: proc {
    hessenberg_qz_iteration_real,
    hessenberg_qz_iteration_complex,
}

// Query workspace size for generalized Schur form (QZ algorithm)
query_workspace_hessenberg_qz_iteration :: proc(
    H: ^Matrix($T),
) -> (
    work: int,
    rwork: int,
) where is_float(T) ||
    is_complex(T) {
    n := H.rows
    job_c := cast(u8)Hessenberg_Job.Schur
    compq_c := cast(u8)Compz.Vectors
    compz_c := cast(u8)Compz.Vectors
    ilo: Blas_Int = 1
    ihi := n
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T

    when is_float(T) {
        when T == f32 {
            lapack.shgeqz_(
                &job_c,
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                nil,
                nil,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        } else when T == f64 {
            lapack.dhgeqz_(
                &job_c,
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                nil,
                nil,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        }
        work = int(work_query)
        rwork = 0
    } else when is_complex(T) {
        when T == complex64 {
            lapack.chgeqz_(
                &job_c,
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                nil,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                nil,
                &info,
            )
        } else when T == complex128 {
            lapack.zhgeqz_(
                &job_c,
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                nil,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                nil,
                &info,
            )
        }
        work = int(real(work_query))
        rwork = int(n)
    }

    return
}

// Compute generalized Schur form using QZ algorithm (real)
hessenberg_qz_iteration_real :: proc(
    H: ^Matrix($T), // Hessenberg matrix (input/output)
    T_mat: ^Matrix(T), // Triangular matrix (input/output)
    alphar: []T, // Real parts of alpha (pre-allocated, size n)
    alphai: []T, // Imaginary parts of alpha (pre-allocated, size n)
    beta: []T, // Beta values (pre-allocated, size n)
    Q: ^Matrix(T), // Left Schur vectors (optional, input/output)
    Z: ^Matrix(T), // Right Schur vectors (optional, input/output)
    work: []T, // Workspace (pre-allocated)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    job: Hessenberg_Job = .Schur,
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := H.rows
    ldh := H.ld
    ldt := T_mat.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(T_mat.rows == T_mat.cols, "Matrix T must be square")
    assert(H.rows == T_mat.rows, "Matrices H and T must have same dimensions")
    assert(len(alphar) >= int(n), "alphar array too small")
    assert(len(alphai) >= int(n), "alphai array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")

    job_c := cast(u8)job
    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^T = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^T = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.shgeqz_(
            &job_c,
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(T_mat.data),
            &ldt,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dhgeqz_(
            &job_c,
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(T_mat.data),
            &ldt,
            raw_data(alphar),
            raw_data(alphai),
            raw_data(beta),
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// Compute generalized Schur form using QZ algorithm (complex)
hessenberg_qz_iteration_complex :: proc(
    H: ^Matrix($Cmplx), // Hessenberg matrix (input/output)
    T_mat: ^Matrix(Cmplx), // Triangular matrix (input/output)
    alpha: []Cmplx, // Alpha values (pre-allocated, size n)
    beta: []Cmplx, // Beta values (pre-allocated, size n)
    Q: ^Matrix(Cmplx), // Left Schur vectors (optional, input/output)
    Z: ^Matrix(Cmplx), // Right Schur vectors (optional, input/output)
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated, size n)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    job: Hessenberg_Job = .Schur,
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := H.rows
    ldh := H.ld
    ldt := T_mat.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(T_mat.rows == T_mat.cols, "Matrix T must be square")
    assert(H.rows == T_mat.rows, "Matrices H and T must have same dimensions")
    assert(len(alpha) >= int(n), "alpha array too small")
    assert(len(beta) >= int(n), "beta array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(n), "rwork array too small")

    job_c := cast(u8)job
    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^Cmplx = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^Cmplx = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.chgeqz_(
            &job_c,
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(T_mat.data),
            &ldt,
            raw_data(alpha),
            raw_data(beta),
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhgeqz_(
            &job_c,
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(T_mat.data),
            &ldt,
            raw_data(alpha),
            raw_data(beta),
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// HSEIN - Inverse iteration for eigenvectors of upper Hessenberg
// ===================================================================================

hessenberg_eigenvectors_inverse :: proc {
    hessenberg_eigenvectors_inverse_real,
    hessenberg_eigenvectors_inverse_complex,
}

// Compute eigenvectors of Hessenberg matrix by inverse iteration (real)
hessenberg_eigenvectors_inverse_real :: proc(
    H: ^Matrix($T), // Upper Hessenberg matrix (not modified)
    WR: []T, // Real parts of eigenvalues (input)
    WI: []T, // Imaginary parts of eigenvalues (input)
    VL: ^Matrix(T), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(T), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (size n, all 1s for all eigenvectors)
    work: []T, // Workspace (pre-allocated, size n*(n+2))
    ifaill: []Blas_Int, // Failure flags for left eigenvectors (pre-allocated, size mm)
    ifailr: []Blas_Int, // Failure flags for right eigenvectors (pre-allocated, size mm)
    side: Side = .Right,
    eigsrc: Eigsrc = .Qr,
    initv: Initv = .None,
    mm: Blas_Int = 0, // Number of columns in VL/VR (0 = auto-detect from n)
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where is_float(T) {
    n := H.rows
    ldh := H.ld

    mm_val := mm
    if mm_val == 0 {
        mm_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(select) >= int(n), "select array too small")
    assert(len(work) >= int(n * (n + 2)), "work array too small")

    side_c := cast(u8)side
    eigsrc_c := cast(u8)eigsrc
    initv_c := cast(u8)initv

    ldvl: Blas_Int = 1
    vl_ptr: ^T = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= int(mm_val), "VL matrix dimensions incorrect")
        assert(len(ifaill) >= int(mm_val), "ifaill array too small")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^T = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= int(mm_val), "VR matrix dimensions incorrect")
        assert(len(ifailr) >= int(mm_val), "ifailr array too small")
        vr_ptr = raw_data(VR.data)
    }

    when T == f32 {
        lapack.shsein_(
            &side_c,
            &eigsrc_c,
            &initv_c,
            raw_data(select),
            &n,
            raw_data(H.data),
            &ldh,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm_val,
            &m,
            raw_data(work),
            raw_data(ifaill),
            raw_data(ifailr),
            &info,
        )
    } else when T == f64 {
        lapack.dhsein_(
            &side_c,
            &eigsrc_c,
            &initv_c,
            raw_data(select),
            &n,
            raw_data(H.data),
            &ldh,
            raw_data(WR),
            raw_data(WI),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm_val,
            &m,
            raw_data(work),
            raw_data(ifaill),
            raw_data(ifailr),
            &info,
        )
    }

    return m, info, info == 0
}

// Compute eigenvectors of Hessenberg matrix by inverse iteration (complex)
hessenberg_eigenvectors_inverse_complex :: proc(
    H: ^Matrix($Cmplx), // Upper Hessenberg matrix (not modified)
    W: []Cmplx, // Eigenvalues (input)
    VL: ^Matrix(Cmplx), // Left eigenvectors (pre-allocated, optional)
    VR: ^Matrix(Cmplx), // Right eigenvectors (pre-allocated, optional)
    select: []Blas_Int, // Selection array (size n, all 1s for all eigenvectors)
    work: []Cmplx, // Workspace (pre-allocated, size n*n)
    rwork: []$Real, // Real workspace (pre-allocated, size n)
    ifaill: []Blas_Int, // Failure flags for left eigenvectors (pre-allocated, size mm)
    ifailr: []Blas_Int, // Failure flags for right eigenvectors (pre-allocated, size mm)
    side: Side = .Right,
    eigsrc: Eigsrc = .Qr,
    initv: Initv = .None,
    mm: Blas_Int = 0, // Number of columns in VL/VR (0 = auto-detect from n)
) -> (
    m: Blas_Int,
    info: Info,
    ok: bool, // Number of eigenvectors computed
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    n := H.rows
    ldh := H.ld

    mm_val := mm
    if mm_val == 0 {
        mm_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(select) >= int(n), "select array too small")
    assert(len(work) >= int(n * n), "work array too small")
    assert(len(rwork) >= int(n), "rwork array too small")

    side_c := cast(u8)side
    eigsrc_c := cast(u8)eigsrc
    initv_c := cast(u8)initv

    ldvl: Blas_Int = 1
    vl_ptr: ^Cmplx = nil
    if (side == .Left || side == .Both) && VL != nil {
        ldvl = VL.ld
        assert(VL.rows == n && VL.cols >= int(mm_val), "VL matrix dimensions incorrect")
        assert(len(ifaill) >= int(mm_val), "ifaill array too small")
        vl_ptr = raw_data(VL.data)
    }

    ldvr: Blas_Int = 1
    vr_ptr: ^Cmplx = nil
    if (side == .Right || side == .Both) && VR != nil {
        ldvr = VR.ld
        assert(VR.rows == n && VR.cols >= int(mm_val), "VR matrix dimensions incorrect")
        assert(len(ifailr) >= int(mm_val), "ifailr array too small")
        vr_ptr = raw_data(VR.data)
    }

    when Cmplx == complex64 {
        lapack.chsein_(
            &side_c,
            &eigsrc_c,
            &initv_c,
            raw_data(select),
            &n,
            raw_data(H.data),
            &ldh,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm_val,
            &m,
            raw_data(work),
            raw_data(rwork),
            raw_data(ifaill),
            raw_data(ifailr),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhsein_(
            &side_c,
            &eigsrc_c,
            &initv_c,
            raw_data(select),
            &n,
            raw_data(H.data),
            &ldh,
            raw_data(W),
            vl_ptr,
            &ldvl,
            vr_ptr,
            &ldvr,
            &mm_val,
            &m,
            raw_data(work),
            raw_data(rwork),
            raw_data(ifaill),
            raw_data(ifailr),
            &info,
        )
    }

    return m, info, info == 0
}

// ===================================================================================
// HSEQR - Compute Schur form of upper Hessenberg
// ===================================================================================

hessenberg_to_schur :: proc {
    hessenberg_to_schur_real,
    hessenberg_to_schur_complex,
}

// Query workspace size for Hessenberg Schur form
query_workspace_hessenberg_to_schur :: proc(H: ^Matrix($T)) -> (work: int) where is_float(T) || is_complex(T) {
    n := H.rows
    job_c := cast(u8)Hessenberg_Job.Schur
    compz_c := cast(u8)Compz.Vectors
    ilo: Blas_Int = 1
    ihi := n
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T

    when is_float(T) {
        when T == f32 {
            lapack.shseqr_(&job_c, &compz_c, &n, &ilo, &ihi, nil, &n, nil, nil, nil, &n, &work_query, &lwork, &info)
        } else when T == f64 {
            lapack.dhseqr_(&job_c, &compz_c, &n, &ilo, &ihi, nil, &n, nil, nil, nil, &n, &work_query, &lwork, &info)
        }
        work = int(work_query)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.chseqr_(&job_c, &compz_c, &n, &ilo, &ihi, nil, &n, nil, nil, &n, &work_query, &lwork, &info)
        } else when T == complex128 {
            lapack.zhseqr_(&job_c, &compz_c, &n, &ilo, &ihi, nil, &n, nil, nil, &n, &work_query, &lwork, &info)
        }
        work = int(real(work_query))
    }

    return
}

// Compute eigenvalues/Schur form of Hessenberg matrix (real)
hessenberg_to_schur_real :: proc(
    H: ^Matrix($T), // Upper Hessenberg matrix (input/output)
    WR: []T, // Real parts of eigenvalues (pre-allocated, size n)
    WI: []T, // Imaginary parts of eigenvalues (pre-allocated, size n)
    Z: ^Matrix(T), // Schur vectors (optional, input/output)
    work: []T, // Workspace (pre-allocated)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    job: Hessenberg_Job = .Schur,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := H.rows
    ldh := H.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(len(WR) >= int(n), "WR array too small")
    assert(len(WI) >= int(n), "WI array too small")
    assert(len(work) > 0, "work array must be provided")

    job_c := cast(u8)job
    compz_c := cast(u8)compz

    ldz: Blas_Int = 1
    z_ptr: ^T = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.shseqr_(
            &job_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(WR),
            raw_data(WI),
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dhseqr_(
            &job_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(WR),
            raw_data(WI),
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// Compute eigenvalues/Schur form of Hessenberg matrix (complex)
hessenberg_to_schur_complex :: proc(
    H: ^Matrix($Cmplx), // Upper Hessenberg matrix (input/output)
    W: []Cmplx, // Eigenvalues (pre-allocated, size n)
    Z: ^Matrix(Cmplx), // Schur vectors (optional, input/output)
    work: []Cmplx, // Workspace (pre-allocated)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    job: Hessenberg_Job = .Schur,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) {
    n := H.rows
    ldh := H.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(H.rows == H.cols, "Matrix H must be square")
    assert(len(W) >= int(n), "W array too small")
    assert(len(work) > 0, "work array must be provided")

    job_c := cast(u8)job
    compz_c := cast(u8)compz

    ldz: Blas_Int = 1
    z_ptr: ^Cmplx = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.chseqr_(
            &job_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(W),
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zhseqr_(
            &job_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(H.data),
            &ldh,
            raw_data(W),
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// GGBAK - Back-transform eigenvectors after balancing
// ===================================================================================

balance_backtransform_generalized :: proc {
    balance_backtransform_generalized_real,
    balance_backtransform_generalized_complex,
}

// Back-transform eigenvectors after generalized balancing (real)
balance_backtransform_generalized_real :: proc(
    lscale: []$T, // Left scale factors from ggbal (input, size n)
    rscale: []T, // Right scale factors from ggbal (input, size n)
    V: ^Matrix(T), // Eigenvectors to transform (input/output)
    ilo: Blas_Int,
    ihi: Blas_Int,
    job: Balance_Job = .Both,
    side: Side = .Right,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    n := Blas_Int(len(lscale))
    m := V.cols
    ldv := V.ld

    assert(len(lscale) == len(rscale), "lscale and rscale must have same length")
    assert(V.rows == int(n), "V matrix rows must match scale array size")

    job_c := cast(u8)job
    side_c := cast(u8)side

    when T == f32 {
        lapack.sggbak_(
            &job_c,
            &side_c,
            &n,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &m,
            raw_data(V.data),
            &ldv,
            &info,
        )
    } else when T == f64 {
        lapack.dggbak_(
            &job_c,
            &side_c,
            &n,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &m,
            raw_data(V.data),
            &ldv,
            &info,
        )
    } else when T == complex64 {
        lapack.cggbak_(
            &job_c,
            &side_c,
            &n,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &m,
            raw_data(V.data),
            &ldv,
            &info,
        )
    } else when T == complex128 {
        lapack.zggbak_(
            &job_c,
            &side_c,
            &n,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            &m,
            raw_data(V.data),
            &ldv,
            &info,
        )
    }

    return info, info == 0
}

// Back-transform eigenvectors after generalized balancing (complex)
balance_backtransform_generalized_complex :: proc(
    lscale: []$Real, // Left scale factors from ggbal (input, size n)
    rscale: []Real, // Right scale factors from ggbal (input, size n)
    V: ^Matrix($Cmplx), // Eigenvectors to transform (input/output)
    ilo: Blas_Int,
    ihi: Blas_Int,
    job: Balance_Job = .Both,
    side: Side = .Right,
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    return ggbak_real(lscale, rscale, V, ilo, ihi, job, side)
}

// ===================================================================================
// GGBAL - Balance matrix pair for better eigenvalue conditioning
// ===================================================================================
// Balance matrix pair for generalized eigenvalue problem
balance_generalized :: proc(
    A: ^Matrix($T), // First matrix (input/output)
    B: ^Matrix(T), // Second matrix (input/output)
    lscale: []T, // Left scale factors (pre-allocated, size n)
    rscale: []T, // Right scale factors (pre-allocated, size n)
    work: []T, // Workspace (pre-allocated, size 6*n)
    job: Balance_Job = .Both,
) -> (
    ilo: Blas_Int,
    ihi: Blas_Int,
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(lscale) >= int(n), "lscale array too small")
    assert(len(rscale) >= int(n), "rscale array too small")
    assert(len(work) >= int(6 * n), "work array too small")

    job_c := cast(u8)job

    when T == f32 {
        lapack.sggbal_(
            &job_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            raw_data(work),
            &info,
        )
    } else when T == f64 {
        lapack.dggbal_(
            &job_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            raw_data(work),
            &info,
        )
    } else when T == complex64 {
        lapack.cggbal_(
            &job_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            raw_data(work),
            &info,
        )
    } else when T == complex128 {
        lapack.zggbal_(
            &job_c,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            &ilo,
            &ihi,
            raw_data(lscale),
            raw_data(rscale),
            raw_data(work),
            &info,
        )
    }

    return ilo, ihi, info, info == 0
}


// ===================================================================================
// GGHRD - Reduce to generalized Hessenberg form
// ===================================================================================

// Query workspace size for generalized Hessenberg reduction (blocked algorithm)
query_workspace_reduce_to_hessenberg_generalized_blocked :: proc(
    A: ^Matrix($T),
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    compq_c := cast(u8)Compz.Vectors
    compz_c := cast(u8)Compz.Vectors
    ilo: Blas_Int = 1
    ihi := n
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T

    when is_float(T) {
        when T == f32 {
            lapack.sgghd3_(
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        } else when T == f64 {
            lapack.dgghd3_(
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        }
        work_size = int(work_query)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.cgghd3_(
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        } else when T == complex128 {
            lapack.zgghd3_(
                &compq_c,
                &compz_c,
                &n,
                &ilo,
                &ihi,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &work_query,
                &lwork,
                &info,
            )
        }
        work_size = int(real(work_query))
    }

    return
}

// Reduce matrix pair to generalized Hessenberg form (real, no workspace)
reduce_to_hessenberg_generalized_basic :: proc {
    reduce_to_hessenberg_generalized_real_basic,
    reduce_to_hessenberg_generalized_complex_basic,
}

// Reduce matrix pair to generalized Hessenberg form - basic algorithm (real)
reduce_to_hessenberg_generalized_real_basic :: proc(
    A: ^Matrix($T), // First matrix (input/output)
    B: ^Matrix(T), // Second matrix (input/output)
    Q: ^Matrix(T), // Left orthogonal matrix (optional, input/output)
    Z: ^Matrix(T), // Right orthogonal matrix (optional, input/output)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")

    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^T = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^T = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    when T == f32 {
        lapack.sgghrd_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            &info,
        )
    } else when T == f64 {
        lapack.dgghrd_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            &info,
        )
    }

    return info, info == 0
}

// Reduce matrix pair to generalized Hessenberg form - blocked algorithm (real)
reduce_to_hessenberg_generalized_real_blocked :: proc(
    A: ^Matrix($T), // First matrix (input/output)
    B: ^Matrix(T), // Second matrix (input/output)
    Q: ^Matrix(T), // Left orthogonal matrix (optional, input/output)
    Z: ^Matrix(T), // Right orthogonal matrix (optional, input/output)
    work: []T, // Workspace (pre-allocated)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(work) > 0, "work array must be provided")

    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^T = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^T = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgghd3_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dgghd3_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// Reduce matrix pair to generalized Hessenberg form (complex)
reduce_to_hessenberg_generalized_blocked :: proc {
    reduce_to_hessenberg_generalized_real_blocked,
    reduce_to_hessenberg_generalized_complex_blocked,
}

// Reduce matrix pair to generalized Hessenberg form - basic algorithm (complex)
reduce_to_hessenberg_generalized_complex_basic :: proc(
    A: ^Matrix($Cmplx), // First matrix (input/output)
    B: ^Matrix(Cmplx), // Second matrix (input/output)
    Q: ^Matrix(Cmplx), // Left unitary matrix (optional, input/output)
    Z: ^Matrix(Cmplx), // Right unitary matrix (optional, input/output)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")

    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^Cmplx = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^Cmplx = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    when Cmplx == complex64 {
        lapack.cgghrd_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgghrd_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            &info,
        )
    }

    return info, info == 0
}

// Reduce matrix pair to generalized Hessenberg form - blocked algorithm (complex)
reduce_to_hessenberg_generalized_complex_blocked :: proc(
    A: ^Matrix($Cmplx), // First matrix (input/output)
    B: ^Matrix(Cmplx), // Second matrix (input/output)
    Q: ^Matrix(Cmplx), // Left unitary matrix (optional, input/output)
    Z: ^Matrix(Cmplx), // Right unitary matrix (optional, input/output)
    work: []Cmplx, // Workspace (pre-allocated)
    ilo: Blas_Int = 1,
    ihi: Blas_Int = 0, // 0 = auto (set to n)
    compq: Compz = .Vectors,
    compz: Compz = .Vectors,
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) {
    n := A.rows
    lda := A.ld
    ldb := B.ld

    ihi_val := ihi
    if ihi_val == 0 {
        ihi_val = n
    }

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
    assert(len(work) > 0, "work array must be provided")

    compq_c := cast(u8)compq
    compz_c := cast(u8)compz

    ldq: Blas_Int = 1
    q_ptr: ^Cmplx = nil
    if compq != .None && Q != nil {
        ldq = Q.ld
        assert(Q.rows == n && Q.cols == n, "Q matrix dimensions incorrect")
        q_ptr = raw_data(Q.data)
    }

    ldz: Blas_Int = 1
    z_ptr: ^Cmplx = nil
    if compz != .None && Z != nil {
        ldz = Z.ld
        assert(Z.rows == n && Z.cols == n, "Z matrix dimensions incorrect")
        z_ptr = raw_data(Z.data)
    }

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgghd3_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgghd3_(
            &compq_c,
            &compz_c,
            &n,
            &ilo,
            &ihi_val,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            q_ptr,
            &ldq,
            z_ptr,
            &ldz,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// TGSYL - Solve generalized Sylvester equation
// ===================================================================================

sylvester_generalized :: proc {
    sylvester_generalized_real,
    sylvester_generalized_complex,
}

// Query workspace size for generalized Sylvester equation
query_workspace_sylvester_generalized :: proc(
    A: ^Matrix($T),
    B: ^Matrix(T),
) -> (
    work_size: int,
    iwork_size: int,
) where is_float(T) ||
    is_complex(T) {
    m := A.rows
    n := B.rows
    trans_c := cast(u8)Transpose.None
    ijob: Blas_Int = 0
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T
    iwork_query: Blas_Int

    when is_float(T) {
        dif: T
        scale: T
        when T == f32 {
            lapack.stgsyl_(
                &trans_c,
                &ijob,
                &m,
                &n,
                nil,
                &m,
                nil,
                &m,
                nil,
                &m,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &dif,
                &scale,
                &work_query,
                &lwork,
                &iwork_query,
                &info,
            )
        } else when T == f64 {
            lapack.dtgsyl_(
                &trans_c,
                &ijob,
                &m,
                &n,
                nil,
                &m,
                nil,
                &m,
                nil,
                &m,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &dif,
                &scale,
                &work_query,
                &lwork,
                &iwork_query,
                &info,
            )
        }
        work_size = int(work_query)
        iwork_size = int(iwork_query)
    } else when is_complex(T) {
        when T == complex64 {
            dif: f32
            scale: f32
            lapack.ctgsyl_(
                &trans_c,
                &ijob,
                &m,
                &n,
                nil,
                &m,
                nil,
                &m,
                nil,
                &m,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &dif,
                &scale,
                &work_query,
                &lwork,
                &iwork_query,
                &info,
            )
        } else when T == complex128 {
            dif: f64
            scale: f64
            lapack.ztgsyl_(
                &trans_c,
                &ijob,
                &m,
                &n,
                nil,
                &m,
                nil,
                &m,
                nil,
                &m,
                nil,
                &n,
                nil,
                &n,
                nil,
                &n,
                &dif,
                &scale,
                &work_query,
                &lwork,
                &iwork_query,
                &info,
            )
        }
        work_size = int(real(work_query))
        iwork_size = int(iwork_query)
    }

    return
}

// Solve generalized Sylvester equation (real)
sylvester_generalized_real :: proc(
    A: ^Matrix($T), // First matrix on left side (not modified)
    B: ^Matrix(T), // Second matrix on left side (not modified)
    C: ^Matrix(T), // First matrix in equation (input/output)
    D: ^Matrix(T), // Third matrix on left side (not modified)
    E: ^Matrix(T), // Fourth matrix on left side (not modified)
    F: ^Matrix(T), // Second matrix in equation (input/output)
    work: []T, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    trans: Transpose = .None,
    ijob: Blas_Int = 0, // 0 = solve only, 1-4 = solve + condition estimate
) -> (
    dif: T,
    scale: T,
    info: Info,
    ok: bool, // Estimate of Dif[(A,D), (B,E)]// Scale factor
) where is_float(T) {
    m := A.rows
    n := B.rows
    lda := A.ld
    ldb := B.ld
    ldc := C.ld
    ldd := D.ld
    lde := E.ld
    ldf := F.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(D.rows == D.cols, "Matrix D must be square")
    assert(E.rows == E.cols, "Matrix E must be square")
    assert(C.rows == int(m) && C.cols == int(n), "C dimensions must be m x n")
    assert(F.rows == int(m) && F.cols == int(n), "F dimensions must be m x n")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) > 0, "iwork array must be provided")

    trans_c := cast(u8)trans
    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.stgsyl_(
            &trans_c,
            &ijob,
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(C.data),
            &ldc,
            raw_data(D.data),
            &ldd,
            raw_data(E.data),
            &lde,
            raw_data(F.data),
            &ldf,
            &dif,
            &scale,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    } else when T == f64 {
        lapack.dtgsyl_(
            &trans_c,
            &ijob,
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(C.data),
            &ldc,
            raw_data(D.data),
            &ldd,
            raw_data(E.data),
            &lde,
            raw_data(F.data),
            &ldf,
            &dif,
            &scale,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    }

    return dif, scale, info, info == 0
}

// Solve generalized Sylvester equation (complex)
sylvester_generalized_complex :: proc(
    A: ^Matrix($Cmplx), // First matrix on left side (not modified)
    B: ^Matrix(Cmplx), // Second matrix on left side (not modified)
    C: ^Matrix(Cmplx), // First matrix in equation (input/output)
    D: ^Matrix(Cmplx), // Third matrix on left side (not modified)
    E: ^Matrix(Cmplx), // Fourth matrix on left side (not modified)
    F: ^Matrix(Cmplx), // Second matrix in equation (input/output)
    work: []Cmplx, // Workspace (pre-allocated)
    iwork: []Blas_Int, // Integer workspace (pre-allocated)
    dif_out: ^$Real, // Output: estimate of Dif[(A,D), (B,E)]
    scale_out: ^Real, // Output: scaling factor for C and F
    trans: Transpose = .None,
    ijob: Blas_Int = 0, // 0 = solve only, 1-4 = solve + condition estimate
) -> (
    info: Info,
    ok: bool,
) where is_complex(Cmplx) && ((Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64)) {
    m := A.rows
    n := B.rows
    lda := A.ld
    ldb := B.ld
    ldc := C.ld
    ldd := D.ld
    lde := E.ld
    ldf := F.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(B.rows == B.cols, "Matrix B must be square")
    assert(D.rows == D.cols, "Matrix D must be square")
    assert(E.rows == E.cols, "Matrix E must be square")
    assert(C.rows == int(m) && C.cols == int(n), "C dimensions must be m x n")
    assert(F.rows == int(m) && F.cols == int(n), "F dimensions must be m x n")
    assert(len(work) > 0, "work array must be provided")
    assert(len(iwork) > 0, "iwork array must be provided")

    trans_c := cast(u8)trans
    lwork := Blas_Int(len(work))

    dif: Real
    scale: Real

    when Cmplx == complex64 {
        lapack.ctgsyl_(
            &trans_c,
            &ijob,
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(C.data),
            &ldc,
            raw_data(D.data),
            &ldd,
            raw_data(E.data),
            &lde,
            raw_data(F.data),
            &ldf,
            &dif,
            &scale,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.ztgsyl_(
            &trans_c,
            &ijob,
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(B.data),
            &ldb,
            raw_data(C.data),
            &ldc,
            raw_data(D.data),
            &ldd,
            raw_data(E.data),
            &lde,
            raw_data(F.data),
            &ldf,
            &dif,
            &scale,
            raw_data(work),
            &lwork,
            raw_data(iwork),
            &info,
        )
    }

    dif_out^ = dif
    scale_out^ = scale

    return info, info == 0
}
