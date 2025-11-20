package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// QR FACTORIZATION (GEQRF family)
// ===================================================================================
//

// Query workspace size for QR factorization
query_workspace_dns_qr_factorize :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgeqrf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgeqrf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgeqrf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgeqrf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// QR factorization: A = Q * R
dns_qr_factorize :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// BLOCKED QR FACTORIZATION (GEQRT family)
// ===================================================================================

// Query workspace size for blocked QR factorization
query_workspace_dns_qr_factorize_blocked :: proc(
    A: ^Matrix($T),
    nb: Blas_Int,
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.cols
    work_size = int(nb * n)
    return work_size
}

// Blocked QR factorization with explicit T factor
dns_qr_factorize_blocked :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    T_matrix: ^Matrix(T), // T factor (pre-allocated, nb x min(m,n))
    work: []T, // Workspace (pre-allocated)
    nb: Blas_Int, // Block size
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    ldt := T_matrix.ld

    assert(T_matrix.rows >= nb, "T matrix has insufficient rows")
    assert(T_matrix.cols >= min(m, n), "T matrix has insufficient columns")
    assert(len(work) >= int(nb * n), "work array too small")

    when T == f32 {
        lapack.sgeqrt_(&m, &n, &nb, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, raw_data(work), &info)
    } else when T == f64 {
        lapack.dgeqrt_(&m, &n, &nb, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, raw_data(work), &info)
    } else when T == complex64 {
        lapack.cgeqrt_(&m, &n, &nb, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, raw_data(work), &info)
    } else when T == complex128 {
        lapack.zgeqrt_(&m, &n, &nb, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, raw_data(work), &info)
    }

    return info, info == 0
}

// ===================================================================================
// LQ FACTORIZATION (GELQF family)
// ===================================================================================

// Query workspace size for LQ factorization
query_workspace_dns_lq_factorize :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgelqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgelqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgelqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgelqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// LQ factorization: A = L * Q
dns_lq_factorize :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with L and Q factors)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// HESSENBERG REDUCTION (GEHRD family)
// ===================================================================================

// Query workspace size for Hessenberg reduction
query_workspace_dns_hessenberg_reduce :: proc(
    A: ^Matrix($T),
    ilo: Blas_Int = 1,
    ihi: Blas_Int = -1,
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    n := A.rows
    lda := A.ld

    ihi_val := ihi
    if ihi_val < 0 {
        ihi_val = n
    }

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgehrd_(&n, &ilo, &ihi_val, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgehrd_(&n, &ilo, &ihi_val, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgehrd_(&n, &ilo, &ihi_val, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgehrd_(&n, &ilo, &ihi_val, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// Reduce a general matrix to upper Hessenberg form
dns_hessenberg_reduce :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Hessenberg form)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size n-1)
    work: []T, // Workspace (pre-allocated)
    ilo: Blas_Int = 1, // First row/column to be reduced
    ihi: Blas_Int = -1, // Last row/column to be reduced (-1 = n)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    n := A.rows
    lda := A.ld

    assert(A.rows == A.cols, "Matrix A must be square")
    assert(len(tau) >= int(n - 1), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    ihi_val := ihi
    if ihi_val < 0 {
        ihi_val = n
    }

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgehrd_(&n, &ilo, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgehrd_(&n, &ilo, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgehrd_(&n, &ilo, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgehrd_(&n, &ilo, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// QR WITH PIVOTING (GEQPF family - deprecated, use GEQP3)
// ===================================================================================

dns_qr_pivoted :: proc {
    dns_qr_pivoted_real,
    dns_qr_pivoted_complex,
}

// QR factorization with column pivoting (legacy interface)
dns_qr_pivoted_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated, size 3*n)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(jpvt) >= int(n), "jpvt array too small")
    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) >= int(3 * n), "work array too small")

    // Initialize pivot array to zero (LAPACK will determine optimal pivoting)
    for i in 0 ..< int(n) {
        jpvt[i] = 0
    }

    when T == f32 {
        lapack.sgeqpf_(&m, &n, raw_data(A.data), &lda, raw_data(jpvt), raw_data(tau), raw_data(work), &info)
    } else when T == f64 {
        lapack.dgeqpf_(&m, &n, raw_data(A.data), &lda, raw_data(jpvt), raw_data(tau), raw_data(work), &info)
    }

    return info, info == 0
}

dns_qr_pivoted_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten with Q and R factors)
    jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
    tau: []Cmplx, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []Cmplx, // Workspace (pre-allocated, size n)
    rwork: []$Real, // Real workspace (pre-allocated, size 2*n)
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(jpvt) >= int(n), "jpvt array too small")
    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) >= int(n), "work array too small")
    assert(len(rwork) >= int(2 * n), "rwork array too small")

    // Initialize pivot array to zero
    for i in 0 ..< int(n) {
        jpvt[i] = 0
    }

    when Cmplx == complex64 {
        lapack.cgeqpf_(
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(jpvt),
            raw_data(tau),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgeqpf_(
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(jpvt),
            raw_data(tau),
            raw_data(work),
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// QL FACTORIZATION (GEQLF family)
// ===================================================================================

// Query workspace size for QL factorization
query_workspace_dns_ql_factorize :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgeqlf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgeqlf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgeqlf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgeqlf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// QL factorization: A = Q * L
dns_ql_factorize :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and L factors)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// RQ FACTORIZATION (GERQF family)
// ===================================================================================

// Query workspace size for RQ factorization
query_workspace_dns_rq_factorize :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgerqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgerqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgerqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgerqf_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// RQ factorization: A = R * Q
dns_rq_factorize :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with R and Q factors)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// QR WITH COLUMN PIVOTING (GEQP3 family)
// ===================================================================================

dns_qr_pivoted_v3 :: proc {
    dns_qr_pivoted_v3_real,
    dns_qr_pivoted_v3_complex,
}

// Query workspace size for QR with column pivoting (GEQP3)
query_workspace_dns_qr_pivoted_v3 :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld

    lwork: Blas_Int = QUERY_WORKSPACE

    when T == f32 {
        work_query: f32
        info: Info
        lapack.sgeqp3_(&m, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == f64 {
        work_query: f64
        info: Info
        lapack.dgeqp3_(&m, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
        work_size = int(work_query)
    } else when T == complex64 {
        work_query: complex64
        info: Info
        lapack.cgeqp3_(&m, &n, nil, &lda, nil, nil, &work_query, &lwork, nil, &info)
        work_size = int(real(work_query))
    } else when T == complex128 {
        work_query: complex128
        info: Info
        lapack.zgeqp3_(&m, &n, nil, &lda, nil, nil, &work_query, &lwork, nil, &info)
        work_size = int(real(work_query))
    }

    return work_size
}

// QR factorization with column pivoting: A*P = Q*R
dns_qr_pivoted_v3_real :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n, input/output)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(jpvt) >= int(n), "jpvt array too small")
    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeqp3_(&m, &n, raw_data(A.data), &lda, raw_data(jpvt), raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgeqp3_(&m, &n, raw_data(A.data), &lda, raw_data(jpvt), raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

dns_qr_pivoted_v3_complex :: proc(
    A: ^Matrix($Cmplx), // Input matrix (overwritten with Q and R factors)
    jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n, input/output)
    tau: []Cmplx, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []Cmplx, // Workspace (pre-allocated)
    rwork: []$Real, // Real workspace (pre-allocated, size 2*n)
) -> (
    info: Info,
    ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(jpvt) >= int(n), "jpvt array too small")
    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")
    assert(len(rwork) >= int(2 * n), "rwork array too small")

    lwork := Blas_Int(len(work))

    when Cmplx == complex64 {
        lapack.cgeqp3_(
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(jpvt),
            raw_data(tau),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    } else when Cmplx == complex128 {
        lapack.zgeqp3_(
            &m,
            &n,
            raw_data(A.data),
            &lda,
            raw_data(jpvt),
            raw_data(tau),
            raw_data(work),
            &lwork,
            raw_data(rwork),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// QR WITH NON-NEGATIVE DIAGONAL (GEQRFP family)
// ===================================================================================

// Query workspace size for QR with non-negative diagonal
query_workspace_dns_qr_nonnegative_diag :: proc(A: ^Matrix($T)) -> (work: int) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T

    when is_float(T) {
        when T == f32 {
            lapack.sgeqrfp_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        } else when T == f64 {
            lapack.dgeqrfp_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        }
        work = int(work_query)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.cgeqrfp_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        } else when T == complex128 {
            lapack.zgeqrfp_(&m, &n, nil, &lda, nil, &work_query, &lwork, &info)
        }
        work = int(real(work_query))
    }

    return
}

// QR factorization with non-negative diagonal elements: A = Q * R (R has non-negative diagonal)
dns_qr_nonnegative_diag :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    tau: []T, // Scalar factors for elementary reflectors (pre-allocated, size min(m,n))
    work: []T, // Workspace (pre-allocated)
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    min_mn := min(m, n)

    assert(len(tau) >= int(min_mn), "tau array too small")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == f64 {
        lapack.dgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex64 {
        lapack.cgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    } else when T == complex128 {
        lapack.zgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
    }

    return info, info == 0
}

// ===================================================================================
// TALL-SKINNY QR (GETSQRHRT family)
// ===================================================================================

// Query workspace size for tall-skinny QR
query_workspace_dns_qr_tall_skinny :: proc(
    A: ^Matrix($T),
    mb1: Blas_Int,
    nb1: Blas_Int,
    nb2: Blas_Int,
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    lwork: Blas_Int = QUERY_WORKSPACE
    info: Info
    work_query: T

    when is_float(T) {
        when T == f32 {
            lapack.sgetsqrhrt_(&m, &n, &mb1, &nb1, &nb2, nil, &lda, nil, &n, &work_query, &lwork, &info)
        } else when T == f64 {
            lapack.dgetsqrhrt_(&m, &n, &mb1, &nb1, &nb2, nil, &lda, nil, &n, &work_query, &lwork, &info)
        }
        work_size = int(work_query)
    } else when is_complex(T) {
        when T == complex64 {
            lapack.cgetsqrhrt_(&m, &n, &mb1, &nb1, &nb2, nil, &lda, nil, &n, &work_query, &lwork, &info)
        } else when T == complex128 {
            lapack.zgetsqrhrt_(&m, &n, &mb1, &nb1, &nb2, nil, &lda, nil, &n, &work_query, &lwork, &info)
        }
        work_size = int(real(work_query))
    }

    return
}

// Optimized QR factorization for tall-skinny matrices (m >> n)
dns_qr_tall_skinny :: proc(
    A: ^Matrix($T), // Input matrix (overwritten with Q and R factors)
    T_matrix: ^Matrix(T), // T factor (pre-allocated, nb2 x n)
    work: []T, // Workspace (pre-allocated)
    mb1: Blas_Int, // First blocking parameter
    nb1: Blas_Int, // Second blocking parameter
    nb2: Blas_Int, // Third blocking parameter
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols
    lda := A.ld
    ldt := T_matrix.ld

    assert(T_matrix.rows >= nb2, "T matrix has insufficient rows")
    assert(T_matrix.cols >= n, "T matrix has insufficient columns")
    assert(len(work) > 0, "work array must be provided")

    lwork := Blas_Int(len(work))

    when T == f32 {
        lapack.sgetsqrhrt_(
            &m,
            &n,
            &mb1,
            &nb1,
            &nb2,
            raw_data(A.data),
            &lda,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == f64 {
        lapack.dgetsqrhrt_(
            &m,
            &n,
            &mb1,
            &nb1,
            &nb2,
            raw_data(A.data),
            &lda,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == complex64 {
        lapack.cgetsqrhrt_(
            &m,
            &n,
            &mb1,
            &nb1,
            &nb2,
            raw_data(A.data),
            &lda,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(work),
            &lwork,
            &info,
        )
    } else when T == complex128 {
        lapack.zgetsqrhrt_(
            &m,
            &n,
            &mb1,
            &nb1,
            &nb2,
            raw_data(A.data),
            &lda,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(work),
            &lwork,
            &info,
        )
    }

    return info, info == 0
}
