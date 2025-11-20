package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"


// ===================================================================================
// SINGLE HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying Householder reflector
query_workspace_householder_reflector :: proc(m: int, n: int, side: ReflectorSide) -> (work_size: int) {
    return side == .Left ? n : m
}

solve_apply_householder_reflector :: proc(
    side: ReflectorSide,
    m: int,
    n: int,
    V: ^Vector($T),
    tau: T,
    C: ^Matrix(T),
    work: []T,
) where is_float(T) ||
    is_complex(T) {
    work_size := side == .Left ? n : m
    assert(len(work) >= work_size, "Work array too small")

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    incv := V.incr
    ldc := C.ld
    side_c := cast(u8)side
    tau_val := tau

    when T == f32 {
        lapack.slarf_(
            &side_c,
            &m_int,
            &n_int,
            &V.data[V.offset],
            &incv,
            &tau_val,
            raw_data(C.data),
            &ldc,
            raw_data(work),
        )
    } else when T == f64 {
        lapack.dlarf_(
            &side_c,
            &m_int,
            &n_int,
            &V.data[V.offset],
            &incv,
            &tau_val,
            raw_data(C.data),
            &ldc,
            raw_data(work),
        )
    } else when T == complex64 {
        lapack.clarf_(
            &side_c,
            &m_int,
            &n_int,
            &V.data[V.offset],
            &incv,
            &tau_val,
            raw_data(C.data),
            &ldc,
            raw_data(work),
        )
    } else when T == complex128 {
        lapack.zlarf_(
            &side_c,
            &m_int,
            &n_int,
            &V.data[V.offset],
            &incv,
            &tau_val,
            raw_data(C.data),
            &ldc,
            raw_data(work),
        )
    }
}

// ===================================================================================
// BLOCK HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying block Householder reflector
query_workspace_block_householder_reflector :: proc(
    $T: typeid,
    m: int,
    n: int,
    k: int,
    side: ReflectorSide,
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    ldwork := side == .Left ? n : m
    return ldwork * k
}

solve_apply_block_householder_reflector :: proc(
    side: ReflectorSide,
    trans: ReflectorTranspose,
    direction: ReflectorDirection,
    storage: ReflectorStorage,
    m: int,
    n: int,
    k: int,
    V: ^Matrix($T), // Matrix of Householder vectors
    T_matrix: ^Matrix(T), // Block reflector T matrix
    C: ^Matrix(T), // Matrix to transform
    work: []T, // Workspace array
) where is_float(T) || is_complex(T) {
    ldwork := side == .Left ? n : m
    assert(len(work) >= ldwork * k, "Work array too small")

    side_c := cast(u8)side
    trans_c := cast(u8)trans
    direct_c := cast(u8)direction
    storev_c := cast(u8)storage

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    k_int := Blas_Int(k)
    ldv := V.ld
    ldt := T_matrix.ld
    ldc := C.ld
    ldwork_int := Blas_Int(ldwork)

    when T == f32 {
        lapack.slarfb_(
            &side_c,
            &trans_c,
            &direct_c,
            &storev_c,
            &m_int,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(C.data),
            &ldc,
            raw_data(work),
            &ldwork_int,
        )
    } else when T == f64 {
        lapack.dlarfb_(
            &side_c,
            &trans_c,
            &direct_c,
            &storev_c,
            &m_int,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(C.data),
            &ldc,
            raw_data(work),
            &ldwork_int,
        )
    } else when T == complex64 {
        lapack.clarfb_(
            &side_c,
            &trans_c,
            &direct_c,
            &storev_c,
            &m_int,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(C.data),
            &ldc,
            raw_data(work),
            &ldwork_int,
        )
    } else when T == complex128 {
        lapack.zlarfb_(
            &side_c,
            &trans_c,
            &direct_c,
            &storev_c,
            &m_int,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(T_matrix.data),
            &ldt,
            raw_data(C.data),
            &ldc,
            raw_data(work),
            &ldwork_int,
        )
    }
}

// ===================================================================================
// HOUSEHOLDER REFLECTOR GENERATION
// ===================================================================================

v_generate_householder_reflector :: proc(X: ^Vector($T)) -> (tau: T) where is_float(T) || is_complex(T) {
    // Generate Householder reflector to zero out all but first element
    n := Blas_Int(X.size)
    incx := Blas_Int(X.incr)

    when T == f32 {
        lapack.slarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)
    } else when T == f64 {
        lapack.dlarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)
    } else when T == complex64 {
        lapack.clarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)
    } else when T == complex128 {
        lapack.zlarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)
    }

    return tau
}

// ===================================================================================
// TRIANGULAR BLOCK HOUSEHOLDER FACTOR FORMATION (LARFT)
// ===================================================================================

form_triangular_block_householder :: proc(
    direction: ReflectorDirection, // Forward or Backward
    storage: ReflectorStorage, // Column-wise or Row-wise storage of V
    n: int, // Order of the matrix V
    k: int, // Number of reflectors
    V: ^Matrix($T), // Householder vectors (n x k)
    tau: []T, // Scalar factors of reflectors (size k)
    T_matrix: ^Matrix(T), // Triangular factor T (k x k, output)
) where is_float(T) || is_complex(T) {
    assert(len(tau) >= k, "tau array too small")
    assert(T_matrix.rows >= Blas_Int(k) && T_matrix.cols >= Blas_Int(k), "T matrix too small")

    direct_c := cast(u8)direction
    storev_c := cast(u8)storage
    n_int := Blas_Int(n)
    k_int := Blas_Int(k)
    ldv := V.ld
    ldt := T_matrix.ld

    when T == f32 {
        lapack.slarft_(
            &direct_c,
            &storev_c,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(tau),
            raw_data(T_matrix.data),
            &ldt,
            1,
            1,
        )
    } else when T == f64 {
        lapack.dlarft_(
            &direct_c,
            &storev_c,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(tau),
            raw_data(T_matrix.data),
            &ldt,
            1,
            1,
        )
    } else when T == complex64 {
        lapack.clarft_(
            &direct_c,
            &storev_c,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(tau),
            raw_data(T_matrix.data),
            &ldt,
            1,
            1,
        )
    } else when T == complex128 {
        lapack.zlarft_(
            &direct_c,
            &storev_c,
            &n_int,
            &k_int,
            raw_data(V.data),
            &ldv,
            raw_data(tau),
            raw_data(T_matrix.data),
            &ldt,
            1,
            1,
        )
    }
}

// ===================================================================================
// ELEMENTARY HOUSEHOLDER REFLECTOR APPLICATION (LARFX)
// ===================================================================================

// Query workspace for elementary Householder reflector
query_workspace_elementary_householder :: proc(
    $T: typeid,
    m: int,
    n: int,
    side: ReflectorSide,
) -> (
    work_size: int,
) where is_float(T) ||
    is_complex(T) {
    // larfx requires workspace of size n (if side=Left) or m (if side=Right)
    return side == .Left ? n : m
}

apply_elementary_householder :: proc(
    side: ReflectorSide, // Left or Right
    m: int, // Number of rows of C
    n: int, // Number of columns of C
    V: []$T, // Householder vector (size m if side=Left, n if side=Right)
    tau: T, // Scalar factor tau
    C: ^Matrix(T), // Matrix to transform (m x n)
    work: []T, // Workspace (size n if side=Left, m if side=Right)
) where is_float(T) || is_complex(T) {
    work_size := side == .Left ? n : m
    assert(len(work) >= work_size, "Work array too small")

    v_size := side == .Left ? m : n
    assert(len(V) >= v_size, "V vector too small")

    side_c := cast(u8)side
    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    ldc := C.ld
    tau_val := tau

    when T == f32 {
        lapack.slarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
    } else when T == f64 {
        lapack.dlarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
    } else when T == complex64 {
        lapack.clarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
    } else when T == complex128 {
        lapack.zlarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
    }
}
