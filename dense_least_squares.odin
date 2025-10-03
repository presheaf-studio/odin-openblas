package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// LEAST SQUARES PROBLEMS (GELS family)
// ===================================================================================

// Overloaded procedures for different algorithms
least_squares :: proc {
	least_squares_real,
	least_squares_complex,
}

least_squares_svd :: proc {
	least_squares_svd_real,
	least_squares_svd_complex,
}

least_squares_dc :: proc {
	least_squares_dc_real,
	least_squares_dc_complex,
}

least_squares_qr :: proc {
	least_squares_qr_real,
	least_squares_qr_complex,
}

least_squares_tall_skinny :: proc {
	least_squares_tall_skinny_real,
	least_squares_tall_skinny_complex,
}

// Query workspace size for basic least squares (GELS)
query_workspace_least_squares :: proc(A: ^Matrix($T), B: ^Matrix(T), trans: TransposeMode = .None) -> (work_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	trans_c := cast(u8)trans
	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		lapack.sgels_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		info: Info
		lapack.dgels_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		info: Info
		lapack.cgels_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		info: Info
		lapack.zgels_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Basic least squares solver using QR or LQ factorization
// Solves overdetermined or underdetermined systems: min ||A*X - B||
least_squares_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with factorization)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	work: []T, // Workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(work) > 0, "work array must be provided")

	// For overdetermined systems (m >= n): B must have at least m rows
	// For underdetermined systems (m < n): B must have at least n rows
	min_rows := trans == .None ? max(m, n) : max(m, n)
	assert(B.rows >= min_rows, "B matrix has insufficient rows for least squares solution")

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgels_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgels_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

least_squares_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten with factorization)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	work: []Cmplx, // Workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(work) > 0, "work array must be provided")

	min_rows := trans == .None ? max(m, n) : max(m, n)
	assert(B.rows >= min_rows, "B matrix has insufficient rows for least squares solution")

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgels_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zgels_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SVD-BASED LEAST SQUARES (GELSS family)
// ===================================================================================

// Query workspace size for SVD-based least squares
query_workspace_least_squares_svd :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	lwork: Blas_Int = QUERY_WORKSPACE
	min_mn := min(m, n)

	when T == f32 {
		work_query: f32
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_s := [1]f32{}
		lapack.sgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == f64 {
		work_query: f64
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_s := [1]f64{}
		lapack.dgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == complex64 {
		work_query: complex64
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_s := [1]f32{}
		dummy_rwork := [1]f32{}
		lapack.cgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn)
	} else when T == complex128 {
		work_query: complex128
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_s := [1]f64{}
		dummy_rwork := [1]f64{}
		lapack.zgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn)
	}

	return work_size, rwork_size
}

// SVD-based least squares with automatic rank determination
least_squares_svd_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	S: []T, // Singular values (pre-allocated, size min(m,n))
	work: []T, // Workspace (pre-allocated)
	rcond: T, // Reciprocal condition number threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, &info)
	}

	return rank, info, info == 0
}

least_squares_svd_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	S: []$Real, // Singular values (pre-allocated, size min(m,n))
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	rcond: Real, // Reciprocal condition number threshold
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(rwork) >= int(5 * min_mn), "rwork array too small")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

// ===================================================================================
// DIVIDE-AND-CONQUER LEAST SQUARES (GELSD family)
// ===================================================================================

// Query workspace size for divide-and-conquer least squares
query_workspace_least_squares_dc :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	min_mn := min(m, n)

	// Conservative estimates based on LAPACK documentation
	when T == f32 || T == f64 {
		work_size = int(12 * min_mn + 2 * min_mn * nrhs)
		if m >= n {
			work_size = max(work_size, int(12 * min_mn + 2 * min_mn * nrhs))
		} else {
			work_size = max(work_size, int(12 * min_mn + 2 * min_mn * nrhs))
		}
		rwork_size = 0
		iwork_size = int(3 * min_mn * (3 + 2 * int(log2(f64(min_mn) / 2))))
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * min_mn + min_mn * nrhs)
		rwork_size = int(10 * min_mn + 2 * min_mn * nrhs + 8 * min_mn)
		iwork_size = int(3 * min_mn * (3 + 2 * int(log2(f64(min_mn) / 2))))
	}

	return work_size, rwork_size, iwork_size
}

// Divide-and-conquer least squares (faster than SVD for large matrices)
least_squares_dc_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	S: []T, // Singular values (pre-allocated, size min(m,n))
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	rcond: T, // Reciprocal condition number threshold (-1 for machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) > 0, "iwork array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(iwork), &info)
	}

	return rank, info, info == 0
}

least_squares_dc_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	S: []$Real, // Singular values (pre-allocated, size min(m,n))
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	rcond: Real, // Reciprocal condition number threshold
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(rwork) > 0, "rwork array must be provided")
	assert(len(iwork) > 0, "iwork array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	}

	return rank, info, info == 0
}

// ===================================================================================
// QR WITH PIVOTING LEAST SQUARES (GELSY family)
// ===================================================================================

// Query workspace size for QR pivoting least squares
query_workspace_least_squares_qr :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_jpvt := [1]Blas_Int{}
		lapack.sgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == f64 {
		work_query: f64
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_jpvt := [1]Blas_Int{}
		lapack.dgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == complex64 {
		work_query: complex64
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_jpvt := [1]Blas_Int{}
		dummy_rwork := [1]f32{}
		lapack.cgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(2 * n)
	} else when T == complex128 {
		work_query: complex128
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_jpvt := [1]Blas_Int{}
		dummy_rwork := [1]f64{}
		lapack.zgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(2 * n)
	}

	return work_size, rwork_size
}

// QR with pivoting least squares (good rank detection)
least_squares_qr_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []T, // Workspace (pre-allocated)
	rcond: T, // Reciprocal condition number threshold (-1 for machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(jpvt) >= int(n), "jpvt array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	// Initialize pivot array to zero (LAPACK will determine optimal pivoting)
	for i in 0 ..< int(n) {
		jpvt[i] = 0
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, &info)
	}

	return rank, info, info == 0
}

least_squares_qr_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []$Real, // Real workspace (pre-allocated)
	rcond: Real, // Reciprocal condition number threshold
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(jpvt) >= int(n), "jpvt array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(rwork) >= int(2 * n), "rwork array too small")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	// Initialize pivot array to zero
	for i in 0 ..< int(n) {
		jpvt[i] = 0
	}

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

// ===================================================================================
// TALL/SKINNY LEAST SQUARES (GETSLS family)
// ===================================================================================

// Query workspace size for tall/skinny least squares
query_workspace_least_squares_tall_skinny :: proc(A: ^Matrix($T), B: ^Matrix(T), trans: TransposeMode = .None) -> (work_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	trans_c := cast(u8)trans
	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		lapack.sgetsls_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		info: Info
		lapack.dgetsls_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		info: Info
		lapack.cgetsls_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		info: Info
		lapack.zgetsls_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Optimized least squares for tall/skinny matrices (m >> n)
least_squares_tall_skinny_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	work: []T, // Workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(work) > 0, "work array must be provided")

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgetsls_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgetsls_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

least_squares_tall_skinny_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	work: []Cmplx, // Workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(work) > 0, "work array must be provided")

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgetsls_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zgetsls_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TALL-SKINNY QR LEAST SQUARES (GELST family)
// ===================================================================================

least_squares_tall_skinny_qr :: proc {
	least_squares_tall_skinny_qr_real,
	least_squares_tall_skinny_qr_complex,
}

// Query workspace size for GELST
query_workspace_least_squares_tall_skinny_qr :: proc(A: ^Matrix($T), B: ^Matrix(T), trans: TransposeMode = .None) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	trans_c := cast(u8)trans
	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_jpvt := [1]Blas_Int{}
		lapack.sgelst_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == f64 {
		work_query: f64
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_jpvt := [1]Blas_Int{}
		lapack.dgelst_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == complex64 {
		work_query: complex64
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_jpvt := [1]Blas_Int{}
		dummy_rwork := [1]f32{}
		lapack.cgelst_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(n)
	} else when T == complex128 {
		work_query: complex128
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_jpvt := [1]Blas_Int{}
		dummy_rwork := [1]f64{}
		lapack.zgelst_(&trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(n)
	}

	return work_size, rwork_size
}

// GELST - Least squares using complete orthogonal factorization (most robust for rank-deficient)
least_squares_tall_skinny_qr_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []T, // Workspace (pre-allocated)
	rcond: T, // Reciprocal condition number threshold
	trans: TransposeMode = .None,
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(jpvt) >= int(n), "jpvt array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	// Initialize pivot array to zero
	for i in 0 ..< int(n) {
		jpvt[i] = 0
	}

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgelst_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgelst_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, &info)
	}

	return rank, info, info == 0
}

least_squares_tall_skinny_qr_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	jpvt: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []$Real, // Real workspace (pre-allocated)
	rcond: Real, // Reciprocal condition number threshold
	trans: TransposeMode = .None,
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(len(jpvt) >= int(n), "jpvt array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(rwork) >= int(n), "rwork array too small")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	// Initialize pivot array to zero
	for i in 0 ..< int(n) {
		jpvt[i] = 0
	}

	trans_c := cast(u8)trans
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgelst_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgelst_(&trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

// ===================================================================================
// EQUALITY-CONSTRAINED LEAST SQUARES (GGLSE family)
// ===================================================================================

least_squares_equality_constrained :: proc {
	least_squares_equality_constrained_real,
	least_squares_equality_constrained_complex,
}

// Query workspace size for GGLSE
query_workspace_least_squares_equality_constrained :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	p := B.rows

	lda := A.ld
	ldb := B.ld

	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		lapack.sgglse_(&m, &n, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		info: Info
		lapack.dgglse_(&m, &n, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		info: Info
		lapack.cgglse_(&m, &n, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		info: Info
		lapack.zgglse_(&m, &n, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// GGLSE - Solve linear equality-constrained least squares problem:
//   min ||c - A*x||_2   subject to   B*x = d
// where A is M-by-N, B is P-by-N, c is M-vector, d is P-vector
least_squares_equality_constrained_real :: proc(
	A: ^Matrix($T), // M-by-N matrix (overwritten)
	B: ^Matrix(T), // P-by-N constraint matrix (overwritten)
	c: []T, // M-vector right-hand side (overwritten with residual)
	d: []T, // P-vector constraint (overwritten)
	x: []T, // N-vector solution (output)
	work: []T, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	assert(B.cols == n, "B matrix must have same number of columns as A")
	assert(len(c) >= int(m), "c array too small")
	assert(len(d) >= int(p), "d array too small")
	assert(len(x) >= int(n), "x array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(int(p) <= int(n) && int(n) <= int(m + p), "Invalid dimensions for equality-constrained least squares")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgglse_(&m, &n, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(c), raw_data(d), raw_data(x), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgglse_(&m, &n, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(c), raw_data(d), raw_data(x), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

least_squares_equality_constrained_complex :: proc(
	A: ^Matrix($Cmplx), // M-by-N matrix (overwritten)
	B: ^Matrix(Cmplx), // P-by-N constraint matrix (overwritten)
	c: []Cmplx, // M-vector right-hand side (overwritten with residual)
	d: []Cmplx, // P-vector constraint (overwritten)
	x: []Cmplx, // N-vector solution (output)
	work: []Cmplx, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	assert(B.cols == n, "B matrix must have same number of columns as A")
	assert(len(c) >= int(m), "c array too small")
	assert(len(d) >= int(p), "d array too small")
	assert(len(x) >= int(n), "x array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(int(p) <= int(n) && int(n) <= int(m + p), "Invalid dimensions for equality-constrained least squares")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgglse_(&m, &n, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(c), raw_data(d), raw_data(x), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zgglse_(&m, &n, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(c), raw_data(d), raw_data(x), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// GAUSS-MARKOV LINEAR MODEL (GGGLM family)
// ===================================================================================

gauss_markov_linear_model :: proc {
	gauss_markov_linear_model_real,
	gauss_markov_linear_model_complex,
}

// Query workspace size for GGGLM
query_workspace_gauss_markov_linear_model :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.rows
	m := A.cols
	p := B.cols

	lda := A.ld
	ldb := B.ld

	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		lapack.sggglm_(&n, &m, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		info: Info
		lapack.dggglm_(&n, &m, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		info: Info
		lapack.cggglm_(&n, &m, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		info: Info
		lapack.zggglm_(&n, &m, &p, nil, &lda, nil, &ldb, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// GGGLM - Solve general Gauss-Markov linear model problem:
//   min ||y||_2   subject to   d = A*x + B*y
// where A is N-by-M, B is N-by-P, d is N-vector
gauss_markov_linear_model_real :: proc(
	A: ^Matrix($T), // N-by-M matrix (overwritten)
	B: ^Matrix(T), // N-by-P matrix (overwritten)
	d: []T, // N-vector (overwritten)
	x: []T, // M-vector solution (output)
	y: []T, // P-vector solution (output)
	work: []T, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	m := A.cols
	p := B.cols
	lda := A.ld
	ldb := B.ld

	assert(B.rows == n, "B matrix must have same number of rows as A")
	assert(len(d) >= int(n), "d array too small")
	assert(len(x) >= int(m), "x array too small")
	assert(len(y) >= int(p), "y array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(int(n) <= int(m) && int(m) <= int(n + p), "Invalid dimensions for Gauss-Markov linear model")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sggglm_(&n, &m, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(d), raw_data(x), raw_data(y), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dggglm_(&n, &m, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(d), raw_data(x), raw_data(y), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

gauss_markov_linear_model_complex :: proc(
	A: ^Matrix($Cmplx), // N-by-M matrix (overwritten)
	B: ^Matrix(Cmplx), // N-by-P matrix (overwritten)
	d: []Cmplx, // N-vector (overwritten)
	x: []Cmplx, // M-vector solution (output)
	y: []Cmplx, // P-vector solution (output)
	work: []Cmplx, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	m := A.cols
	p := B.cols
	lda := A.ld
	ldb := B.ld

	assert(B.rows == n, "B matrix must have same number of rows as A")
	assert(len(d) >= int(n), "d array too small")
	assert(len(x) >= int(m), "x array too small")
	assert(len(y) >= int(p), "y array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(int(n) <= int(m) && int(m) <= int(n + p), "Invalid dimensions for Gauss-Markov linear model")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cggglm_(&n, &m, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(d), raw_data(x), raw_data(y), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zggglm_(&n, &m, &p, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(d), raw_data(x), raw_data(y), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// QR WITH PIVOTING LEAST SQUARES (GELSY family)
// ===================================================================================
// NOTE: The existing least_squares_qr functions already implement GELSY

// ===================================================================================
// SIMPLE SVD LEAST SQUARES (GELSS family)
// ===================================================================================
// Alternative SVD algorithm to the divide-and-conquer version (GELSD)

least_squares_svd_simple :: proc {
	least_squares_svd_simple_real,
	least_squares_svd_simple_complex,
}

// Query workspace size for simple SVD least squares (GELSS)
query_workspace_least_squares_svd_simple :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	lwork: Blas_Int = QUERY_WORKSPACE

	when T == f32 {
		work_query: f32
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_s := [1]f32{}
		lapack.sgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == f64 {
		work_query: f64
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_s := [1]f64{}
		lapack.dgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0
	} else when T == complex64 {
		work_query: complex64
		info: Info
		rank: Blas_Int
		rcond: f32 = -1
		dummy_s := [1]f32{}
		dummy_rwork := [1]f32{}
		lapack.cgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn)
	} else when T == complex128 {
		work_query: complex128
		info: Info
		rank: Blas_Int
		rcond: f64 = -1
		dummy_s := [1]f64{}
		dummy_rwork := [1]f64{}
		lapack.zgelss_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_s[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn)
	}

	return work_size, rwork_size
}

// Simple SVD least squares (GELSS algorithm - more control than GELSD)
least_squares_svd_simple_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	S: []T, // Singular values (pre-allocated, size min(m,n))
	work: []T, // Workspace (pre-allocated)
	rcond: T, // Reciprocal condition number threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_float(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, &info)
	}

	return rank, info, info == 0
}

least_squares_svd_simple_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	S: []$Real, // Singular values (pre-allocated, size min(m,n))
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	rcond: Real, // Reciprocal condition number threshold
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool, // Effective rank of matrix
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(rwork) >= int(5 * min_mn), "rwork array too small")
	assert(B.rows >= max(m, n), "B matrix has insufficient rows")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}
