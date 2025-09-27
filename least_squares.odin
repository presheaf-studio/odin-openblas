package openblas

import lapack "./f77"

// Solve least squares problem: minimize ||A*x - B||_2 or ||A^T*x - B||_2
// Uses QR or LQ factorization (fastest, but requires full rank)
least_squares :: proc {
	least_squares_f32_c64,
	least_squares_f64_c128,
}

// Solve least squares using SVD (most robust, handles rank deficiency)
// minimize ||A*x - B||_2 using divide-and-conquer SVD
// Singular values S are always real ([]f32 for complex64, []f64 for complex128)
least_squares_svd :: proc {
	least_squares_svd_f32_c64,
	least_squares_svd_f64_c128,
}

// ===================================================================================
// LEAST SQUARES PROBLEMS
// Solve overdetermined or underdetermined systems in the least squares sense
// ===================================================================================
// Query workspace size for least squares (QR/LQ factorization - legacy gels algorithm)
query_workspace_least_squares_legacy :: proc(A: ^Matrix($T), B: ^Matrix(T), transpose: bool = false) -> (work_size: int, rwork_size: int, info: Info) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	when T == f32 || T == f64 {
		trans_c = transpose ? cstring("T") : cstring("N")
	} else {
		trans_c = transpose ? cstring("C") : cstring("N") // Conjugate transpose
	}

	lwork := Blas_Int(-1)
	rwork_size = 0 // Initialize rwork_size

	when T == f32 {
		work_query: f32
		lapack.sgels_(
			trans_c,
			&m,
			&n,
			&nrhs,
			nil,
			&lda, // A dummy
			nil,
			&ldb, // B dummy
			&work_query,
			&lwork,
			&info,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == f64 {
		work_query: f64
		lapack.dgels_(trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == complex64 {
		work_query: complex64
		lapack.cgels_(trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
		rwork_size = 0 // Complex types using QR/LQ don't need rwork
	} else when T == complex128 {
		work_query: complex128
		lapack.zgels_(trans_c, &m, &n, &nrhs, nil, &lda, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
		rwork_size = 0 // Complex types using QR/LQ don't need rwork
	}

	return work_size, rwork_size, info
}

least_squares_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with factorization)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	transpose: bool = false, // Solve A^T*x = B (or A^H*x = B for complex)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Verify workspace size
	assert(len(work) > 0, "work array must be provided")

	// Set transpose character based on type
	trans_c: cstring
	when T == f32 {
		trans_c = transpose ? cstring("T") : cstring("N")
	} else {
		trans_c = transpose ? cstring("C") : cstring("N") // C = conjugate transpose
	}

	lwork := Blas_Int(len(work))

	// Solve least squares
	when T == f32 {
		lapack.sgels_(trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.cgels_(trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

least_squares_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with factorization)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	transpose: bool = false, // Solve A^T*x = B (or A^H*x = B for complex)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Verify workspace size
	assert(len(work) > 0, "work array must be provided")

	// Set transpose character based on type
	trans_c: cstring
	when T == f64 {
		trans_c = transpose ? cstring("T") : cstring("N")
	} else {
		trans_c = transpose ? cstring("C") : cstring("N") // C = conjugate transpose
	}

	lwork := Blas_Int(len(work))

	// Solve least squares
	when T == f64 {
		lapack.dgels_(trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zgels_(trans_c, &m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}


// Query workspace size for SVD-based least squares (legacy gelsd algorithm)
query_workspace_least_squares_svd_legacy :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	s_size: int,
	info: Info,
) where T == f32 ||
	T == f64 ||
	T == complex64 ||
	T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	s_size = int(min_mn)
	iwork_size = int(3 * min_mn)

	// Dummy arrays for query
	dummy_s := [1]f64{}
	dummy_iwork := [1]Blas_Int{}

	lwork := Blas_Int(-1)
	rwork_size = 0 // Initialize rwork_size
	rcond := T(0) // Dummy rcond for query
	rank := Blas_Int(0) // Dummy rank

	when T == f32 {
		work_query: f32
		lapack.sgelsd_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda, // A dummy
			nil,
			&ldb, // B dummy
			cast(^f32)&dummy_s[0], // S dummy
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == f64 {
		work_query: f64
		lapack.dgelsd_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			&dummy_s[0], // S dummy
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		lrwork := Blas_Int(-1)
		lapack.cgelsd_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			cast(^f32)&dummy_s[0], // S dummy (always real)
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		lrwork := Blas_Int(-1)
		lapack.zgelsd_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			&dummy_s[0], // S dummy (always real)
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	}

	return work_size, rwork_size, iwork_size, s_size, info
}

least_squares_svd_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	S: []f32, // Singular values (pre-allocated, always f32)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	rcond: f32 = -1, // Singular value threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(3 * min_mn), "iwork array too small (minimum 3*min(m,n))")

	// For complex types, verify rwork is provided
	when T == complex64 {
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with SVD
	when T == f32 {
		lapack.sgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == complex64 {
		lrwork := Blas_Int(len(rwork))
		lapack.cgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)
	}

	return rank, info, info == 0
}

least_squares_svd_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	S: []f64, // Singular values (pre-allocated, always f64)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	rcond: f64 = -1, // Singular value threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(3 * min_mn), "iwork array too small (minimum 3*min(m,n))")

	// For complex types, verify rwork is provided
	when T == complex128 {
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with SVD
	when T == f64 {
		lapack.dgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == complex128 {
		lrwork := Blas_Int(len(rwork))
		lapack.zgelsd_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)
	}

	return rank, info, info == 0
}

// Query workspace size for SVD-based least squares (simple gelss algorithm)
query_workspace_least_squares_svd_simple :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int, s_size: int, info: Info) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	s_size = int(min_mn)
	rwork_size = 0 // Initialize rwork_size

	// Dummy arrays for query
	dummy_s := [1]f64{}

	lwork := Blas_Int(-1)
	rcond := T(0) // Dummy rcond for query
	rank := Blas_Int(0) // Dummy rank

	when T == f32 {
		work_query: f32
		lapack.sgelss_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda, // A dummy
			nil,
			&ldb, // B dummy
			cast(^f32)&dummy_s[0], // S dummy
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == f64 {
		work_query: f64
		lapack.dgelss_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			&dummy_s[0], // S dummy
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == complex64 {
		work_query: complex64
		dummy_rwork := [1]f32{}
		lapack.cgelss_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			cast(^f32)&dummy_s[0], // S dummy (always real)
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn) // gelss needs 5*min(m,n) rwork for complex
	} else when T == complex128 {
		work_query: complex128
		dummy_rwork := [1]f64{}
		lapack.zgelss_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda,
			nil,
			&ldb,
			&dummy_s[0], // S dummy
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(5 * min_mn) // gelss needs 5*min(m,n) rwork for complex
	}

	return work_size, rwork_size, s_size, info
}

// Solve least squares using SVD with explicit threshold control
// Uses standard SVD algorithm (slower than divide-and-conquer but more control)
least_squares_svd_simple :: proc {
	least_squares_svd_simple_f32_c64,
	least_squares_svd_simple_f64_c128,
}

least_squares_svd_simple_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	S: []f32, // Singular values (pre-allocated, always f32)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	rcond: f32 = -1, // Singular value threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	// For complex types, verify rwork is provided
	when T == complex64 {
		assert(len(rwork) >= int(5 * min_mn), "rwork array too small (minimum 5*min(m,n) for gelss)")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with SVD
	when T == f32 {
		lapack.sgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

least_squares_svd_simple_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	S: []f64, // Singular values (pre-allocated, always f64)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	rcond: f64 = -1, // Singular value threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	// For complex types, verify rwork is provided
	when T == complex128 {
		assert(len(rwork) >= int(5 * min_mn), "rwork array too small (minimum 5*min(m,n) for gelss)")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with SVD
	when T == f64 {
		lapack.dgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgelss_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(S), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

// Query workspace size for QR with column pivoting least squares (gelsy algorithm)
query_workspace_least_squares_qrp :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int, jpvt_size: int, info: Info) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	jpvt_size = int(n)
	rwork_size = 0 // Initialize rwork_size

	// Dummy arrays for query
	dummy_jpvt := [1]Blas_Int{}

	lwork := Blas_Int(-1)
	rcond := T(0) // Dummy rcond for query
	rank := Blas_Int(0) // Dummy rank

	when T == f32 {
		work_query: f32
		lapack.sgelsy_(
			&m,
			&n,
			&nrhs,
			nil,
			&lda, // A dummy
			nil,
			&ldb, // B dummy
			&dummy_jpvt[0],
			&rcond,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == f64 {
		work_query: f64
		lapack.dgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0 // Real types don't need rwork
	} else when T == complex64 {
		work_query: complex64
		dummy_rwork := [1]f32{}
		lapack.cgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(2 * n) // gelsy needs 2*n rwork for complex
	} else when T == complex128 {
		work_query: complex128
		dummy_rwork := [1]f64{}
		lapack.zgelsy_(&m, &n, &nrhs, nil, &lda, nil, &ldb, &dummy_jpvt[0], &rcond, &rank, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = int(2 * n) // gelsy needs 2*n rwork for complex
	}

	return work_size, rwork_size, jpvt_size, info
}

// Solve least squares using QR with column pivoting
// Good balance between speed and robustness for rank-deficient problems
least_squares_qrp :: proc {
	least_squares_qrp_f32_c64,
	least_squares_qrp_f64_c128,
}

least_squares_qrp_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with QR factorization)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	jpvt: []Blas_Int, // Column pivot indices (pre-allocated, zero-initialized)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	rcond: f32 = -1, // Rank determination threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Verify array sizes
	assert(len(jpvt) >= int(n), "jpvt array too small (minimum n)")
	assert(len(work) > 0, "work array must be provided")

	// For complex types, verify rwork is provided
	when T == complex64 {
		assert(len(rwork) >= int(2 * n), "rwork array too small (minimum 2*n for gelsy)")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with QR pivoting
	when T == f32 {
		lapack.sgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond_val, &rank, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}

least_squares_qrp_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with QR factorization)
	B: ^Matrix(T), // Right-hand side / solution matrix (overwritten with solution)
	jpvt: []Blas_Int, // Column pivot indices (pre-allocated, zero-initialized)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	rcond: f64 = -1, // Rank determination threshold (-1 = machine precision)
) -> (
	rank: Blas_Int,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Verify array sizes
	assert(len(jpvt) >= int(n), "jpvt array too small (minimum n)")
	assert(len(work) > 0, "work array must be provided")

	// For complex types, verify rwork is provided
	when T == complex128 {
		assert(len(rwork) >= int(2 * n), "rwork array too small (minimum 2*n for gelsy)")
	}

	rcond_val := rcond
	lwork := Blas_Int(len(work))

	// Solve least squares with QR pivoting
	when T == f64 {
		lapack.dgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond_val, &rank, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgelsy_(&m, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(jpvt), &rcond_val, &rank, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rank, info, info == 0
}
