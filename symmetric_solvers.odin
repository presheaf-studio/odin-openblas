package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC SYSTEM SOLVERS - NON-ALLOCATING API
// ============================================================================
// Pre-allocated workspace and result arrays

// Query workspace for symmetric system solver (SYSV)
query_workspace_solve_symmetric :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := matrix_region_to_cstring(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	}

	return work_size
}

// Query result array sizes for symmetric solver
query_result_sizes_solve_symmetric :: proc(n: int) -> (ipiv_size: int) {
	// Pivot array size is always n
	return n
}

// Solve symmetric system using Bunch-Kaufman pivoting
m_solve_symmetric :: proc(
	A: ^Matrix($T), // System matrix (modified on output)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dsysv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csysv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsysv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// HERMITIAN SYSTEM SOLVERS - NON-ALLOCATING API
// ============================================================================
// For complex matrices with Hermitian property

// Query workspace for Hermitian system solver (HESV)
query_workspace_solve_hermitian :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int, rwork_size: int) where is_complex(T) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := matrix_region_to_cstring(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == complex64 {
		work_query: complex64
		lapack.chesv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
		rwork_size = 0 // HESV doesn't need real workspace
	} else when T == complex128 {
		work_query: complex128
		lapack.zhesv_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
		rwork_size = 0 // HESV doesn't need real workspace
	}

	return work_size, rwork_size
}

// Solve Hermitian system using Bunch-Kaufman pivoting
m_solve_hermitian :: proc(
	A: ^Matrix($T), // System matrix (modified on output)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.chesv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zhesv_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// AASEN SYMMETRIC SOLVERS
// ============================================================================
// Uses Aasen's algorithm for symmetric indefinite matrices

// Query workspace for Aasen symmetric solver
query_workspace_solve_symmetric_aasen :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ldb := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_aa_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_aa_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_aa_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_aa_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	}

	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Solve using Aasen algorithm for symmetric indefinite matrices - non-allocating
m_solve_symmetric_aasen :: proc(
	A: ^Matrix($T), // In: matrix to factor, Out: factorized matrix
	B: ^Matrix(T), // In: right-hand side, Out: solution
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_aa_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dsysv_aa_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csysv_aa_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsysv_aa_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}


// ============================================================================
// 2-STAGE AASEN SYMMETRIC SOLVERS
// ============================================================================
// Two-stage Aasen algorithm for improved performance on large matrices

// Query workspace for 2-stage Aasen symmetric solver
query_workspace_solve_symmetric_aasen_2stage :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ltb := Blas_Int(4 * n) // Band matrix storage
	ldb := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_aa_2stage_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // tb
			&ltb,
			nil, // ipiv
			nil, // ipiv2
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_aa_2stage_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // tb
			&ltb,
			nil, // ipiv
			nil, // ipiv2
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_aa_2stage_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // tb
			&ltb,
			nil, // ipiv
			nil, // ipiv2
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_aa_2stage_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // tb
			&ltb,
			nil, // ipiv
			nil, // ipiv2
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	}

	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Solve using 2-stage Aasen algorithm for symmetric indefinite matrices - non-allocating
m_solve_symmetric_aasen_2stage :: proc(
	A: ^Matrix($T), // In: matrix to factor, Out: factorized matrix
	TB: ^Matrix(T), // Band matrix storage (4*n, nb)
	B: ^Matrix(T), // In: right-hand side, Out: solution
	ipiv: []Blas_Int, // Pre-allocated first stage pivot array (size n)
	ipiv2: []Blas_Int, // Pre-allocated second stage pivot array (size n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(TB.rows >= 4 * n, "Band matrix TB rows insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ltb := Blas_Int(TB.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_aa_2stage_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(TB.data), &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dsysv_aa_2stage_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(TB.data), &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csysv_aa_2stage_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(TB.data), &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsysv_aa_2stage_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(TB.data), &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// RK (BOUNDED BUNCH-KAUFMAN) SYMMETRIC SOLVERS
// ============================================================================
// Uses bounded Bunch-Kaufman pivoting with additional E factor

// Query workspace for RK symmetric solver
query_workspace_solve_symmetric_rk :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ldb := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_rk_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // e
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_rk_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // e
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_rk_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // e
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_rk_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // e
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	}

	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Solve using RK (bounded Bunch-Kaufman) algorithm - non-allocating
m_solve_symmetric_rk :: proc(
	A: ^Matrix($T), // In: matrix to factor, Out: factorized matrix
	E: []T, // E factor from RK factorization (size n)
	B: ^Matrix(T), // In: right-hand side, Out: solution
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(len(E) >= n, "E vector too small")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_rk_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dsysv_rk_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csysv_rk_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsysv_rk_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// ROOK PIVOTING SYMMETRIC SOLVERS
// ============================================================================
// Uses rook pivoting strategy for improved numerical stability

// Query workspace for Rook symmetric solver
query_workspace_solve_symmetric_rook :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ldb := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_rook_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_rook_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_rook_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_rook_(
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // ipiv
			nil, // b
			&ldb,
			&work_query,
			&lwork,
			&info,
			len(uplo_c),
		)
		work_size = int(real(work_query))
	}

	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Solve using Rook pivoting strategy - non-allocating
m_solve_symmetric_rook :: proc(
	A: ^Matrix($T), // In: matrix to factor, Out: factorized matrix
	B: ^Matrix(T), // In: right-hand side, Out: solution
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_rook_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dsysv_rook_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csysv_rook_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsysv_rook_(uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// EXPERT SYMMETRIC SOLVERS
// ============================================================================
// Expert interface with condition estimation and error bounds

// Query workspace for expert symmetric solver
query_workspace_solve_symmetric_expert :: proc($T: typeid, n: int, nrhs: int, fact := FactorizationOption.Factor, uplo := MatrixRegion.Upper) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1, 1
	}

	fact_c := factorization_to_cstring(fact)

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ldaf := n_int
	ldb := n_int
	ldx := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info
	rcond := 0.0

	when T == f32 {
		work_query: f32
		lapack.ssysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			nil, // ferr
			nil, // berr
			&work_query,
			&lwork,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
		work_size = int(work_query)
		rwork_size = 1 // No real workspace for real types
	} else when T == f64 {
		work_query: f64
		lapack.dsysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			nil, // ferr
			nil, // berr
			&work_query,
			&lwork,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
		work_size = int(work_query)
		rwork_size = 1 // No real workspace for real types
	} else when T == complex64 {
		work_query: complex64
		lapack.csysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			nil, // ferr
			nil, // berr
			&work_query,
			&lwork,
			nil, // rwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
		work_size = int(real(work_query))
		rwork_size = n // Real workspace for complex types
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			nil, // ferr
			nil, // berr
			&work_query,
			&lwork,
			nil, // rwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
		work_size = int(real(work_query))
		rwork_size = n // Real workspace for complex types
	}

	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	return work_size, rwork_size
}

// Expert solve with condition estimation and error bounds - non-allocating (f32/complex64)
m_solve_symmetric_expert_f32_c64 :: proc(
	A: ^Matrix($T), // Original matrix (if fact == FACTORIZE)
	AF: ^Matrix(T), // Factorized matrix
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	ferr: []f32, // Forward error bounds (size nrhs)
	berr: []f32, // Backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Real workspace for complex64 (size n)
	fact := FactorizationOption.Factor,
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f32,
	info: Info,
	ok: bool, // Condition number estimate
) where T == f32 || T == complex64 {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(AF.rows >= n, "Matrix AF rows insufficient")
	assert(AF.cols >= n, "Matrix AF cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(X.rows >= n, "Matrix X rows insufficient")
	assert(X.cols >= nrhs, "Matrix X cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")

	fact_c := factorization_to_cstring(fact)

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldaf := Blas_Int(AF.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			&lwork,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
	} else when T == complex64 {
		assert(rwork != nil && len(rwork) >= n, "Real workspace required for complex types")
		lapack.csysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
		)
	}

	ok = info == 0
	return rcond, info, ok
}

// Expert solve with condition estimation and error bounds - non-allocating (f64/complex128)
m_solve_symmetric_expert_f64_c128 :: proc(
	A: ^Matrix($T), // Original matrix (if fact == FACTORIZE)
	AF: ^Matrix(T), // Factorized matrix
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	ferr: []f64, // Forward error bounds (size nrhs)
	berr: []f64, // Backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Real workspace for complex128 (size n)
	fact := FactorizationOption.Factor,
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f64,
	info: Info,
	ok: bool, // Condition number estimate
) where T == f64 || T == complex128 {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(AF.rows >= n, "Matrix AF rows insufficient")
	assert(AF.cols >= n, "Matrix AF cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(X.rows >= n, "Matrix X rows insufficient")
	assert(X.cols >= nrhs, "Matrix X cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")

	fact_c := factorization_to_cstring(fact)

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldaf := Blas_Int(AF.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	lwork := Blas_Int(len(work))

	when T == f64 {
		lapack.dsysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			&lwork,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
		)
	} else when T == complex128 {
		assert(rwork != nil && len(rwork) >= n, "Real workspace required for complex types")
		lapack.zsysvx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
		)
	}

	ok = info == 0
	return rcond, info, ok
}

m_solve_symmetric_expert :: proc {
	m_solve_symmetric_expert_f32_c64,
	m_solve_symmetric_expert_f64_c128,
}

// ============================================================================
// EXTENDED EXPERT SYMMETRIC SOLVERS
// ============================================================================
// Most advanced interface with equilibration and comprehensive error analysis

// Query workspace for extended expert symmetric solver
query_workspace_solve_symmetric_extended_expert :: proc(
	$T: typeid,
	n: int,
	nrhs: int,
	fact := FactorizationOption.Factor,
	uplo := MatrixRegion.Upper,
	n_err_bnds: int = 3,
	nparams: int = 0,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
) where is_float(T) ||
	is_complex(T) {
	if n <= 0 || nrhs <= 0 {
		return 1, 1, 1
	}

	fact_c := factorization_to_cstring(fact)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := n_int
	ldaf := n_int
	ldb := n_int
	ldx := n_int
	lwork := Blas_Int(QUERY_WORKSPACE)
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info: Info
	rcond := 0.0
	rpvgrw := 0.0

	when T == f32 {
		work_query: f32
		lapack.ssysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // equed
			nil, // s
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			&rpvgrw,
			nil, // berr
			&n_err_bnds_int,
			nil, // err_bounds_norm
			nil, // err_bounds_comp
			&nparams_int,
			nil, // params
			&work_query,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
		work_size = int(work_query)
		rwork_size = 1 // No real workspace for real types
		iwork_size = n // Integer workspace for real types
	} else when T == f64 {
		work_query: f64
		lapack.dsysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // equed
			nil, // s
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			&rpvgrw,
			nil, // berr
			&n_err_bnds_int,
			nil, // err_bounds_norm
			nil, // err_bounds_comp
			&nparams_int,
			nil, // params
			&work_query,
			nil, // iwork
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
		work_size = int(work_query)
		rwork_size = 1 // No real workspace for real types
		iwork_size = n // Integer workspace for real types
	} else when T == complex64 {
		work_query: complex64
		lapack.csysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // equed
			nil, // s
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			&rpvgrw,
			nil, // berr
			&n_err_bnds_int,
			nil, // err_bounds_norm
			nil, // err_bounds_comp
			&nparams_int,
			nil, // params
			&work_query,
			nil, // rwork
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
		work_size = int(real(work_query))
		rwork_size = 2 * n // Real workspace for complex types
		iwork_size = 1 // No integer workspace for complex types
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			nil, // a
			&lda,
			nil, // af
			&ldaf,
			nil, // ipiv
			nil, // equed
			nil, // s
			nil, // b
			&ldb,
			nil, // x
			&ldx,
			&rcond,
			&rpvgrw,
			nil, // berr
			&n_err_bnds_int,
			nil, // err_bounds_norm
			nil, // err_bounds_comp
			&nparams_int,
			nil, // params
			&work_query,
			nil, // rwork
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
		work_size = int(real(work_query))
		rwork_size = 2 * n // Real workspace for complex types
		iwork_size = 1 // No integer workspace for complex types
	}

	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	return work_size, rwork_size, iwork_size
}

// Extended expert solve with equilibration and comprehensive error analysis - non-allocating (f32/complex64)
m_solve_symmetric_extended_expert_f32_c64 :: proc(
	A: ^Matrix($T), // Original matrix (if fact == FACTORIZE)
	AF: ^Matrix(T), // Factorized matrix
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	equed: ^byte, // Equilibration state (output)
	s: []f32, // Scale factors (size n)
	berr: []f32, // Backward error bounds (size nrhs)
	err_bounds_norm: []f32, // Normwise bounds (nrhs * n_err_bnds)
	err_bounds_comp: []f32, // Componentwise bounds (nrhs * n_err_bnds)
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Real workspace for complex64 (size rwork_size)
	iwork: []Blas_Int = nil, // Integer workspace for f32 (size n)
	params: []f32 = nil, // Parameters array
	fact := FactorizationOption.Factor,
	uplo := MatrixRegion.Upper,
	n_err_bnds: int = 3,
	nparams: int = 0,
) -> (
	rcond: f32,
	rpvgrw: f32,
	info: Info,
	ok: bool, // Condition number estimate// Reciprocal pivot growth factor
) where T == f32 || T == complex64 {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(AF.rows >= n, "Matrix AF rows insufficient")
	assert(AF.cols >= n, "Matrix AF cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(X.rows >= n, "Matrix X rows insufficient")
	assert(X.cols >= nrhs, "Matrix X cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(s) >= n, "Scale factor array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bounds_norm) >= nrhs * n_err_bnds, "Normwise error bounds array too small")
	assert(len(err_bounds_comp) >= nrhs * n_err_bnds, "Componentwise error bounds array too small")

	fact_c := factorization_to_cstring(fact)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldaf := Blas_Int(AF.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)

	when T == f32 {
		assert(iwork != nil && len(iwork) >= n, "Integer workspace required for f32")
		lapack.ssysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			equed,
			raw_data(s),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
	} else when T == complex64 {
		assert(rwork != nil, "Real workspace required for complex64")
		lapack.csysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			equed,
			raw_data(s),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
	}

	ok = info == 0
	return rcond, rpvgrw, info, ok
}

// Extended expert solve with equilibration and comprehensive error analysis - non-allocating (f64/complex128)
m_solve_symmetric_extended_expert_f64_c128 :: proc(
	A: ^Matrix($T), // Original matrix (if fact == FACTORIZE)
	AF: ^Matrix(T), // Factorized matrix
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ipiv: []Blas_Int, // Pre-allocated pivot array (size n)
	equed: ^byte, // Equilibration state (output)
	s: []f64, // Scale factors (size n)
	berr: []f64, // Backward error bounds (size nrhs)
	err_bounds_norm: []f64, // Normwise bounds (nrhs * n_err_bnds)
	err_bounds_comp: []f64, // Componentwise bounds (nrhs * n_err_bnds)
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Real workspace for complex128 (size rwork_size)
	iwork: []Blas_Int = nil, // Integer workspace for f64 (size n)
	params: []f64 = nil, // Parameters array
	fact := FactorizationOption.Factor,
	uplo := MatrixRegion.Upper,
	n_err_bnds: int = 3,
	nparams: int = 0,
) -> (
	rcond: f64,
	rpvgrw: f64,
	info: Info,
	ok: bool, // Condition number estimate// Reciprocal pivot growth factor
) where T == f64 || T == complex128 {
	n := A.cols
	nrhs := B.cols
	assert(A.rows >= n, "Matrix A rows insufficient")
	assert(A.cols >= n, "Matrix A cols insufficient")
	assert(AF.rows >= n, "Matrix AF rows insufficient")
	assert(AF.cols >= n, "Matrix AF cols insufficient")
	assert(B.rows >= n, "Matrix B rows insufficient")
	assert(B.cols >= nrhs, "Matrix B cols insufficient")
	assert(X.rows >= n, "Matrix X rows insufficient")
	assert(X.cols >= nrhs, "Matrix X cols insufficient")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(s) >= n, "Scale factor array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bounds_norm) >= nrhs * n_err_bnds, "Normwise error bounds array too small")
	assert(len(err_bounds_comp) >= nrhs * n_err_bnds, "Componentwise error bounds array too small")

	fact_c := factorization_to_cstring(fact)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(A.ld)
	ldaf := Blas_Int(AF.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)

	when T == f64 {
		assert(iwork != nil && len(iwork) >= n, "Integer workspace required for f64")
		lapack.dsysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			equed,
			raw_data(s),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
	} else when T == complex128 {
		assert(rwork != nil, "Real workspace required for complex128")
		lapack.zsysvxx_(
			fact_c,
			uplo_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			equed,
			raw_data(s),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1, // equed length
		)
	}

	ok = info == 0
	return rcond, rpvgrw, info, ok
}

m_solve_symmetric_extended_expert :: proc {
	m_solve_symmetric_extended_expert_f32_c64,
	m_solve_symmetric_extended_expert_f64_c128,
}
