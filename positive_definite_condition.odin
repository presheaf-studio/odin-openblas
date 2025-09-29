package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE MATRIX CONDITION NUMBER ESTIMATION
// ===================================================================================

// Condition number estimation proc group
m_condition_positive_definite :: proc {
	m_condition_positive_definite_f32_c64,
	m_condition_positive_definite_f64_c128,
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for condition number estimation
query_workspace_condition_positive_definite :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		return 3 * n, n, 0 // Real types need work and iwork
	} else when is_complex(T) {
		return 2 * n, 0, n // Complex types need work and rwork, no iwork
	}
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Estimate condition number of positive definite matrix (f32/complex64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f32_c64 :: proc(
	A: ^Matrix($T), // Factored matrix from Cholesky
	anorm: f32, // Norm of original matrix (before factorization)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f32 (nil for complex64)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rcond: f32,
	info: Info,
	ok: bool, // Reciprocal condition number
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.spocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.cpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// Estimate condition number of positive definite matrix (f64/complex128)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f64_c128 :: proc(
	A: ^Matrix($T), // Factored matrix from Cholesky
	anorm: f64, // Norm of original matrix (before factorization)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f64 (nil for complex128)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rcond: f64,
	info: Info,
	ok: bool, // Reciprocal condition number
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.dpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.zpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}
