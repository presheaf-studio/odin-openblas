package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE CHOLESKY FACTORIZATION
// ===================================================================================

// Standard Cholesky factorization proc group
m_cholesky :: proc {
	m_cholesky_f32_c64,
	m_cholesky_f64_c128,
}

// Recursive Cholesky factorization proc group
m_cholesky_recursive :: proc {
	m_cholesky_recursive_f32_c64,
	m_cholesky_recursive_f64_c128,
}

// Banded Cholesky factorization proc group
m_cholesky_factor_banded :: proc {
	m_cholesky_factor_banded_f32_c64,
	m_cholesky_factor_banded_f64_c128,
}

// Triangular solve using Cholesky factorization proc group
m_cholesky_solve_banded :: proc {
	m_cholesky_solve_banded_f32_c64,
	m_cholesky_solve_banded_f64_c128,
}

// Banded condition number estimation proc group
m_cholesky_condition_banded :: proc {
	m_cholesky_condition_banded_f32_c64,
	m_cholesky_condition_banded_f64_c128,
}

// Banded equilibration proc group
m_cholesky_equilibrate_banded :: proc {
	m_cholesky_equilibrate_banded_f32_c64,
	m_cholesky_equilibrate_banded_f64_c128,
}

// Banded iterative refinement proc group
m_cholesky_refine_banded :: proc {
	m_cholesky_refine_banded_f32_c64,
	m_cholesky_refine_banded_f64_c128,
}

// ===================================================================================
// CHOLESKY FACTORIZATION RESULT
// ===================================================================================

// Cholesky factorization result
CholeskyFactorization :: struct($T: typeid) {
	L:                    Matrix(T), // Lower triangular factor (or U if upper)
	uplo:                 MatrixRegion, // Upper or Lower triangular storage
	is_positive_definite: bool, // True if factorization succeeded
	first_non_pd_index:   int, // Index of first non-positive diagonal (if failed)
}

// ===================================================================================
// STANDARD CHOLESKY FACTORIZATION (POTRF)
// ===================================================================================

// Standard Cholesky factorization (f32/complex64)
// Computes the Cholesky factorization A = L*L^T or A = U^T*U (A = L*L^H or A = U^H*U for complex)
m_cholesky_f32_c64 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex64 {
		lapack.cpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// Standard Cholesky factorization (f64/complex128)
// Computes the Cholesky factorization A = L*L^T or A = U^T*U (A = L*L^H or A = U^H*U for complex)
m_cholesky_f64_c128 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f64 {
		lapack.dpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex128 {
		lapack.zpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RECURSIVE CHOLESKY FACTORIZATION (POTRF2)
// ===================================================================================

// Recursive Cholesky factorization (f32/complex64)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_f32_c64 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spotrf2_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex64 {
		lapack.cpotrf2_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// Recursive Cholesky factorization (f64/complex128)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_f64_c128 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := A.rows
	lda := A.ld
	uplo_c := cast(u8)uplo

	when T == f64 {
		lapack.dpotrf2_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex128 {
		lapack.zpotrf2_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED CHOLESKY FACTORIZATION (PBTRF)
// ===================================================================================

// Cholesky factorization for positive definite banded matrix (f32/complex64)
// Computes L or U such that A = L*L^T or A = U^T*U (A = L*L^H or A = U^H*U for complex)
m_cholesky_factor_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper, // Upper or lower triangular storage
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f32 {
		lapack.spbtrf_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info)
	} else when T == complex64 {
		lapack.cpbtrf_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// Cholesky factorization for positive definite banded matrix (f64/complex128)
// Computes L or U such that A = L*L^T or A = U^T*U (A = L*L^H or A = U^H*U for complex)
m_cholesky_factor_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper, // Upper or lower triangular storage
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f64 {
		lapack.dpbtrf_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info)
	} else when T == complex128 {
		lapack.zpbtrf_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED TRIANGULAR SOLVE (PBTRS)
// ===================================================================================

// Solve system using Cholesky factorization (f32/complex64)
// Solves A*X = B using factorization from SPBTRF/CPBTRF
m_cholesky_solve_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Factorized matrix from SPBTRF/CPBTRF
	B: ^Matrix(T), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper, // Upper or lower triangular storage
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(AB.data) > 0 && len(B.data) > 0, "Matrices cannot be empty")
	assert(AB.rows == AB.cols, "AB must be square")
	assert(B.rows == AB.rows, "System dimensions must be consistent")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	when T == f32 {
		lapack.spbtrs_(&uplo_c, &n, &kd_val, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cpbtrs_(&uplo_c, &n, &kd_val, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve system using Cholesky factorization (f64/complex128)
// Solves A*X = B using factorization from DPBTRF/ZPBTRF
m_cholesky_solve_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Factorized matrix from DPBTRF/ZPBTRF
	B: ^Matrix(T), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper, // Upper or lower triangular storage
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(AB.data) > 0 && len(B.data) > 0, "Matrices cannot be empty")
	assert(AB.rows == AB.cols, "AB must be square")
	assert(B.rows == AB.rows, "System dimensions must be consistent")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	when T == f64 {
		lapack.dpbtrs_(&uplo_c, &n, &kd_val, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpbtrs_(&uplo_c, &n, &kd_val, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED CONDITION NUMBER ESTIMATION (PBCON)
// ===================================================================================

// Query workspace for banded condition number estimation
query_workspace_cholesky_condition_banded :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) {
	when T == f32 || T == f64 {
		return 3 * n, n, 0 // Real types need work and iwork
	} else when T == complex64 || T == complex128 {
		return 2 * n, 0, n // Complex types need work and rwork, no iwork
	}
}

// Estimate condition number of banded positive definite matrix (f32/complex64)
m_cholesky_condition_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Factorized banded matrix from pbtrf
	kd: int, // Number of super/sub-diagonals
	anorm: f32, // 1-norm of original matrix (use m_norm_banded)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f32 (nil for complex64)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f32,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.spbcon_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.cpbcon_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// Estimate condition number of banded positive definite matrix (f64/complex128)
m_cholesky_condition_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Factorized banded matrix from pbtrf
	kd: int, // Number of super/sub-diagonals
	anorm: f64, // 1-norm of original matrix (use m_norm_banded)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f64 (nil for complex128)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f64,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.dpbcon_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.zpbcon_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// BANDED EQUILIBRATION (PBEQU)
// ===================================================================================

// Compute equilibration scaling factors for banded positive definite matrix (f32/complex64)
m_cholesky_equilibrate_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix
	kd: int, // Number of super/sub-diagonals
	S: []f32, // Scaling factors (pre-allocated, size n)
	uplo := MatrixRegion.Upper,
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where T == f32 || T == complex64 {
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(len(S) >= AB.cols, "S array too small")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f32 {
		lapack.spbequ_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, raw_data(S), &scond, &amax, &info)
	} else when T == complex64 {
		lapack.cpbequ_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Compute equilibration scaling factors for banded positive definite matrix (f64/complex128)
m_cholesky_equilibrate_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix
	kd: int, // Number of super/sub-diagonals
	S: []f64, // Scaling factors (pre-allocated, size n)
	uplo := MatrixRegion.Upper,
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where T == f64 || T == complex128 {
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(len(S) >= AB.cols, "S array too small")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	ldab := AB.ld

	when T == f64 {
		lapack.dpbequ_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, raw_data(S), &scond, &amax, &info)
	} else when T == complex128 {
		lapack.zpbequ_(&uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// BANDED ITERATIVE REFINEMENT (PBRFS)
// ===================================================================================

// Query workspace for banded iterative refinement
query_workspace_cholesky_refine_banded :: proc($T: typeid, n: int, nrhs: int) -> (work_size: int, iwork_size: int, rwork_size: int) {
	when T == f32 || T == f64 {
		return 3 * n, n, 0 // Real types need work and iwork
	} else when T == complex64 || T == complex128 {
		return 2 * n, 0, n // Complex types need work and rwork
	}
}

// Iterative refinement for banded positive definite system (f32/complex64)
m_cholesky_refine_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized banded matrix from pbtrf
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	kd: int, // Number of super/sub-diagonals
	ferr: []f32, // Forward error bounds (pre-allocated, size nrhs)
	berr: []f32, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for f32
	rwork: []f32 = nil, // Real workspace for complex64
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	assert(AB.rows == AB.cols && AFB.rows == AFB.cols, "Matrices must be square")
	assert(B.rows == AB.rows && X.rows == AB.rows, "System dimensions must match")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= B.cols && len(berr) >= B.cols, "Error arrays too small")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.spbrfs_(
			&uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.cpbrfs_(
			&uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return info, info == 0
}

// Iterative refinement for banded positive definite system (f64/complex128)
m_cholesky_refine_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized banded matrix from pbtrf
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	kd: int, // Number of super/sub-diagonals
	ferr: []f64, // Forward error bounds (pre-allocated, size nrhs)
	berr: []f64, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for f64
	rwork: []f64 = nil, // Real workspace for complex128
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	assert(AB.rows == AB.cols && AFB.rows == AFB.cols, "Matrices must be square")
	assert(B.rows == AB.rows && X.rows == AB.rows, "System dimensions must match")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= B.cols && len(berr) >= B.cols, "Error arrays too small")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	uplo_c := cast(u8)uplo
	n := AB.cols
	kd_val := Blas_Int(kd)
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.dpbrfs_(
			&uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.zpbrfs_(
			&uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return info, info == 0
}
