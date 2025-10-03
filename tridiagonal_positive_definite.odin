package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// POSITIVE DEFINITE TRIDIAGONAL OPERATIONS (PT PREFIX)
// ===================================================================================
// Linear system solvers and eigenvalue computation for positive definite tridiagonal matrices
// Non-allocating API with pre-allocated arrays

// ===================================================================================
// POSITIVE DEFINITE TRIDIAGONAL FACTORIZATION (PTTRF)
// ===================================================================================

// Factorize positive definite tridiagonal matrix using Cholesky factorization
// For real matrices: T = L*D*L^T where L is unit lower triangular, D is diagonal
// For complex matrices: T = L*D*L^H where L is unit lower triangular, D is real diagonal
positive_definite_tridiagonal_factorize :: proc {
	positive_definite_tridiagonal_factorize_f32_f64,
	positive_definite_tridiagonal_factorize_c64_c128,
}

// Factorize positive definite tridiagonal matrix for f32/f64
positive_definite_tridiagonal_factorize_f32_f64 :: proc(
	d: []$T, // Diagonal elements (modified to factorized diagonal)
	e: []T, // Off-diagonal elements (modified to factorized off-diagonal)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	when T == f32 {
		lapack.spttrf_(&n_int, raw_data(d), raw_data(e), &info)
	} else when T == f64 {
		lapack.dpttrf_(&n_int, raw_data(d), raw_data(e), &info)
	}

	return info, info == 0
}

// Factorize positive definite tridiagonal matrix for c64/c128
positive_definite_tridiagonal_factorize_c64_c128 :: proc(
	d: []$R, // Diagonal elements (real, modified to factorized diagonal)
	e: []$T, // Off-diagonal elements (complex, modified to factorized off-diagonal)
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.cpttrf_(&n_int, raw_data(d), raw_data(e), &info)
	} else when T == complex128 {
		lapack.zpttrf_(&n_int, raw_data(d), raw_data(e), &info)
	}

	return info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE TRIDIAGONAL SOLVE USING FACTORIZATION (PTTRS)
// ===================================================================================

// Solve positive definite tridiagonal system using factorization from PTTRF
positive_definite_tridiagonal_solve_factorized :: proc {
	positive_definite_tridiagonal_solve_factorized_f32_f64,
	positive_definite_tridiagonal_solve_factorized_c64_c128,
}

// Solve factorized positive definite tridiagonal system for f32/f64
positive_definite_tridiagonal_solve_factorized_f32_f64 :: proc(
	d: []$T, // Factorized diagonal from PTTRF
	e: []T, // Factorized off-diagonal from PTTRF
	B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n, "B matrix too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == f32 {
		lapack.spttrs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpttrs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve factorized positive definite tridiagonal system for c64/c128
positive_definite_tridiagonal_solve_factorized_c64_c128 :: proc(
	d: []$R, // Factorized diagonal from PTTRF (real)
	e: []$T, // Factorized off-diagonal from PTTRF (complex)
	B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
	uplo := MatrixRegion.Lower, // Storage format for B
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n, "B matrix too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == complex64 {
		lapack.cpttrs_(&uplo_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpttrs_(&uplo_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE TRIDIAGONAL DIRECT SOLVE (PTSV)
// ===================================================================================

// Solve positive definite tridiagonal system directly (factorization + solve in one call)
positive_definite_tridiagonal_solve :: proc {
	positive_definite_tridiagonal_solve_f32_f64,
	positive_definite_tridiagonal_solve_c64_c128,
}

// Solve positive definite tridiagonal system directly for f32/f64
positive_definite_tridiagonal_solve_f32_f64 :: proc(
	d: []$T, // Diagonal elements (modified during factorization)
	e: []T, // Off-diagonal elements (modified during factorization)
	B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n, "B matrix too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == f32 {
		lapack.sptsv_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dptsv_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve positive definite tridiagonal system directly for c64/c128
positive_definite_tridiagonal_solve_c64_c128 :: proc(
	d: []$R, // Diagonal elements (real, modified during factorization)
	e: []$T, // Off-diagonal elements (complex, modified during factorization)
	B: ^Matrix(T), // Right-hand side matrix (overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n, "B matrix too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == complex64 {
		lapack.cptsv_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zptsv_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION (PTCON)
// ===================================================================================

// Query workspace for condition number estimation
query_workspace_positive_definite_tridiagonal_condition :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) {
	when T == f32 || T == f64 {
		// Real types need work array
		return n, 0
	} else {
		// Complex types need rwork array
		return 0, n
	}
}

// Estimate condition number of factorized positive definite tridiagonal matrix
positive_definite_tridiagonal_condition :: proc {
	positive_definite_tridiagonal_condition_f32_f64,
	positive_definite_tridiagonal_condition_c64_c128,
}

// Estimate condition number for f32/f64
positive_definite_tridiagonal_condition_f32_f64 :: proc(
	d: []$T, // Factorized diagonal from PTTRF
	e: []T, // Factorized off-diagonal from PTTRF
	anorm: T, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size n)
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= n, "Work array too small")

	n_int := Blas_Int(n)
	anorm_val := anorm

	when T == f32 {
		lapack.sptcon_(&n_int, raw_data(d), raw_data(e), &anorm_val, &rcond, raw_data(work), &info)
	} else when T == f64 {
		lapack.dptcon_(&n_int, raw_data(d), raw_data(e), &anorm_val, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// Estimate condition number for c64/c128
positive_definite_tridiagonal_condition_c64_c128 :: proc(
	d: []$R, // Factorized diagonal from PTTRF (real)
	e: []$T, // Factorized off-diagonal from PTTRF (complex)
	anorm: R, // 1-norm of original matrix
	rwork: []R, // Pre-allocated real workspace (size n)
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(rwork) >= n, "Real work array too small")

	n_int := Blas_Int(n)
	anorm_val := anorm

	when T == complex64 {
		lapack.cptcon_(&n_int, raw_data(d), raw_data(e), &anorm_val, &rcond, raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zptcon_(&n_int, raw_data(d), raw_data(e), &anorm_val, &rcond, raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (PTRFS)
// ===================================================================================

// Query workspace for iterative refinement
query_workspace_positive_definite_tridiagonal_refinement :: proc($T: typeid, nrhs: int) -> (work_size: int, rwork_size: int, ferr_size: int, berr_size: int) {
	when T == f32 || T == f64 {
		// Real types need work array
		return nrhs, 0, nrhs, nrhs
	} else {
		// Complex types need work and rwork arrays
		return nrhs, nrhs, nrhs, nrhs
	}
}

// Perform iterative refinement for positive definite tridiagonal system
positive_definite_tridiagonal_refine :: proc {
	positive_definite_tridiagonal_refine_f32_f64,
	positive_definite_tridiagonal_refine_c64_c128,
}

// Iterative refinement for f32/f64
positive_definite_tridiagonal_refine_f32_f64 :: proc(
	d: []$T, // Original diagonal
	e: []T, // Original off-diagonal
	df: []T, // Factorized diagonal from PTTRF
	ef: []T, // Factorized off-diagonal from PTTRF
	B: ^Matrix(T), // Original right-hand side
	X: ^Matrix(T), // Solution (input initial guess, output refined solution)
	ferr: []T, // Forward error bounds (size nrhs)
	berr: []T, // Backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size nrhs)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
	assert(B.cols == X.cols, "Number of right-hand sides must match")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")
	assert(len(work) >= nrhs, "Work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.sptrfs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == f64 {
		lapack.dptrfs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	}

	return info, info == 0
}

// Iterative refinement for c64/c128
positive_definite_tridiagonal_refine_c64_c128 :: proc(
	d: []$R, // Original diagonal (real)
	e: []$T, // Original off-diagonal (complex)
	df: []R, // Factorized diagonal from PTTRF (real)
	ef: []T, // Factorized off-diagonal from PTTRF (complex)
	B: ^Matrix(T), // Original right-hand side
	X: ^Matrix(T), // Solution (input initial guess, output refined solution)
	ferr: []R, // Forward error bounds (size nrhs)
	berr: []R, // Backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size nrhs)
	rwork: []R, // Pre-allocated real workspace (size nrhs)
	uplo := MatrixRegion.Lower, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
	assert(B.cols == X.cols, "Number of right-hand sides must match")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")
	assert(len(work) >= nrhs, "Work array too small")
	assert(len(rwork) >= nrhs, "Real work array too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == complex64 {
		lapack.cptrfs_(&uplo_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zptrfs_(&uplo_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT DRIVER (PTSVX)
// ===================================================================================

// Expert driver for solving positive definite tridiagonal systems with optional
// factorization, condition estimation, and error bounds
positive_definite_tridiagonal_solve_expert :: proc {
	positive_definite_tridiagonal_solve_expert_f32_f64,
	positive_definite_tridiagonal_solve_expert_c64_c128,
}

// Expert driver for f32/f64
positive_definite_tridiagonal_solve_expert_f32_f64 :: proc(
	d: []$T, // Diagonal elements
	e: []T, // Off-diagonal elements
	df: []T, // Factorized diagonal (output if fact='N', input if fact='F')
	ef: []T, // Factorized off-diagonal (output if fact='N', input if fact='F')
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ferr: []T, // Forward error bounds (output, size nrhs)
	berr: []T, // Backward error bounds (output, size nrhs)
	work: []T, // Pre-allocated workspace
	fact := FactorizationOption.Factor,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
	assert(B.cols == X.cols, "Number of right-hand sides must match")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

	fact_c := cast(u8)fact
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.sptsvx_(&fact_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == f64 {
		lapack.dptsvx_(&fact_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// Expert driver for c64/c128
positive_definite_tridiagonal_solve_expert_c64_c128 :: proc(
	d: []$R, // Diagonal elements (real)
	e: []$T, // Off-diagonal elements (complex)
	df: []R, // Factorized diagonal (output if fact='N', input if fact='F')
	ef: []T, // Factorized off-diagonal (output if fact='N', input if fact='F')
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (output)
	ferr: []R, // Forward error bounds (output, size nrhs)
	berr: []R, // Backward error bounds (output, size nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []R, // Pre-allocated real workspace
	fact := FactorizationOption.Factor,
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	nrhs := B.cols
	assert(B.rows >= n && X.rows >= n, "Matrix dimensions incorrect")
	assert(B.cols == X.cols, "Number of right-hand sides must match")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

	fact_c := cast(u8)fact
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == complex64 {
		lapack.cptsvx_(&fact_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zptsvx_(&fact_c, &n_int, &nrhs_int, raw_data(d), raw_data(e), raw_data(df), raw_data(ef), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE TRIDIAGONAL EIGENVALUES (PTEQR)
// ===================================================================================

// Query workspace for positive definite tridiagonal eigenvalue computation
query_workspace_positive_definite_tridiagonal_eigenvalues :: proc($T: typeid, n: int) -> (work_size: int) {
	// PTEQR requires 4*n workspace for real types
	when T == f32 || T == complex64 {
		return 4 * n
	} else when T == f64 || T == complex128 {
		return 4 * n
	}
}

// Compute eigenvalues/eigenvectors of positive definite tridiagonal matrix
positive_definite_tridiagonal_eigenvalues :: proc {
	positive_definite_tridiagonal_eigenvalues_f32,
	positive_definite_tridiagonal_eigenvalues_f64,
	positive_definite_tridiagonal_eigenvalues_c64,
	positive_definite_tridiagonal_eigenvalues_c128,
}

// Eigenvalue computation for f32
positive_definite_tridiagonal_eigenvalues_f32 :: proc(
	d: []f32, // Diagonal elements (modified to eigenvalues on output)
	e: []f32, // Off-diagonal elements (destroyed)
	Z: ^Matrix(f32) = nil, // Eigenvector matrix (optional output)
	work: []f32, // Pre-allocated workspace (size 4*n)
	compz := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	if compz != .None {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if Z != nil && compz != .None {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.spteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &info)

	return info, info == 0
}

// Eigenvalue computation for f64
positive_definite_tridiagonal_eigenvalues_f64 :: proc(
	d: []f64, // Diagonal elements (modified to eigenvalues on output)
	e: []f64, // Off-diagonal elements (destroyed)
	Z: ^Matrix(f64) = nil, // Eigenvector matrix (optional output)
	work: []f64, // Pre-allocated workspace (size 4*n)
	compz := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	if compz != .None {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if Z != nil && compz != .None {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.dpteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &info)

	return info, info == 0
}

// Eigenvalue computation for c64
positive_definite_tridiagonal_eigenvalues_c64 :: proc(
	d: []f32, // Diagonal elements (real, modified to eigenvalues on output)
	e: []f32, // Off-diagonal elements (real, destroyed)
	Z: ^Matrix(complex64) = nil, // Eigenvector matrix (optional output)
	work: []f32, // Pre-allocated workspace (size 4*n)
	compz := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	if compz != .None {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if Z != nil && compz != .None {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.cpteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &info)

	return info, info == 0
}

// Eigenvalue computation for c128
positive_definite_tridiagonal_eigenvalues_c128 :: proc(
	d: []f64, // Diagonal elements (real, modified to eigenvalues on output)
	e: []f64, // Off-diagonal elements (real, destroyed)
	Z: ^Matrix(complex128) = nil, // Eigenvector matrix (optional output)
	work: []f64, // Pre-allocated workspace (size 4*n)
	compz := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	if compz != .None {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if Z != nil && compz != .None {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.zpteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &info)

	return info, info == 0
}

// ===================================================================================
// TRIDIAGONAL MATRIX-VECTOR MULTIPLICATION
// ===================================================================================

// Bandwidth-efficient matrix-vector multiply for tridiagonal matrices
tridiagonal_matrix_vector_multiply :: proc(
	d: []$T, // Diagonal
	e: []$S, // Off-diagonal
	x: ^Vector($U), // Input vector
	y: ^Vector(U), // Output vector
) {
	n := len(d)
	if x.len != n || y.len != n {
		panic("Vector dimension mismatch")
	}

	// y[0] = d[0]*x[0] + e[0]*x[1]
	if n > 0 {
		val := U(d[0]) * vector_get(x, 0)
		if n > 1 {
			val += U(e[0]) * vector_get(x, 1)
		}
		vector_set(y, 0, val)
	}

	// y[i] = e[i-1]*x[i-1] + d[i]*x[i] + e[i]*x[i+1]
	for i in 1 ..< n - 1 {
		val := U(e[i - 1]) * vector_get(x, i - 1) + U(d[i]) * vector_get(x, i) + U(e[i]) * vector_get(x, i + 1)
		vector_set(y, i, val)
	}

	// y[n-1] = e[n-2]*x[n-2] + d[n-1]*x[n-1]
	if n > 1 {
		val := U(e[n - 2]) * vector_get(x, n - 2) + U(d[n - 1]) * vector_get(x, n - 1)
		vector_set(y, n - 1, val)
	}
}

// ===================================================================================
// SOLUTION QUALITY ANALYSIS
// ===================================================================================

SolutionQuality :: enum {
	Excellent,
	Good,
	Fair,
	Poor,
	IllConditioned,
	Singular,
}

// Check solution accuracy for tridiagonal system
check_tridiagonal_solution :: proc(
	d: []$T, // Diagonal
	e: []$S, // Off-diagonal
	B: ^Matrix($U), // Original RHS
	X: ^Matrix(U), // Solution
	allocator := context.allocator,
) -> (
	residual_norm: f64,
	relative_error: f64,
) {
	n := len(d)

	// Compute residual r = B - A*X
	residual := matrix_clone(B, allocator)
	defer matrix_delete(&residual)

	// Compute A*X using tridiagonal structure
	for j in 0 ..< X.cols {
		for i in 0 ..< n {
			ax_val := U(d[i]) * matrix_get(X, i, j)

			if i > 0 {
				ax_val += U(e[i - 1]) * matrix_get(X, i - 1, j)
			}
			if i < n - 1 {
				ax_val += U(e[i]) * matrix_get(X, i + 1, j)
			}

			// r[i,j] = b[i,j] - ax_val
			r_val := matrix_get(&residual, i, j) - ax_val
			matrix_set(&residual, i, j, r_val)
		}
	}

	// Compute norms
	residual_norm = 0.0
	b_norm := 0.0

	for j in 0 ..< B.cols {
		for i in 0 ..< n {
			r_val := matrix_get(&residual, i, j)
			b_val := matrix_get(B, i, j)

			when U == complex64 || U == complex128 {
				residual_norm += real(r_val * conj(r_val))
				b_norm += real(b_val * conj(b_val))
			} else {
				residual_norm += f64(r_val * r_val)
				b_norm += f64(b_val * b_val)
			}
		}
	}

	residual_norm = math.sqrt(residual_norm)
	b_norm = math.sqrt(b_norm)

	if b_norm > 0 {
		relative_error = residual_norm / b_norm
	} else {
		relative_error = residual_norm
	}

	return residual_norm, relative_error
}
