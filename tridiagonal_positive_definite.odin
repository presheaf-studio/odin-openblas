package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"
import "core:slice"

// ===================================================================================
// TRIDIAGONAL POSITIVE DEFINITE EIGENVALUES, REFINEMENT, AND SOLVERS
// ===================================================================================

// Query workspace for tridiagonal eigenvalue computation with eigenvectors (PTEQR)
query_workspace_tridiagonal_pteqr :: proc($T: typeid, n: int) -> (work_size: int) {
	// PTEQR requires 4*n workspace for real types
	return 4 * n
}

// Compute eigenvalues/eigenvectors of symmetric positive definite tridiagonal matrix for f32/c64
m_compute_tridiagonal_pteqr_f32_c64 :: proc(
	D: []f32, // Diagonal elements (modified to eigenvalues on output)
	E: []f32, // Off-diagonal elements (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []f32, // Pre-allocated workspace
	mode := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	n_val := Blas_Int(n)
	compz_c := cast(u8)mode

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if mode != .None && Z != nil {
		assert(Z.rows == n && Z.cols == n, "Z matrix must be n×n")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.spteqr_(&compz_c, &n_val, raw_data(D), raw_data(E), cast(^f32)z_ptr, &ldz, raw_data(work), &info)
	} else when T == complex64 {
		lapack.cpteqr_(&compz_c, &n_val, raw_data(D), raw_data(E), z_ptr, &ldz, raw_data(work), &info)
	}


	return info, info == 0
}

// Compute eigenvalues/eigenvectors of symmetric positive definite tridiagonal matrix for f64/c128
m_compute_tridiagonal_pteqr_f64_c128 :: proc(
	D: []f64, // Diagonal elements (modified to eigenvalues on output)
	E: []f64, // Off-diagonal elements (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []f64, // Pre-allocated workspace
	mode := CompzOption.None, // Eigenvector computation mode
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")

	n_val := Blas_Int(n)
	compz_c := cast(u8)mode

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if mode != .None && Z != nil {
		assert(Z.rows == n && Z.cols == n, "Z matrix must be n×n")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f64 {
		lapack.dpteqr_(&compz_c, &n_val, raw_data(D), raw_data(E), z_ptr, &ldz, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zpteqr_(&compz_c, &n_val, raw_data(D), raw_data(E), z_ptr, &ldz, raw_data(work), &info)
	}


	return info, info == 0
}

// Proc group for tridiagonal eigenvalue computation
m_compute_tridiagonal_pteqr :: proc {
	m_compute_tridiagonal_pteqr_f32_c64,
	m_compute_tridiagonal_pteqr_f64_c128,
}

// ===================================================================================
// ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Query workspace for tridiagonal positive definite refinement
query_workspace_tridiagonal_refinement :: proc($T: typeid, n: int, nrhs: int) -> (work_size: int, rwork_size: int) {
	when T == f32 || T == f64 {
		// Real types: work is real, no rwork
		return n, 0
	} else when T == complex64 || T == complex128 {
		// Complex types: work is complex, rwork is real
		return n, n
	}
}

// ====================================================================================
// ITERATIVE REFINEMENT FOR TRIDIAGONAL POSITIVE DEFINITE SYSTEMS
// ====================================================================================

// Refactor tridiagonal positive definite refinement for f32/c64
m_refine_tridiagonal_pd_f32_c64 :: proc(
	D: []$RealType, // Original diagonal
	E: []$T, // Original off-diagonal (real for f32, complex for c64)
	DF: []RealType, // Factored diagonal
	EF: []T, // Factored off-diagonal
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []RealType, // Pre-allocated forward error bounds (nrhs)
	berr: []RealType, // Pre-allocated backward error bounds (nrhs)
	work: []T, // Pre-allocated workspace (n for complex, 2*n for real)
	rwork: []RealType = nil, // Pre-allocated real workspace (n for complex only)
	uplo := MatrixRegion.Upper, // Upper or lower (complex only)
) -> (
	info: Info,
	ok: bool,
) where (T == f32 && RealType == f32) || (T == complex64 && RealType == f32) {
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array dimension mismatch")
	assert(len(DF) == n && len(EF) >= n - 1 || n <= 1, "Factored array dimension mismatch")
	assert(B.rows == n && X.rows == n, "RHS/solution dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error arrays too small")

	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		// Real case: no uplo parameter
		assert(len(work) >= 2 * n, "Insufficient workspace for real")

		lapack.sptrfs_(&n_val, &nrhs_val, raw_data(D), raw_data(E), raw_data(DF), raw_data(EF), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == complex64 {
		// Complex case: has uplo parameter
		assert(len(work) >= n, "Insufficient workspace for complex")
		assert(len(rwork) >= n, "Insufficient real workspace for complex")

		uplo_c := matrix_region_to_cstring(uplo)

		lapack.cptrfs_(
			&uplo_c,
			&n_val,
			&nrhs_val,
			raw_data(D),
			raw_data(E),
			raw_data(DF),
			raw_data(EF),
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

// Refactor tridiagonal positive definite refinement for f64/c128
m_refine_tridiagonal_pd_f64_c128 :: proc(
	D: []$RealType, // Original diagonal
	E: []$T, // Original off-diagonal (real for f64, complex for c128)
	DF: []RealType, // Factored diagonal
	EF: []T, // Factored off-diagonal
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []RealType, // Pre-allocated forward error bounds (nrhs)
	berr: []RealType, // Pre-allocated backward error bounds (nrhs)
	work: []T, // Pre-allocated workspace (n for complex, 2*n for real)
	rwork: []RealType = nil, // Pre-allocated real workspace (n for complex only)
	uplo := MatrixRegion.Upper, // Upper or lower (complex only)
) -> (
	info: Info,
	ok: bool,
) where (T == f64 && RealType == f64) || (T == complex128 && RealType == f64) {
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array dimension mismatch")
	assert(len(DF) == n && len(EF) >= n - 1 || n <= 1, "Factored array dimension mismatch")
	assert(B.rows == n && X.rows == n, "RHS/solution dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error arrays too small")

	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(nrhs)
	ldb := B.ld
	ldx := X.ld

	when T == f64 {
		// Real case: no uplo parameter
		assert(len(work) >= 2 * n, "Insufficient workspace for real")

		lapack.dptrfs_(&n_val, &nrhs_val, raw_data(D), raw_data(E), raw_data(DF), raw_data(EF), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == complex128 {
		// Complex case: has uplo parameter
		assert(len(work) >= n, "Insufficient workspace for complex")
		assert(len(rwork) >= n, "Insufficient real workspace for complex")

		uplo_c := matrix_region_to_cstring(uplo)

		lapack.zptrfs_(
			&uplo_c,
			&n_val,
			&nrhs_val,
			raw_data(D),
			raw_data(E),
			raw_data(DF),
			raw_data(EF),
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

// Proc group for tridiagonal positive definite refinement
m_refine_tridiagonal_pd :: proc {
	m_refine_tridiagonal_pd_f32_c64,
	m_refine_tridiagonal_pd_f64_c128,
}


// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Workspace query for tridiagonal positive definite solver
query_workspace_solve_tridiagonal_pd :: proc($T: typeid, n: int, nrhs: int) -> (work_size: int) {
	// Simple solver (xPTSV) requires no workspace
	return 0
}

// Solve tridiagonal positive definite system (f32/complex64)
// Uses Cholesky factorization with pivoting if needed
m_solve_tridiagonal_pd_f32_c64 :: proc(
	D: []f32, // Diagonal (modified)
	E: []$T, // Off-diagonal (modified)
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "E must have at least n-1 elements")
	assert(B.rows == n, "RHS dimension mismatch")

	n_int := Blas_Int(n)
	nrhs := B.cols
	ldb := B.ld

	when T == f32 {
		lapack.sptsv_(&n_int, &nrhs, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cptsv_(&n_int, &nrhs, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve tridiagonal positive definite system (f64/complex128)
// Uses Cholesky factorization with pivoting if needed
m_solve_tridiagonal_pd_f64_c128 :: proc(
	D: []f64, // Diagonal (modified)
	E: []$T, // Off-diagonal (modified)
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "E must have at least n-1 elements")
	assert(B.rows == n, "RHS dimension mismatch")

	n_int := Blas_Int(n)
	nrhs := B.cols
	ldb := B.ld

	when T == f64 {
		lapack.dptsv_(&n_int, &nrhs, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zptsv_(&n_int, &nrhs, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Proc group for tridiagonal positive definite solver
m_solve_tridiagonal_pd :: proc {
	m_solve_tridiagonal_pd_f32_c64,
	m_solve_tridiagonal_pd_f64_c128,
}


// Bandwidth-efficient matrix-vector multiply for tridiagonal
tridiagonal_matvec :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	x: ^Vector($U), // Input vector
	y: ^Vector(U), // Output vector
) {
	n := len(D)
	if x.len != n || y.len != n {
		panic("Vector dimension mismatch")
	}

	// y[0] = D[0]*x[0] + E[0]*x[1]
	if n > 0 {
		val := U(D[0]) * vector_get(x, 0)
		if n > 1 {
			val += U(E[0]) * vector_get(x, 1)
		}
		vector_set(y, 0, val)
	}

	// y[i] = E[i-1]*x[i-1] + D[i]*x[i] + E[i]*x[i+1]
	for i in 1 ..< n - 1 {
		val := U(E[i - 1]) * vector_get(x, i - 1) + U(D[i]) * vector_get(x, i) + U(E[i]) * vector_get(x, i + 1)
		vector_set(y, i, val)
	}

	// y[n-1] = E[n-2]*x[n-2] + D[n-1]*x[n-1]
	if n > 1 {
		val := U(E[n - 2]) * vector_get(x, n - 2) + U(D[n - 1]) * vector_get(x, n - 1)
		vector_set(y, n - 1, val)
	}
}
