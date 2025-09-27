package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// TRIDIAGONAL POSITIVE DEFINITE EXPERT SOLVER
// ===================================================================================

// Query workspace for tridiagonal positive definite expert solver
query_workspace_tridiagonal_pd_expert :: proc($T: typeid, n: int, nrhs: int) -> (work_size: int, rwork_size: int, ferr_size: int, berr_size: int) {
	when T == f32 || T == f64 {
		// Real types: work only, no rwork
		return 2 * n, 0, nrhs, nrhs
	} else when T == complex64 || T == complex128 {
		// Complex types: work and rwork
		return n, n, nrhs, nrhs
	}
}

// Expert solver for tridiagonal positive definite system (f32/complex64)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_pd_expert_f32_c64 :: proc(
	D: []$RealType, // Diagonal elements
	E: []$T, // Off-diagonal elements
	DF: []RealType, // Factored diagonal (input/output)
	EF: []T, // Factored off-diagonal (input/output)
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (output)
	ferr: []RealType, // Pre-allocated forward error bounds (nrhs)
	berr: []RealType, // Pre-allocated backward error bounds (nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []RealType = nil, // Pre-allocated real workspace (complex only)
	fact := FactorizationOption.Equilibrate, // Factorization control
) -> (
	rcond: RealType,
	info: Info,
	ok: bool,
) where (T == f32 && RealType == f32) || (T == complex64 && RealType == f32) {
	// Validate inputs
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal must have at least n-1 elements")
	assert(len(DF) == n && (len(EF) >= n - 1 || n <= 1), "Factored arrays dimension mismatch")
	assert(B.rows == n && X.rows == n, "RHS and solution dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error arrays too small")

	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(nrhs)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	fact_c := factorization_to_char(fact)

	// If computing factorization, copy D and E to DF and EF
	if fact == .Equilibrate {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	when T == f32 {
		assert(len(work) >= 2 * n, "Insufficient workspace for real")

		lapack.sptsvx_(&fact_c, &n_val, &nrhs_val, raw_data(D), raw_data(E), raw_data(DF), raw_data(EF), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == complex64 {
		assert(len(work) >= n, "Insufficient workspace for complex")
		assert(len(rwork) >= n, "Insufficient real workspace for complex")

		lapack.cptsvx_(
			&fact_c,
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
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// Expert solver for tridiagonal positive definite system (f64/complex128)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_pd_expert_f64_c128 :: proc(
	D: []$RealType, // Diagonal elements
	E: []$T, // Off-diagonal elements
	DF: []RealType, // Factored diagonal (input/output)
	EF: []T, // Factored off-diagonal (input/output)
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (output)
	ferr: []RealType, // Pre-allocated forward error bounds (nrhs)
	berr: []RealType, // Pre-allocated backward error bounds (nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []RealType = nil, // Pre-allocated real workspace (complex only)
	fact := FactorizationOption.Equilibrate, // Factorization control
) -> (
	rcond: RealType,
	info: Info,
	ok: bool,
) where (T == f64 && RealType == f64) || (T == complex128 && RealType == f64) {
	// Validate inputs
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal must have at least n-1 elements")
	assert(len(DF) == n && (len(EF) >= n - 1 || n <= 1), "Factored arrays dimension mismatch")
	assert(B.rows == n && X.rows == n, "RHS and solution dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error arrays too small")

	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(nrhs)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	fact_c := factorization_to_char(fact)

	// If computing factorization, copy D and E to DF and EF
	if fact == .Equilibrate {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	when T == f64 {
		assert(len(work) >= 2 * n, "Insufficient workspace for real")

		lapack.dptsvx_(&fact_c, &n_val, &nrhs_val, raw_data(D), raw_data(E), raw_data(DF), raw_data(EF), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &info)
	} else when T == complex128 {
		assert(len(work) >= n, "Insufficient workspace for complex")
		assert(len(rwork) >= n, "Insufficient real workspace for complex")

		lapack.zptsvx_(
			&fact_c,
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
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// Proc group for tridiagonal positive definite expert solver
m_solve_tridiagonal_pd_expert :: proc {
	m_solve_tridiagonal_pd_expert_f32_c64,
	m_solve_tridiagonal_pd_expert_f64_c128,
}


// ===================================================================================
// CONVENIENCE FUNCTIONS
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
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	B: ^Matrix($U), // Original RHS
	X: ^Matrix(U), // Equilibrated solution
	allocator := context.allocator,
) -> (
	residual_norm: f64,
	relative_error: f64,
) {
	n := len(D)

	// Equilibrate residual r = B - A*X
	residual := matrix_clone(B, allocator)
	defer matrix_delete(&residual)

	// Equilibrate A*X using tridiagonal structure
	for j in 0 ..< X.cols {
		for i in 0 ..< n {
			ax_val := U(D[i]) * matrix_get(X, i, j)

			if i > 0 {
				ax_val += U(E[i - 1]) * matrix_get(X, i - 1, j)
			}
			if i < n - 1 {
				ax_val += U(E[i]) * matrix_get(X, i + 1, j)
			}

			// r[i,j] = b[i,j] - ax_val
			r_val := matrix_get(&residual, i, j) - ax_val
			matrix_set(&residual, i, j, r_val)
		}
	}

	// Equilibrate norms
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
