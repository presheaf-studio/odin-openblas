package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// HERMITIAN LINEAR SYSTEM SOLVERS - NON-ALLOCATING API
// ============================================================================
// Complex Hermitian matrices: A^H = A (conjugate transpose equals original)
// Bunch-Kaufman diagonal pivoting for indefinite Hermitian matrices

// Query workspace for Hermitian system solver (HESV)
query_workspace_solve_hermitian :: proc {
	query_workspace_solve_hermitian_complex,
}

query_workspace_solve_hermitian_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.chesv_(
			&uplo_c,
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
		)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zhesv_(
			&uplo_c,
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
		)
		work_size = int(real(work_query))
	}

	return work_size
}

// Solve Hermitian system using Bunch-Kaufman pivoting
solve_hermitian :: proc {
	solve_hermitian_complex,
}

solve_hermitian_complex :: proc(
	A: ^Matrix($Cmplx), // System matrix (modified on output)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (n)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chesv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhesv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// HERMITIAN ROOK PIVOTING SOLVERS - Enhanced numerical stability
// ============================================================================

// Query workspace for Hermitian system solver with Rook pivoting (HESV_ROOK)
query_workspace_solve_hermitian_rook :: proc {
	query_workspace_solve_hermitian_rook_complex,
}

query_workspace_solve_hermitian_rook_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.chesv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zhesv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Solve Hermitian system using Rook pivoting (enhanced numerical stability)
solve_hermitian_rook :: proc {
	solve_hermitian_rook_complex,
}

solve_hermitian_rook_complex :: proc(
	A: ^Matrix($Cmplx), // System matrix (modified on output)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (n)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chesv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhesv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// HERMITIAN RK PIVOTING SOLVERS - Bounded Bunch-Kaufman
// ============================================================================

// Query workspace for Hermitian system solver with RK pivoting (HESV_RK)
query_workspace_solve_hermitian_rk :: proc {
	query_workspace_solve_hermitian_rk_complex,
}

query_workspace_solve_hermitian_rk_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.chesv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zhesv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Solve Hermitian system using RK pivoting (bounded Bunch-Kaufman)
solve_hermitian_rk :: proc {
	solve_hermitian_rk_complex,
}

solve_hermitian_rk_complex :: proc(
	A: ^Matrix($Cmplx), // System matrix (modified on output)
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	E: ^Matrix(Cmplx), // Factor E from RK factorization
	ipiv: []Blas_Int, // Pre-allocated pivot indices (n)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(E.rows == n && E.cols == n, "E matrix dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	lde := E.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chesv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhesv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// EXPERT DRIVERS - With error bounds and condition estimation
// ============================================================================

// Query workspace for expert Hermitian solver (HESVX)
query_workspace_solve_hermitian_expert :: proc {
	query_workspace_solve_hermitian_expert_complex,
}

query_workspace_solve_hermitian_expert_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	fact_c: u8 = 'N' // New factorization
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldaf := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	ldx := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		rwork_query: f32
		lapack.chesvx_(
			&fact_c,
			&uplo_c,
			&n_int,
			&nrhs_int,
			nil,
			&lda,
			nil,
			&ldaf,
			nil, // A, AF, ipiv
			nil,
			&ldb,
			nil,
			&ldx, // B, X
			nil,
			nil,
			nil, // rcond, ferr, berr
			&work_query,
			&lwork,
			&rwork_query,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork_query: f64
		lapack.zhesvx_(
			&fact_c,
			&uplo_c,
			&n_int,
			&nrhs_int,
			nil,
			&lda,
			nil,
			&ldaf,
			nil, // A, AF, ipiv
			nil,
			&ldb,
			nil,
			&ldx, // B, X
			nil,
			nil,
			nil, // rcond, ferr, berr
			&work_query,
			&lwork,
			&rwork_query,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	}

	return work_size, rwork_size
}

// Expert driver for Hermitian linear systems with error bounds and condition estimation
solve_hermitian_expert :: proc {
	solve_hermitian_expert_complex,
}

solve_hermitian_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (preserved)
	B: ^Matrix(Cmplx), // RHS matrix (preserved)
	X: ^Matrix(Cmplx), // Solution matrix (output)
	AF: ^Matrix(Cmplx), // Factorization matrix (workspace/output)
	ipiv: []Blas_Int, // Pivot indices (output)
	work: []Cmplx, // Workspace
	rwork: []$Real, // Real workspace
	uplo := MatrixRegion.Upper,
	fact: u8 = 'N', // 'N' = new factorization, 'F' = use given factorization
) -> (
	rcond: Real,
	ferr: []Real,// Reciprocal condition number
	berr: []Real,// Forward error bounds
	info: Info,// Backward error bounds
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n && X.rows == n, "Dimension mismatch")
	assert(B.cols == X.cols, "RHS/solution dimension mismatch")
	assert(AF.rows == n && AF.cols == n, "Factorization matrix size mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(ferr) >= nrhs && len(berr) >= nrhs, "Error bound arrays too small")

	fact_c := fact
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chesvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zhesvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rcond, ferr, berr, info, info == 0
}

// ============================================================================
// HERMITIAN FACTORIZATIONS (for use with factorization-based solvers)
// ============================================================================

// Query workspace for Hermitian factorization (HETRF)
query_workspace_factorize_hermitian :: proc {
	query_workspace_factorize_hermitian_complex,
}

query_workspace_factorize_hermitian_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.chetrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zhetrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize Hermitian matrix using Bunch-Kaufman diagonal pivoting
factorize_hermitian :: proc {
	factorize_hermitian_complex,
}

factorize_hermitian_complex :: proc(
	A: ^Matrix($Cmplx), // Input/output matrix (overwritten with factorization)
	ipiv: []Blas_Int, // Output pivot indices (length n)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chetrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhetrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using Hermitian factorization from HETRF
solve_using_hermitian_factorization :: proc {
	solve_using_hermitian_factorization_complex,
}

solve_using_hermitian_factorization_complex :: proc(
	A: ^Matrix($Cmplx), // Factorized matrix from factorize_hermitian
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld

	when Cmplx == complex64 {
		lapack.chetrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when Cmplx == complex128 {
		lapack.zhetrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve system using improved Hermitian algorithm (HETRS2) with workspace
solve_using_hermitian_factorization_improved :: proc {
	solve_using_hermitian_factorization_improved_complex,
}

solve_using_hermitian_factorization_improved_complex :: proc(
	A: ^Matrix($Cmplx), // Factorized matrix from factorize_hermitian
	B: ^Matrix(Cmplx), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []Cmplx, // Workspace for improved algorithm
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld

	when Cmplx == complex64 {
		lapack.chetrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zhetrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	}

	return info, info == 0
}
// ============================================================================
// CONDITION NUMBER ESTIMATION (HECON family)
// ============================================================================

hermitian_condition_number :: proc {
	hermitian_condition_number_complex,
}

query_workspace_hermitian_condition_number :: proc($Cmplx: typeid, n: int) -> (work_size: int) where is_complex(Cmplx) {
	return int(2 * n)
}

// Estimate condition number of Hermitian matrix
hermitian_condition_number_complex :: proc(
	A: ^Matrix($Cmplx), // Factored matrix from hetrf
	ipiv: []Blas_Int, // Pivot indices from hetrf
	anorm: $Real, // 1-norm of original matrix
	work: []Cmplx, // Workspace (pre-allocated, size 2*n)
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	info: Info,// Reciprocal condition number
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(2 * n), "work array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.checon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zhecon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// ============================================================================
// IMPROVED EQUILIBRATION (HEEQUB family)
// ============================================================================

hermitian_equilibrate_improved :: proc {
	hermitian_equilibrate_improved_complex,
}

query_workspace_hermitian_equilibrate_improved :: proc($Cmplx: typeid, n: int) -> (work_size: int) where is_complex(Cmplx) {
	return int(3 * n)
}

// Improved equilibration for Hermitian matrices
hermitian_equilibrate_improved_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (not modified)
	S: []$Real, // Scaling factors (pre-allocated, size n)
	work: []Cmplx, // Workspace (pre-allocated, size 3*n)
	uplo: MatrixRegion = .Upper,
) -> (
	scond: Real,
	amax: Real,// Ratio of smallest to largest scaling factor
	info: Info,// Absolute value of largest matrix element
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(3 * n), "work array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cheequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zheequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	}

	return scond, amax, info, info == 0
}

// ============================================================================
// ITERATIVE REFINEMENT (HERFS family)
// ============================================================================

hermitian_iterative_refinement :: proc {
	hermitian_iterative_refinement_complex,
}

query_workspace_hermitian_iterative_refinement :: proc($Cmplx: typeid, n: int) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	return int(2 * n), int(n)
}

// Iterative refinement for Hermitian linear systems
hermitian_iterative_refinement_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Factored matrix from hetrf
	ipiv: []Blas_Int, // Pivot indices from hetrf
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution (improved on output)
	ferr: []$Real, // Forward error bounds (pre-allocated, size nrhs)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cherfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zherfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ============================================================================
// EXPERT ITERATIVE REFINEMENT (HERFSX family)
// ============================================================================

hermitian_iterative_refinement_expert :: proc {
	hermitian_iterative_refinement_expert_complex,
}

query_workspace_hermitian_iterative_refinement_expert :: proc($Cmplx: typeid, n: int, n_err_bnds: int = 3) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	return int(2 * n), int(2 * n)
}

// Expert iterative refinement with multiple error bounds
hermitian_iterative_refinement_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Factored matrix from hetrf
	ipiv: []Blas_Int, // Pivot indices from hetrf
	S: []$Real, // Scaling factors
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution (improved on output)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []Real, // Normwise error bounds [nrhs x n_err_bnds]
	err_bnds_comp: []Real, // Componentwise error bounds [nrhs x n_err_bnds]
	params: []Real, // Algorithm parameters (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	equed: Equilibration_Type = .None,
	n_err_bnds: int = 3,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(2 * n), "rwork array too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)equed
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when Cmplx == complex64 {
		lapack.cherfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zherfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// ============================================================================
// MATRIX INVERSION (HETRI family)
// ============================================================================

hermitian_invert :: proc {
	hermitian_invert_complex,
}

query_workspace_hermitian_invert :: proc($Cmplx: typeid, n: int) -> (work_size: int) where is_complex(Cmplx) {
	return int(n)
}

// Invert Hermitian matrix using factorization
hermitian_invert_complex :: proc(
	A: ^Matrix($T), // Factored matrix (overwritten with inverse)
	ipiv: []Blas_Int, // Pivot indices from hetrf
	work: []T, // Workspace (pre-allocated, size n)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(n), "work array too small")

	uplo_c := cast(u8)uplo

	when T == complex64 {
		lapack.chetri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zhetri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	}

	return info, info == 0
}

// ============================================================================
// ROW/COLUMN SWAPPING (HESWAPR family)
// ============================================================================

hermitian_swap_rows :: proc {
	hermitian_swap_rows_complex,
}

// Apply row/column swaps to Hermitian matrix
hermitian_swap_rows_complex :: proc(
	A: ^Matrix($T), // Matrix to modify
	i1: int, // First row/column index
	i2: int, // Second row/column index
	uplo: MatrixRegion = .Upper,
) where is_complex(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(i1 >= 1 && i1 <= int(n), "i1 out of bounds")
	assert(i2 >= 1 && i2 <= int(n), "i2 out of bounds")

	uplo_c := cast(u8)uplo
	i1_c := Blas_Int(i1)
	i2_c := Blas_Int(i2)

	when T == complex64 {
		lapack.cheswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	} else when T == complex128 {
		lapack.zheswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	}
}
