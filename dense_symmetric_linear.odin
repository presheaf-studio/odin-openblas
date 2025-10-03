package openblas

import lapack "./f77"
import "base:intrinsics"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC LINEAR SYSTEM SOLVERS - NON-ALLOCATING API
// ============================================================================
// Bunch-Kaufman diagonal pivoting for indefinite symmetric matrices
// Pre-allocated workspace and result arrays

query_workspace_solve_symmetric :: proc {
	query_workspace_solve_symmetric_real,
	query_workspace_solve_symmetric_complex,
}

solve_symmetric :: proc {
	solve_symmetric_real,
	solve_symmetric_complex,
}

query_workspace_solve_symmetric_rook :: proc {
	query_workspace_solve_symmetric_rook_real,
	query_workspace_solve_symmetric_rook_complex,
}

solve_symmetric_rook :: proc {
	solve_symmetric_rook_real,
	solve_symmetric_rook_complex,
}

query_workspace_solve_symmetric_rk :: proc {
	query_workspace_solve_symmetric_rk_real,
	query_workspace_solve_symmetric_rk_complex,
}

solve_symmetric_rk :: proc {
	solve_symmetric_rk_real,
	solve_symmetric_rk_complex,
}

query_workspace_solve_symmetric_expert :: proc {
	query_workspace_solve_symmetric_expert_real,
	query_workspace_solve_symmetric_expert_complex,
}

solve_symmetric_expert :: proc {
	solve_symmetric_expert_real,
	solve_symmetric_expert_complex,
}

// Query workspace for symmetric system solver (SYSV)
query_workspace_solve_symmetric_real :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_(
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
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_(
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
		work_size = int(work_query)
	}

	return work_size
}

query_workspace_solve_symmetric_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
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
		lapack.csysv_(
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
		lapack.zsysv_(
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

// Solve symmetric system using Bunch-Kaufman pivoting
solve_symmetric :: proc(
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

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsysv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csysv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsysv_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC ROOK PIVOTING SOLVERS - Advanced numerical stability
// ============================================================================

// Query workspace for symmetric system solver with Rook pivoting (SYSV_ROOK)
query_workspace_solve_symmetric_rook :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_rook_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Solve symmetric system using Rook pivoting (enhanced numerical stability)
solve_symmetric_rook :: proc(
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

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssysv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsysv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csysv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsysv_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC RK PIVOTING SOLVERS - Bounded Bunch-Kaufman
// ============================================================================

// Query workspace for symmetric system solver with RK pivoting (SYSV_RK)
query_workspace_solve_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(1)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssysv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsysv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csysv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsysv_rk_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, nil, &ldb, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Solve symmetric system using RK pivoting (bounded Bunch-Kaufman)
solve_symmetric_rk :: proc(
	A: ^Matrix($T), // System matrix (modified on output)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	E: ^Matrix(T), // Factor E from RK factorization
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

	when T == f32 {
		lapack.ssysv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsysv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csysv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsysv_rk_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// EXPERT DRIVERS - With error bounds and condition estimation
// ============================================================================

// Query workspace for expert symmetric solver (SYSVX)
query_workspace_solve_symmetric_expert :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
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

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssysvx_(
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
			&iwork_query,
			&info,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsysvx_(
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
			&iwork_query,
			&info,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		lapack.csysvx_(
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
		iwork_size = int(rwork_query) // For complex, this is rwork
	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		lapack.zsysvx_(
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
		iwork_size = int(rwork_query) // For complex, this is rwork
	}

	return work_size, iwork_size
}

// Expert driver for symmetric linear systems with error bounds and condition estimation
solve_symmetric_expert :: proc(
	A: ^Matrix($T), // Input matrix (preserved)
	B: ^Matrix(T), // RHS matrix (preserved)
	X: ^Matrix(T), // Solution matrix (output)
	AF: ^Matrix(T), // Factorization matrix (workspace/output)
	ipiv: []Blas_Int, // Pivot indices (output)
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace (or rwork for complex)
	ferr: []$Real, // Forward error bounds
	berr: []Real, // Backward error bounds
	uplo := MatrixRegion.Upper,
	fact: u8 = 'N', // 'N' = new factorization, 'F' = use given factorization
) -> (
	rcond: Real,
	info: Info,
	ok: bool, // Reciprocal condition number
) where (is_float(T) || is_complex(T)) && ((T == f32 && Real == f32) || (T == f64 && Real == f64) || (T == complex64 && Real == f32) || (T == complex128 && Real == f64)) {
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

	when T == f32 {
		lapack.ssysvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dsysvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == complex64 {
		rwork := transmute([]f32)iwork // For complex, iwork is really rwork
		lapack.csysvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(rwork), &info)
	} else when T == complex128 {
		rwork := transmute([]f64)iwork // For complex, iwork is really rwork
		lapack.zsysvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ============================================================================
// CONDITION NUMBER ESTIMATION (SYCON family)
// ============================================================================

symmetric_condition_number :: proc {
	symmetric_condition_number_real,
	symmetric_condition_number_complex,
}

query_workspace_symmetric_condition_number :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	when T == f32 || T == f64 {
		work_size = int(2 * n)
		iwork_size = int(n)
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		iwork_size = 0 // Complex variants don't use iwork
	}
	return work_size, iwork_size
}

// Estimate condition number of symmetric matrix
symmetric_condition_number_real :: proc(
	A: ^Matrix($T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: T, // 1-norm of original matrix
	work: []T, // Workspace (pre-allocated, size 2*n)
	iwork: []Blas_Int, // Integer workspace (pre-allocated, size n)
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: T,
	info: Info,
	ok: bool,// Reciprocal condition number
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssycon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dsycon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

symmetric_condition_number_complex :: proc(
	A: ^Matrix($Cmplx), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: $Real, // 1-norm of original matrix
	work: []Cmplx, // Workspace (pre-allocated, size 2*n)
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,// Reciprocal condition number
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(2 * n), "work array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.csycon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zsycon_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// ============================================================================
// IMPROVED EQUILIBRATION (SYEQUB family)
// ============================================================================

symmetric_equilibrate_improved :: proc {
	symmetric_equilibrate_improved_real,
	symmetric_equilibrate_improved_complex,
}

query_workspace_symmetric_equilibrate_improved :: proc($T: typeid, n: int) -> (work_size: int) where is_float(T) || is_complex(T) {
	return int(3 * n)
}

// Improved equilibration for symmetric matrices
symmetric_equilibrate_improved_real :: proc(
	A: ^Matrix($T), // Input matrix (not modified)
	S: []T, // Scaling factors (pre-allocated, size n)
	work: []T, // Workspace (pre-allocated, size 3*n)
	uplo: MatrixRegion = .Upper,
) -> (
	scond: T,
	amax: T,
	info: Info,// Ratio of smallest to largest scaling factor
	ok: bool,// Absolute value of largest matrix element
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(3 * n), "work array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssyequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsyequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	}

	return scond, amax, info, info == 0
}

symmetric_equilibrate_improved_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (not modified)
	S: []$Real, // Scaling factors (pre-allocated, size n)
	work: []Cmplx, // Workspace (pre-allocated, size 3*n)
	uplo: MatrixRegion = .Upper,
) -> (
	scond: Real,
	amax: Real,
	info: Info,// Ratio of smallest to largest scaling factor
	ok: bool,// Absolute value of largest matrix element
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(3 * n), "work array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.csyequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zsyequb_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, raw_data(work), &info)
	}

	return scond, amax, info, info == 0
}

// ============================================================================
// ITERATIVE REFINEMENT (SYRFS family)
// ============================================================================

symmetric_iterative_refinement :: proc {
	symmetric_iterative_refinement_real,
	symmetric_iterative_refinement_complex,
}

query_workspace_symmetric_iterative_refinement :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	when T == f32 || T == f64 {
		work_size = int(3 * n)
		iwork_size = int(n)
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		iwork_size = int(n) // For complex, this is rwork
	}
	return work_size, iwork_size
}

// Iterative refinement for symmetric linear systems
symmetric_iterative_refinement_real :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution (improved on output)
	ferr: []T, // Forward error bounds (pre-allocated, size nrhs)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
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
	assert(len(work) >= int(3 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssyrfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dsyrfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

symmetric_iterative_refinement_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
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
		lapack.csyrfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zsyrfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ============================================================================
// EXPERT ITERATIVE REFINEMENT (SYRFSX family)
// ============================================================================

symmetric_iterative_refinement_expert :: proc {
	symmetric_iterative_refinement_expert_real,
	symmetric_iterative_refinement_expert_complex,
}

query_workspace_symmetric_iterative_refinement_expert :: proc($T: typeid, n: int, n_err_bnds: int = 3) -> (work_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	when T == f32 || T == f64 {
		work_size = int(4 * n)
		iwork_size = int(n)
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		iwork_size = int(2 * n) // For complex, this is rwork
	}
	return work_size, iwork_size
}

// Expert iterative refinement with multiple error bounds
symmetric_iterative_refinement_expert_real :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	S: []T, // Scaling factors
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution (improved on output)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []T, // Normwise error bounds [nrhs x n_err_bnds]
	err_bnds_comp: []T, // Componentwise error bounds [nrhs x n_err_bnds]
	params: []T, // Algorithm parameters (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	equed: Equilibration_Type = .None,
	n_err_bnds: int = 3,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
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
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)equed
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when T == f32 {
		lapack.ssyrfsx_(
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
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dsyrfsx_(
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
			raw_data(iwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

symmetric_iterative_refinement_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
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
		lapack.csyrfsx_(
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
		lapack.zsyrfsx_(
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
// MATRIX INVERSION (SYTRI family)
// ============================================================================

symmetric_invert :: proc {
	symmetric_invert_real,
	symmetric_invert_complex,
}

query_workspace_symmetric_invert :: proc($T: typeid, n: int) -> (work_size: int) where is_float(T) || is_complex(T) {
	return int(n)
}

// Invert symmetric matrix using factorization
symmetric_invert_real :: proc(
	A: ^Matrix($T), // Factored matrix (overwritten with inverse)
	ipiv: []Blas_Int, // Pivot indices from sytrf
	work: []T, // Workspace (pre-allocated, size n)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(n), "work array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssytri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	} else when T == f64 {
		lapack.dsytri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	}

	return info, info == 0
}

symmetric_invert_complex :: proc(
	A: ^Matrix($T), // Factored matrix (overwritten with inverse)
	ipiv: []Blas_Int, // Pivot indices from sytrf
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
		lapack.csytri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zsytri_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &info)
	}

	return info, info == 0
}

// ============================================================================
// STORAGE FORMAT CONVERSION (SYCONV family)
// ============================================================================

symmetric_convert_storage :: proc {
	symmetric_convert_storage_real,
	symmetric_convert_storage_complex,
}

// Convert between symmetric matrix storage formats
symmetric_convert_storage_real :: proc(
	A: ^Matrix($T), // Matrix to convert (modified)
	ipiv: []Blas_Int, // Pivot indices
	E: []T, // Factor E (pre-allocated, size n)
	way: u8 = 'C', // 'C' = convert, 'R' = revert
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(E) >= int(n), "E array too small")

	uplo_c := cast(u8)uplo
	way_c := way

	when T == f32 {
		lapack.ssyconv_(&uplo_c, &way_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(E), &info)
	} else when T == f64 {
		lapack.dsyconv_(&uplo_c, &way_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(E), &info)
	}

	return info, info == 0
}

symmetric_convert_storage_complex :: proc(
	A: ^Matrix($T), // Matrix to convert (modified)
	ipiv: []Blas_Int, // Pivot indices
	E: []T, // Factor E (pre-allocated, size n)
	way: u8 = 'C', // 'C' = convert, 'R' = revert
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(E) >= int(n), "E array too small")

	uplo_c := cast(u8)uplo
	way_c := way

	when T == complex64 {
		lapack.csyconv_(&uplo_c, &way_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(E), &info)
	} else when T == complex128 {
		lapack.zsyconv_(&uplo_c, &way_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(E), &info)
	}

	return info, info == 0
}

// ============================================================================
// ROW/COLUMN SWAPPING (SYSWAPR family)
// ============================================================================

symmetric_swap_rows :: proc {
	symmetric_swap_rows_real,
	symmetric_swap_rows_complex,
}

// Apply row/column swaps to symmetric matrix
symmetric_swap_rows_real :: proc(
	A: ^Matrix($T), // Matrix to modify
	i1: int, // First row/column index
	i2: int, // Second row/column index
	uplo: MatrixRegion = .Upper,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix must be square")
	assert(i1 >= 1 && i1 <= int(n), "i1 out of bounds")
	assert(i2 >= 1 && i2 <= int(n), "i2 out of bounds")

	uplo_c := cast(u8)uplo
	i1_c := Blas_Int(i1)
	i2_c := Blas_Int(i2)

	when T == f32 {
		lapack.ssyswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	} else when T == f64 {
		lapack.dsyswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	}
}

symmetric_swap_rows_complex :: proc(
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
		lapack.csyswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	} else when T == complex128 {
		lapack.zsyswapr_(&uplo_c, &n, raw_data(A.data), &lda, &i1_c, &i2_c)
	}
}
