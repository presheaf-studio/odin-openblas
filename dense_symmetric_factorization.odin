package openblas

import lapack "./f77"
import "base:intrinsics"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC FACTORIZATIONS - NON-ALLOCATING API
// ============================================================================
// Bunch-Kaufman diagonal pivoting for indefinite symmetric matrices
// LDL^T factorization with 1x1 and 2x2 pivots

// ============================================================================
// STANDARD BUNCH-KAUFMAN FACTORIZATION (SYTRF)
// ============================================================================

query_workspace_factorize_symmetric :: proc {
	query_workspace_factorize_symmetric_real,
	query_workspace_factorize_symmetric_complex,
}

factorize_symmetric :: proc {
	factorize_symmetric_real,
	factorize_symmetric_complex,
}

solve_using_symmetric_factorization :: proc {
	solve_using_symmetric_factorization_real,
	solve_using_symmetric_factorization_complex,
}

// Query workspace for symmetric factorization (SYTRF)
query_workspace_factorize_symmetric_real :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	}

	return work_size
}

query_workspace_factorize_symmetric_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.csytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using Bunch-Kaufman diagonal pivoting
factorize_symmetric_real :: proc(
	A: ^Matrix($T), // Input/output matrix (overwritten with factorization)
	ipiv: []Blas_Int, // Output pivot indices (length n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

factorize_symmetric_complex :: proc(
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
		lapack.csytrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zsytrf_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using symmetric factorization from SYTRF
solve_using_symmetric_factorization_real :: proc(
	A: ^Matrix($T), // Factorized matrix from factorize_symmetric
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
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

	when T == f32 {
		lapack.ssytrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

solve_using_symmetric_factorization_complex :: proc(
	A: ^Matrix($Cmplx), // Factorized matrix from factorize_symmetric
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
		lapack.csytrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when Cmplx == complex128 {
		lapack.zsytrs_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// ROOK PIVOTING FACTORIZATION - Enhanced numerical stability
// ============================================================================

query_workspace_factorize_symmetric_rook :: proc {
	query_workspace_factorize_symmetric_rook_real,
	query_workspace_factorize_symmetric_rook_complex,
}

factorize_symmetric_rook :: proc {
	factorize_symmetric_rook_real,
	factorize_symmetric_rook_complex,
}

solve_using_symmetric_rook_factorization :: proc {
	solve_using_symmetric_rook_factorization_real,
	solve_using_symmetric_rook_factorization_complex,
}

// Query workspace for symmetric Rook factorization (SYTRF_ROOK)
query_workspace_factorize_symmetric_rook_real :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	}

	return work_size
}

query_workspace_factorize_symmetric_rook_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.csytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using Rook pivoting (enhanced stability)
factorize_symmetric_rook :: proc(
	A: ^Matrix($T), // Input/output matrix (overwritten with factorization)
	ipiv: []Blas_Int, // Output pivot indices (length n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rook_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_rook_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csytrf_rook_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsytrf_rook_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using Rook factorization
solve_using_symmetric_rook_factorization :: proc(
	A: ^Matrix($T), // Factorized matrix from factorize_symmetric_rook
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
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

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.csytrs_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zsytrs_rook_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// RK FACTORIZATION - Bounded Bunch-Kaufman
// ============================================================================

// Query workspace for symmetric RK factorization (SYTRF_RK)
query_workspace_factorize_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using RK pivoting (bounded Bunch-Kaufman)
factorize_symmetric_rk :: proc(
	A: ^Matrix($T), // Input/output matrix (overwritten with factorization)
	E: ^Matrix(T), // Output factor E
	ipiv: []Blas_Int, // Output pivot indices (length n)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(E.rows == n && E.cols == n, "E matrix dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lde := E.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rk_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_rk_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csytrf_rk_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsytrf_rk_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using RK factorization
solve_using_symmetric_rk_factorization :: proc(
	A: ^Matrix($T), // Factorized matrix from factorize_symmetric_rk
	E: ^Matrix(T), // Factor E from factorization
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	nrhs := B.cols
	assert(A.rows == A.cols, "Matrix must be square")
	assert(E.rows == n && E.cols == n, "E matrix dimension mismatch")
	assert(B.rows == n, "RHS dimension mismatch")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	lde := E.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs_3_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_3_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.csytrs_3_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zsytrs_3_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// IMPROVED SOLVERS WITH WORKSPACE (SYTRS2)
// ============================================================================

// Solve system using improved algorithm (SYTRS2) with workspace
solve_using_symmetric_factorization_improved :: proc(
	A: ^Matrix($T), // Factorized matrix from factorize_symmetric
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace for improved algorithm
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
	assert(len(work) >= n, "Workspace too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsytrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex64 {
		lapack.csytrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zsytrs2_(&uplo_c, &n_int, &nrhs_int, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC GENERALIZED EIGENVALUE REDUCTION (SYGST)
// ============================================================================
// Reduce generalized symmetric eigenvalue problem to standard form
// Transforms A*x = lambda*B*x into C*y = lambda*y where C = inv(U^T)*A*inv(U) or inv(L)*A*inv(L^T)

// Reduce generalized symmetric eigenvalue problem to standard form (real)
reduce_generalized_symmetric_to_standard_form_real :: proc(
	A: ^Matrix($T), // Input/output: Symmetric matrix A, overwritten with transformed matrix
	B: ^Matrix(T), // Input: Cholesky factor of B (from potrf)
	itype: int = 1, // Problem type: 1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimension")
	assert(itype >= 1 && itype <= 3, "Invalid problem type (must be 1, 2, or 3)")

	uplo_c := cast(u8)uplo
	itype_blas := Blas_Int(itype)
	n_int := Blas_Int(n)
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssygst_(&itype_blas, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsygst_(&itype_blas, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

reduce_generalized_symmetric_to_standard_form :: proc {
	reduce_generalized_symmetric_to_standard_form_real,
}

// ============================================================================
// SYMMETRIC TRIDIAGONAL REDUCTION (SYTRD)
// ============================================================================
// Reduce symmetric matrix to tridiagonal form using orthogonal similarity transformations

// Query workspace for symmetric tridiagonal reduction
query_workspace_reduce_symmetric_to_tridiagonal :: proc {
	query_workspace_reduce_symmetric_to_tridiagonal_real,
	query_workspace_reduce_symmetric_to_tridiagonal_complex,
}

query_workspace_reduce_symmetric_to_tridiagonal_real :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	}

	return work_size
}

query_workspace_reduce_symmetric_to_tridiagonal_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.csytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zsytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Reduce symmetric matrix to tridiagonal form (real)
reduce_symmetric_to_tridiagonal_real :: proc(
	A: ^Matrix($T), // Input/output: Symmetric matrix, overwritten with Q
	D: []T, // Output: Diagonal elements (length n)
	E: []T, // Output: Off-diagonal elements (length n-1)
	tau: []T, // Output: Scalar factors of elementary reflectors (length n-1)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(D) >= n, "Diagonal array too small")
	assert(len(E) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Reduce symmetric matrix to tridiagonal form (complex - symmetric, not Hermitian)
reduce_symmetric_to_tridiagonal_complex :: proc(
	A: ^Matrix($Cmplx), // Input/output: Symmetric matrix, overwritten with Q
	D: []Cmplx, // Output: Diagonal elements (length n)
	E: []Cmplx, // Output: Off-diagonal elements (length n-1)
	tau: []Cmplx, // Output: Scalar factors of elementary reflectors (length n-1)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(D) >= n, "Diagonal array too small")
	assert(len(E) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.csytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zsytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

reduce_symmetric_to_tridiagonal :: proc {
	reduce_symmetric_to_tridiagonal_real,
	reduce_symmetric_to_tridiagonal_complex,
}
