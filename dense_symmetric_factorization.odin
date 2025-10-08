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

// Query workspace for symmetric factorization (SYTRF)
query_workspace_dns_factorize_symmetric :: proc(A: ^Matrix($T), uplo: MatrixRegion) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.cols
	uplo_c := cast(u8)uplo
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	when T == f32 {
		lapack.ssytrf_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dsytrf_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		lapack.csytrf_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		lapack.zsytrf_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using Bunch-Kaufman diagonal pivoting
dns_factorize_symmetric :: proc(
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
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsytrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using symmetric factorization from SYTRF
dns_solve_symmetric_factorized :: proc(
	A: ^Matrix($T), // Factorized matrix from dns_factorize_symmetric
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
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.csytrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zsytrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// ROOK PIVOTING FACTORIZATION - Enhanced numerical stability
// ============================================================================

// Query workspace for symmetric Rook factorization (SYTRF_ROOK)
query_workspace_dns_factorize_symmetric_rook :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.cols
	uplo_c := cast(u8)uplo
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	when T == f32 {
		lapack.ssytrf_rook_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dsytrf_rook_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		lapack.csytrf_rook_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		lapack.zsytrf_rook_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using Rook pivoting (enhanced stability)
dns_factorize_symmetric_rook :: proc(
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
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rook_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_rook_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csytrf_rook_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsytrf_rook_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using Rook factorization
dns_solve_symmetric_rook_factorized :: proc(
	A: ^Matrix($T), // Factorized matrix from dns_factorize_symmetric_rook
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
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs_rook_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_rook_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.csytrs_rook_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zsytrs_rook_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// RK FACTORIZATION - Bounded Bunch-Kaufman
// ============================================================================

// Query workspace for symmetric RK factorization (SYTRF_RK)
query_workspace_dns_factorize_symmetric_rk :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.cols
	uplo_c := cast(u8)uplo
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	when T == f32 {
		lapack.ssytrf_rk_(&uplo_c, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dsytrf_rk_(&uplo_c, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		lapack.csytrf_rk_(&uplo_c, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		lapack.zsytrf_rk_(&uplo_c, &n, nil, &lda, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Factorize symmetric matrix using RK pivoting (bounded Bunch-Kaufman)
dns_factorize_symmetric_rk :: proc(
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
	lda := A.ld
	lde := E.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rk_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrf_rk_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.csytrf_rk_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zsytrf_rk_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Solve system using RK factorization
dns_solve_symmetric_rk_factorized :: proc(
	A: ^Matrix($T), // Factorized matrix from dns_factorize_symmetric_rk
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
	lda := A.ld
	lde := E.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs_3_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsytrs_3_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.csytrs_3_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zsytrs_3_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(E.data), raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ============================================================================
// IMPROVED SOLVERS WITH WORKSPACE (SYTRS2)
// ============================================================================

// Solve system using improved algorithm (SYTRS2) with workspace
dns_solve_symmetric_factorized_improved :: proc(
	A: ^Matrix($T), // Factorized matrix from dns_factorize_symmetric
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
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssytrs2_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsytrs2_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex64 {
		lapack.csytrs2_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zsytrs2_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(work), &info)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC GENERALIZED EIGENVALUE REDUCTION (SYGST)
// ============================================================================
// Reduce generalized symmetric eigenvalue problem to standard form
// Transforms A*x = lambda*B*x into C*y = lambda*y where C = inv(U^T)*A*inv(U) or inv(L)*A*inv(L^T)

// Reduce generalized symmetric eigenvalue problem to standard form (real)
dns_reduce_symmetric_generalized :: proc(
	A: ^Matrix($T), // Input/output: Symmetric matrix A, overwritten with transformed matrix
	B: ^Matrix(T), // Input: Cholesky factor of B (from potrf)
	itype: int = 1, // Problem type: 1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x FIXME
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
	lda := A.ld
	ldb := B.ld

	when T == f32 {
		lapack.ssygst_(&itype_blas, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dsygst_(&itype_blas, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}
