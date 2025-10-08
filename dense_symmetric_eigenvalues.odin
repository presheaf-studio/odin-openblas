package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - NON-ALLOCATING API
// ============================================================================
// Real symmetric matrices: all eigenvalues are real
// Using standard QR algorithm and more advanced methods

// Query workspace for symmetric eigenvalue computation (QR algorithm)
query_workspace_dns_eigen_symmetric :: proc(A: ^Matrix($T), jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssyev_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsyev_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
	}

	return work_size
}

// Compute eigenvalues and eigenvectors of symmetric matrix using QR algorithm
dns_eigen_symmetric :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed/overwritten with eigenvectors if requested)
	W: []T, // Output eigenvalues (length n)
	work: []T, // Pre-allocated workspace
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyev_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsyev_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// DIVIDE AND CONQUER ALGORITHM - More efficient for large matrices
// ============================================================================

// Query workspace for symmetric eigenvalue computation (Divide and Conquer)
query_workspace_dns_eigen_symmetric_dc :: proc(A: ^Matrix($T), jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info
	work_query: T
	iwork_query: Blas_Int

	when T == f32 {
		lapack.ssyevd_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	} else when T == f64 {
		lapack.dsyevd_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	}
	work_size = int(work_query)
	iwork_size = int(iwork_query)

	return work_size, iwork_size
}

// Compute eigenvalues and eigenvectors using Divide and Conquer algorithm
dns_eigen_symmetric_dc :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed/overwritten with eigenvectors if requested)
	W: []T, // Output eigenvalues (length n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lda := A.ld
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssyevd_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dsyevd_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// RELATIVELY ROBUST REPRESENTATIONS (RRR) - Most advanced algorithm
// ============================================================================

// Query workspace for symmetric eigenvalue computation (RRR algorithm)
query_workspace_dns_eigen_symmetric_mrrr :: proc(A: ^Matrix($T), Z: ^Matrix(T), range: EigenRangeOption, jobz: EigenJobOption, uplo: MatrixRegion) -> (work_size: int, iwork_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	ldz := Z.ld
	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	// Dummy values for query
	vl: T = 0
	vu: T = 0
	il: Blas_Int = 1
	iu: Blas_Int = n
	abstol: T = 0
	m: Blas_Int = 0
	work_query: T
	iwork_query: Blas_Int

	when T == f32 {
		lapack.ssyevr_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz,
			nil, // W, Z, ISUPPZ
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	} else when T == f64 {
		lapack.dsyevr_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz,
			nil, // W, Z, ISUPPZ
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	}
	work_size = int(work_query)
	iwork_size = int(iwork_query)

	return work_size, iwork_size
}

// Compute eigenvalues and eigenvectors using RRR algorithm (most robust)
dns_eigen_symmetric_mrrr :: proc(
	A: ^Matrix($T), // Input symmetric matrix (preserved)
	W: []T, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(T), // Output eigenvectors (n x m)
	ISUPPZ: []Blas_Int, // Support indices (length 2*max(1,m))
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	vl: T, // Value range lower bound (if range == 'V')
	vu: T, // Value range upper bound (if range == 'V')
	il: int = 1, // Index range lower bound (if range == 'I'), 1-based
	iu: int = 1, // Index range upper bound (if range == 'I'), 1-based
	abstol: T, // Absolute tolerance for eigenvalues
	range := EigenRangeOption.ALL,
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	lda := A.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int

	when T == f32 {
		lapack.ssyevr_(&jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(ISUPPZ), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dsyevr_(&jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(ISUPPZ), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return int(m_int), info, info == 0
}

// ============================================================================
// SELECTED EIGENVALUES - Legacy Expert Driver (SYEVX)
// ============================================================================

// Query workspace for symmetric eigenvalue computation (Expert driver)
query_workspace_dns_eigen_symmetric_expert :: proc(A: ^Matrix($T), range: EigenRangeOption, jobz: EigenJobOption, uplo: MatrixRegion) -> (work_size: int, iwork_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n := Blas_Int(A.cols)
	lda := A.ld
	ldz := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	// Dummy values for query
	vl: T = 0
	vu: T = 0
	il: Blas_Int = 1
	iu: Blas_Int = n
	abstol: T = 0
	m: Blas_Int = 0
	work_query: T

	when T == f32 {
		lapack.ssyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			nil,
			nil, // iwork, IFAIL
			&info,
		)
	} else when T == f64 {
		lapack.dsyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			nil,
			nil, // iwork, IFAIL
			&info,
		)
	}
	work_size = int(work_query)
	iwork_size = 5 * n // Fixed size for SYEVX

	return work_size, iwork_size
}

// Compute selected eigenvalues and eigenvectors using expert driver
dns_eigen_symmetric_expert :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed)
	W: []T, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(T), // Output eigenvectors (n x m)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	IFAIL: []Blas_Int, // Failed convergence indices (length n)
	vl: T, // Value range lower bound (if range == 'V')
	vu: T, // Value range upper bound (if range == 'V')
	il: int = 1, // Index range lower bound (if range == 'I'), 1-based
	iu: int = 1, // Index range upper bound (if range == 'I'), 1-based
	abstol: T, // Absolute tolerance for eigenvalues
	range := EigenRangeOption.ALL,
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(IFAIL) >= n, "IFAIL array too small")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n := Blas_Int(n)
	lda := A.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int

	when T == f32 {
		lapack.ssyevx_(&jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(iwork), raw_data(IFAIL), &info)
	} else when T == f64 {
		lapack.dsyevx_(&jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(iwork), raw_data(IFAIL), &info)
	}

	return int(m_int), info, info == 0
}

// ============================================================================
// TWO-STAGE ALGORITHMS - For very large matrices
// ============================================================================

// Query workspace for two-stage symmetric eigenvalue computation
query_workspace_dns_eigen_symmetric_2stage :: proc(A: ^Matrix($T), jobz: EigenJobOption, uplo: MatrixRegion) -> (work_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	when T == f32 {
		lapack.ssyev_2stage_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dsyev_2stage_(
			&jobz_c,
			&uplo_c,
			&n,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
	}
	work_size = int(work_query)

	return work_size
}

// Compute eigenvalues and eigenvectors using two-stage algorithm
dns_eigen_symmetric_2stage :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed/overwritten with eigenvectors if requested)
	W: []T, // Output eigenvalues (length n)
	work: []T, // Pre-allocated workspace
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyev_2stage_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsyev_2stage_(&jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED SYMMETRIC EIGENVALUE PROBLEMS
// ============================================================================
// Solve generalized eigenvalue problems of the form:
//   itype=1: A*x = lambda*B*x
//   itype=2: A*B*x = lambda*x
//   itype=3: B*A*x = lambda*x
// where A is symmetric and B is symmetric positive definite

// Query workspace for generalized symmetric eigenvalue computation (QR algorithm)
query_workspace_dns_eigen_symmetric_generalized :: proc(A: ^Matrix($T), B: ^Matrix(T), jobz: EigenJobOption, uplo: MatrixRegion) -> (work_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	ldb := B.ld
	lwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)
	work_query: T

	when T == f32 {
		lapack.ssygv_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dsygv_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&info,
		)
	}
	work_size = int(work_query)

	return work_size
}

// Compute generalized eigenvalues and eigenvectors using QR algorithm
dns_eigen_symmetric_generalized :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed/overwritten with eigenvectors if requested)
	B: ^Matrix(T), // Input symmetric positive definite matrix (destroyed)
	W: []T, // Output eigenvalues (length n)
	work: []T, // Pre-allocated workspace
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x) FIXME
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))
	itype_blas := Blas_Int(itype)

	when T == f32 {
		lapack.ssygv_(&itype_blas, &jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsygv_(&itype_blas, &jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED DIVIDE AND CONQUER ALGORITHM
// ============================================================================

// Query workspace for generalized symmetric eigenvalue computation (Divide and Conquer)
query_workspace_dns_eigen_symmetric_generalized_dc :: proc(A: ^Matrix($T), B: ^Matrix(T), jobz: EigenJobOption, uplo: MatrixRegion) -> (work_size: int, iwork_size: int) where is_float(T) {
	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := A.cols
	lda := A.ld
	ldb := B.ld
	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)
	work_query: T
	iwork_query: Blas_Int

	when T == f32 {
		lapack.ssygvd_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	} else when T == f64 {
		lapack.dsygvd_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)
	}
	work_size = int(work_query)
	iwork_size = int(iwork_query)

	return work_size, iwork_size
}

// Compute generalized eigenvalues and eigenvectors using Divide and Conquer algorithm
dns_eigen_symmetric_generalized_dc :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed/overwritten with eigenvectors if requested)
	B: ^Matrix(T), // Input symmetric positive definite matrix (destroyed)
	W: []T, // Output eigenvalues (length n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x) FIXME
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n := Blas_Int(n)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))
	itype_blas := Blas_Int(itype)

	when T == f32 {
		lapack.ssygvd_(&itype_blas, &jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dsygvd_(&itype_blas, &jobz_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED EXPERT DRIVER (SYGVX)
// ============================================================================

// Query workspace for generalized symmetric eigenvalue computation (Expert driver)
query_workspace_dns_eigen_symmetric_generalized_expert :: proc(A: ^Matrix($T), B: ^Matrix(T), Z: ^Matrix(T), range: EigenRangeOption, jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n := Blas_Int(A.cols)
	lda := A.ld
	ldb := B.ld
	ldz := Z.ld
	lwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)

	// Dummy values for query
	vl: T = 0
	vu: T = 0
	il: Blas_Int = 1
	iu: Blas_Int = n
	abstol: T = 0
	m: Blas_Int = 0
	work_query: T

	when T == f32 {
		lapack.ssygvx_(
			&itype,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			nil,
			&ldb, // B
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			nil,
			nil, // iwork, IFAIL
			&info,
		)
	} else when T == f64 {
		lapack.dsygvx_(
			&itype,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n,
			nil,
			&lda, // A
			nil,
			&ldb, // B
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			nil,
			nil, // iwork, IFAIL
			&info,
		)
	}
	work_size = int(work_query)
	iwork_size = 5 * n // Fixed size for SYGVX

	return work_size, iwork_size
}

// Compute selected generalized eigenvalues and eigenvectors using expert driver
dns_eigen_symmetric_generalized_expert :: proc(
	A: ^Matrix($T), // Input symmetric matrix (destroyed)
	B: ^Matrix(T), // Input symmetric positive definite matrix (destroyed)
	W: []T, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(T), // Output eigenvectors (n x m)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	IFAIL: []Blas_Int, // Failed convergence indices (length n)
	vl: T, // Value range lower bound (if range == 'V')
	vu: T, // Value range upper bound (if range == 'V')
	il: int = 1, // Index range lower bound (if range == 'I'), 1-based
	iu: int = 1, // Index range upper bound (if range == 'I'), 1-based
	abstol: T, // Absolute tolerance for eigenvalues
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	range := EigenRangeOption.ALL,
	jobz := EigenJobOption.VALUES_AND_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where is_float(T) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(IFAIL) >= n, "IFAIL array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n := Blas_Int(n)
	lda := A.ld
	ldb := B.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	itype_blas := Blas_Int(itype)
	m_int: Blas_Int

	when T == f32 {
		lapack.ssygvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(iwork), raw_data(IFAIL), &info)
	} else when T == f64 {
		lapack.dsygvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(iwork), raw_data(IFAIL), &info)
	}

	return int(m_int), info, info == 0
}
