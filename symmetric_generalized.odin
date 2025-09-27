package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// GENERALIZED SYMMETRIC EIGENVALUE PROBLEMS
// ============================================================================

// Problem type for generalized eigenvalue problems
GeneralizedProblemType :: enum {
	AX_LBX = 1, // A*x = λ*B*x
	ABX_LX = 2, // A*B*x = λ*x
	BAX_LX = 3, // B*A*x = λ*x
}

// ============================================================================
// REDUCTION TO STANDARD FORM
// ============================================================================

// Reduce generalized symmetric eigenvalue problem to standard form
m_reduce_symmetric_generalized :: proc(
	a: ^Matrix($T), // Matrix A (modified in place)
	b: ^Matrix(T), // Cholesky factored B from potrf
	itype := GeneralizedProblemType.AX_LBX,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssygst_(&itype_int, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsygst_(&itype_int, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, &info, 1)
	}

	ok = info == 0
	return info, ok
}


// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - QR ALGORITHM
// ============================================================================

// Query workspace for generalized symmetric eigenvalue problem
query_workspace_solve_symmetric_generalized :: proc($T: typeid, n: int, itype := GeneralizedProblemType.AX_LBX, jobz := EigenJobOption.VALUES_ONLY, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssygv_(&itype_int, &jobz_c, &uplo_c, &n_int, nil, &lda, nil, &ldb, nil, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsygv_(&itype_int, &jobz_c, &uplo_c, &n_int, nil, &lda, nil, &ldb, nil, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
	}

	return work_size
}

// Solve generalized symmetric eigenvalue problem
m_solve_symmetric_generalized :: proc(
	a: ^Matrix($T), // Matrix A (eigenvectors on output if jobz == VALUES_VECTORS)
	b: ^Matrix(T), // Matrix B (Cholesky factor on output)
	w: []T, // Eigenvalues (size n)
	work: []T, // Workspace
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssygv_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsygv_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	}

	return info, info == 0
}


// Query workspace for 2-stage generalized symmetric eigenvalue problem
query_workspace_solve_symmetric_generalized_2stage :: proc(
	$T: typeid,
	n: int,
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
) where is_float(T) {
	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssygv_2stage_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsygv_2stage_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	}

	return work_size
}

// Solve 2-stage generalized symmetric eigenvalue problem
m_solve_symmetric_generalized_2stage :: proc(
	a: ^Matrix($T), // Matrix A (eigenvectors on output if jobz == VALUES_VECTORS)
	b: ^Matrix(T), // Matrix B (Cholesky factor on output)
	w: []T, // Eigenvalues (size n)
	work: []T, // Workspace
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssygv_2stage_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsygv_2stage_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	}

	return info, info == 0
}


// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - DIVIDE AND CONQUER
// ============================================================================

// Query workspace for divide-and-conquer generalized symmetric eigenvalue problem
query_workspace_solve_symmetric_generalized_divide_conquer :: proc(
	$T: typeid,
	n: int,
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
	iwork_size: int,
) where is_float(T) {
	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssygvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsygvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Solve divide-and-conquer generalized symmetric eigenvalue problem
m_solve_symmetric_generalized_divide_conquer :: proc(
	a: ^Matrix($T), // Matrix A (eigenvectors on output if jobz == VALUES_VECTORS)
	b: ^Matrix(T), // Matrix B (Cholesky factor on output)
	w: []T, // Eigenvalues (size n)
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssygvd_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsygvd_(&itype_int, &jobz_c, &uplo_c, &n_int, a.data, &lda, b.data, &ldb, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	return info, info == 0
}


// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - BISECTION AND INVERSE ITERATION
// ============================================================================

// Query workspace for selective generalized symmetric eigenvalue problem
query_workspace_solve_symmetric_generalized_selective :: proc(
	$T: typeid,
	n: int,
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
) where is_float(T) {
	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info
	m: Blas_Int

	// Dummy values for range parameters
	vl, vu: T
	il := Blas_Int(1)
	iu := Blas_Int(n)
	abstol: T

	when T == f32 {
		work_query: f32
		lapack.ssygvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsygvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // b
			&ldb,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	}

	return work_size
}

// Solve selective generalized symmetric eigenvalue problem
m_solve_symmetric_generalized_selective :: proc(
	a: ^Matrix($T), // Matrix A (modified on output)
	b: ^Matrix(T), // Matrix B (modified on output)
	w: []T, // Eigenvalues (size n)
	z: ^Matrix(T), // Eigenvectors (if jobz == VALUES_VECTORS)
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace (size 5*n)
	ifail: []Blas_Int, // Failed indices (size n)
	vl: T, // Lower bound (if range == VALUE_RANGE)
	vu: T, // Upper bound (if range == VALUE_RANGE)
	il: int = 0, // Lower index (if range == INDEX_RANGE, 1-based)
	iu: int = 0, // Upper index (if range == INDEX_RANGE, 1-based)
	abstol: T, // Absolute tolerance
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	n_int := Blas_Int(n)
	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	lwork := Blas_Int(len(work))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.ld)
		z_ptr = raw_data(z.data)
	}

	when T == f32 {
		lapack.ssygvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsygvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	}

	m = int(m_int)
	return m, info, info == 0
}
