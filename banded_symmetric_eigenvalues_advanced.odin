package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// Selective eigenvalue result
SelectiveEigenResult :: struct($T: typeid) {
	eigenvalues:    []T, // Selected eigenvalues
	eigenvectors:   Matrix(T), // Selected eigenvectors
	num_found:      int, // Number of eigenvalues found
	failed_indices: []Blas_Int, // Indices of eigenvectors that failed to converge
	num_failures:   int, // Number of convergence failures
	all_converged:  bool, // True if all eigenvectors converged
}

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================
// Uses divide-and-conquer algorithm for faster computation on large matrices

// Pre-allocated divide-and-conquer banded eigenvalue functions

// Query workspace for divide-and-conquer banded eigenvalue computation (SBEVD)
query_workspace_banded_symmetric_eigenvalues_dc :: proc($T: typeid, n: int, kd: int, jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace sizes
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(kd + 1)
	ldz := Blas_Int(1)
	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssbevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // w
			nil, // z
			&ldz,
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

		lapack.dsbevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // w
			nil, // z
			&ldz,
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

// Compute divide-and-conquer banded eigenvalues for f32/f64
m_compute_banded_symmetric_eigenvalues_dc_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbevd_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsbevd_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	return info, info == 0
}

// Query workspace for 2-stage divide-and-conquer (SBEVD_2STAGE)
query_workspace_banded_eigenvalues_dc_2stage :: proc($T: typeid, n: int, kd: int, jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) {
	// Query LAPACK for optimal workspace sizes
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(kd + 1)
	ldz := Blas_Int(1)
	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssbevd_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // w
			nil, // z
			&ldz,
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

		lapack.dsbevd_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // w
			nil, // z
			&ldz,
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

// Compute 2-stage divide-and-conquer banded eigenvalues for f32/f64
m_compute_banded_eigenvalues_dc_2stage_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := ab.cols
	assert(int(ab.rows) >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbevd_2stage_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsbevd_2stage_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE - SELECTIVE COMPUTATION
// ============================================================================
// Computes selected eigenvalues and optionally eigenvectors

// Query workspace for selective banded eigenvalue computation (SBEVX)
query_workspace_banded_eigenvalues_selective :: proc($T: typeid, n: int, kd: int, jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int, iwork_size: int) where is_float(T) {
	// SBEVX requires:
	// work: 7*n
	// iwork: 5*n
	return 7 * n, 5 * n
}

// Compute selective banded eigenvalues for f32/f64
m_compute_banded_eigenvalues_selective_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (preserved on output)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	Q: ^Matrix(T) = nil, // Orthogonal matrix from reduction (optional)
	work: []T, // Pre-allocated workspace (7*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (5*n)
	ifail: []Blas_Int, // Pre-allocated failure array (size n)
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 7 * n, "Insufficient workspace")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m: Blas_Int

	// Handle Q matrix
	ldq: Blas_Int = 1
	q_ptr: rawptr = nil
	if Q != nil {
		assert(Q.rows >= n && Q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(Q.ld)
		q_ptr = raw_data(Q.data)
	}

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			q_ptr,
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsbevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			q_ptr,
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	}

	num_found = int(m)
	return num_found, info, info == 0
}

// Query workspace for 2-stage selective (SBEVX_2STAGE)
query_workspace_banded_eigenvalues_selective_2stage :: proc(
	$T: typeid,
	n: int,
	kd: int,
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
	iwork_size: int,
) where is_float(T) {
	// Query LAPACK for optimal workspace
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(kd + 1)
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	vl: T = 0
	vu: T = 1
	il_int := Blas_Int(1)
	iu_int := Blas_Int(1)
	abstol: T = 0
	m: Blas_Int
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32

		lapack.ssbevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // q
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
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

		lapack.dsbevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			nil, // ab
			&ldab,
			nil, // q
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
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

	// iwork is always 5*n for SBEVX_2STAGE
	iwork_size = 5 * n
	return work_size, iwork_size
}

// Compute 2-stage selective banded eigenvalues for f32/f64
m_compute_banded_eigenvalues_selective_2stage_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (preserved on output)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	Q: ^Matrix(T) = nil, // Orthogonal matrix from reduction (optional)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (5*n)
	ifail: []Blas_Int, // Pre-allocated failure array (size n)
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	lwork := Blas_Int(len(work))
	m: Blas_Int

	// Handle Q matrix
	ldq: Blas_Int = 1
	q_ptr: rawptr = nil
	if Q != nil {
		assert(Q.rows >= n && Q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(Q.ld)
		q_ptr = raw_data(Q.data)
	}

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			q_ptr,
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m,
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
		lapack.dsbevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			q_ptr,
			&ldq,
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m,
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

	num_found = int(m)
	return num_found, info, info == 0
}
