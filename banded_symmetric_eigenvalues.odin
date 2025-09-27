package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// Banded eigenvalue result structure
BandedEigenResult :: struct($T: typeid) {
	eigenvalues:      []T,
	eigenvectors:     Matrix(T),
	min_eigenvalue:   f64,
	max_eigenvalue:   f64,
	condition_number: f64,
	num_negative:     int,
	num_zero:         int,
	num_positive:     int,
	all_positive:     bool,
	all_non_negative: bool,
}

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE COMPUTATION
// ============================================================================
// Computes all eigenvalues and optionally eigenvectors of a real symmetric
// band matrix using the divide and conquer algorithm

// Query workspace for banded eigenvalue computation (SBEV)
query_workspace_banded_symmetric_eigenvalues :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 {
	// SBEV requires 3*n-2 workspace
	work_size = 3 * n - 2
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Compute banded eigenvalues for f32/f64
m_compute_banded_symmetric_eigenvalues_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace (3*n-2)
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n - 2 || n <= 0, "Insufficient workspace")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbev_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	} else when T == f64 {
		lapack.dsbev_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// 2-STAGE SYMMETRIC BANDED EIGENVALUE COMPUTATION
// ============================================================================
// Uses a 2-stage algorithm for improved performance on large matrices

// Query workspace for 2-stage banded eigenvalue computation (SBEV_2STAGE)
query_workspace_banded_symmetric_eigenvalues_2stage :: proc($T: typeid, n: int, kd: int, jobz: EigenJobOption, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(kd + 1)
	ldz := Blas_Int(1)
	lwork := Blas_Int(-1)
	info: Info

	when T == f32 {
		work_query: f32

		lapack.ssbev_2stage_(
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
			&info,
			1,
			1,
		)

		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64

		lapack.dsbev_2stage_(
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
			&info,
			1,
			1,
		)

		work_size = int(work_query)
	}

	return work_size
}

// Compute 2-stage banded eigenvalues for f32/f64
m_compute_banded_eigenvalues_symmetric_2stage_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	kd: int = 0, // Number of superdiagonals/subdiagonals
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)
	lwork := Blas_Int(len(work))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbev_2stage_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsbev_2stage_(&jobz_c, &uplo_c, &n_int, &kd_int, raw_data(ab.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}
