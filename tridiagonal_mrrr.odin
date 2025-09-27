package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

m_compute_eigenvectors_inverse_iter :: proc {
	m_compute_eigenvectors_inverse_iter_f32_c64,
	m_compute_eigenvectors_inverse_iter_f64_c128,
}


// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - MRRR ALGORITHM
// ============================================================================
// Multiple Relatively Robust Representations - fastest algorithm for tridiagonal eigenproblems

// Query workspace for MRRR algorithm (STEGR)
query_workspace_mrrr :: proc($T: typeid, n: int, jobz: EigenJobOption, range: EigenRangeOption) -> (work_size: int, iwork_size: int, isuppz_size: int) {
	// Query LAPACK for optimal workspace sizes
	n_int := Blas_Int(n)
	jobz_c := eigen_job_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)

	// Dummy values for workspace query
	dummy_d := [1]f64{}
	dummy_e := [1]f64{}
	dummy_w := [1]f64{}
	vl: f64 = 0
	vu: f64 = 0
	il_int := Blas_Int(1)
	iu_int := Blas_Int(n)
	abstol: f64 = 0
	m_int: Blas_Int
	ldz: Blas_Int = 1

	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)
		abstol_f32 := f32(abstol)

		lapack.sstegr_(
			jobz_c,
			range_c,
			&n_int,
			cast(^f32)&dummy_d[0],
			cast(^f32)&dummy_e[0],
			&vl_f32,
			&vu_f32,
			&il_int,
			&iu_int,
			&abstol_f32,
			&m_int,
			cast(^f32)&dummy_w[0],
			nil, // Z
			&ldz,
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dstegr_(
			jobz_c,
			range_c,
			&n_int,
			&dummy_d[0],
			&dummy_e[0],
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			&dummy_w[0],
			nil, // Z
			&ldz,
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == complex64 {
		work_query: f32
		iwork_query: Blas_Int
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)
		abstol_f32 := f32(abstol)

		lapack.cstegr_(
			jobz_c,
			range_c,
			&n_int,
			cast(^f32)&dummy_d[0],
			cast(^f32)&dummy_e[0],
			&vl_f32,
			&vu_f32,
			&il_int,
			&iu_int,
			&abstol_f32,
			&m_int,
			cast(^f32)&dummy_w[0],
			nil, // Z
			&ldz,
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == complex128 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.zstegr_(
			jobz_c,
			range_c,
			&n_int,
			&dummy_d[0],
			&dummy_e[0],
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			&dummy_w[0],
			nil, // Z
			&ldz,
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	// Support array is 2*m where m is number of eigenvalues found
	if jobz == .VALUES_VECTORS {
		isuppz_size = 2 * n // Maximum possible
	} else {
		isuppz_size = 0
	}

	return work_size, iwork_size, isuppz_size
}

// Compute eigenvalues/eigenvectors using MRRR for f32/c64
m_compute_tridiagonal_mrrr_f32_c64 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues)
	e: []f32, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	w: []f32, // Pre-allocated eigenvalue array
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []f32, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance (0 = machine precision)
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == f32 || T == complex64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := eigen_job_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f32 {
		lapack.sstegr_(jobz_c, range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == complex64 {
		lapack.cstegr_(jobz_c, range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	ok = info == 0
	return m, info, ok
}

// Compute eigenvalues/eigenvectors using MRRR for f64/c128
m_compute_tridiagonal_mrrr_f64_c128 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues)
	e: []f64, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	w: []f64, // Pre-allocated eigenvalue array
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []f64, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance (0 = machine precision)
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == f64 || T == complex128 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := eigen_job_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f64 {
		lapack.dstegr_(jobz_c, range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == complex128 {
		lapack.zstegr_(jobz_c, range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	ok = info == 0
	return m, info, ok
}

// Proc group for MRRR algorithm
m_compute_tridiagonal_mrrr :: proc {
	m_compute_tridiagonal_mrrr_f32_c64,
	m_compute_tridiagonal_mrrr_f64_c128,
}

// ============================================================================
// TRIDIAGONAL INVERSE ITERATION
// ============================================================================
// Computes eigenvectors for given eigenvalues using inverse iteration

// Query workspace for inverse iteration (STEIN)
query_workspace_inverse_iteration :: proc(
	$T: typeid,
	n: int,
	m: int, // Number of eigenvalues
) -> (
	work_size: int,
	iwork_size: int,
	ifail_size: int,
) {
	// STEIN requires 5*n real workspace, n integer workspace, m ifail array
	when T == f32 || T == f64 || T == complex64 || T == complex128 {
		work_size = 5 * n
		iwork_size = n
		ifail_size = m
	}
	return work_size, iwork_size, ifail_size
}

// Compute eigenvectors using inverse iteration for f32/c64
m_compute_eigenvectors_inverse_iter_f32_c64 :: proc(
	d: []f32, // Diagonal elements
	e: []f32, // Off-diagonal elements
	w: []f32, // Eigenvalues
	iblock: []Blas_Int, // Block indices from stebz
	isplit: []Blas_Int, // Split points from stebz
	Z: ^Matrix($T), // Eigenvector matrix (output)
	work: []f32, // Pre-allocated workspace (5*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (n)
	ifail: []Blas_Int, // Pre-allocated failed indices (m)
) -> (
	nfailed: int,
	info: Info,
	ok: bool, // Number of failed eigenvectors
) where T == f32 || T == complex64 {
	n := len(d)
	m := len(w)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(int(Z.rows) >= n && int(Z.cols) >= m, "Eigenvector matrix too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= n, "Insufficient integer workspace")
	assert(len(ifail) >= m, "Insufficient ifail array")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(Z.ld)

	when T == f32 {
		lapack.sstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == complex64 {
		lapack.cstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	nfailed = int(info)
	ok = info == 0
	return nfailed, info, ok
}

// Compute eigenvectors using inverse iteration for f64/c128
m_compute_eigenvectors_inverse_iter_f64_c128 :: proc(
	d: []f64, // Diagonal elements
	e: []f64, // Off-diagonal elements
	w: []f64, // Eigenvalues
	iblock: []Blas_Int, // Block indices from stebz
	isplit: []Blas_Int, // Split points from stebz
	Z: ^Matrix($T), // Eigenvector matrix (output)
	work: []f64, // Pre-allocated workspace (5*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (n)
	ifail: []Blas_Int, // Pre-allocated failed indices (m)
) -> (
	nfailed: int,
	info: Info,
	ok: bool, // Number of failed eigenvectors
) where T == f64 || T == complex128 {
	n := len(d)
	m := len(w)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(int(Z.rows) >= n && int(Z.cols) >= m, "Eigenvector matrix too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= n, "Insufficient integer workspace")
	assert(len(ifail) >= m, "Insufficient ifail array")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(Z.ld)

	when T == f64 {
		lapack.dstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == complex128 {
		lapack.zstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	nfailed = int(info)
	ok = info == 0
	return nfailed, info, ok
}
