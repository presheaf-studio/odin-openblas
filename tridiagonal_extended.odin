package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - EXTENDED MRRR WITH TRYRAC
// ============================================================================
// Extended version of MRRR with additional control over algorithm selection

// Pre-allocated extended MRRR functions
m_compute_tridiagonal_extended_mrrr :: proc {
	m_compute_tridiagonal_extended_mrrr_f32_c64,
	m_compute_tridiagonal_extended_mrrr_f64_c128,
}

// Query workspace for extended MRRR (STEMR)
query_workspace_extended_mrrr :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, iwork_size: int) {
	// Query LAPACK for optimal workspace sizes
	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)EigenRangeOption.ALL

	// Dummy values for workspace query
	dummy_d := [1]f64{}
	dummy_e := [1]f64{}
	dummy_w := [1]f64{}
	vl: f64 = 0
	vu: f64 = 0
	il_int := Blas_Int(1)
	iu_int := Blas_Int(n)
	m_int: Blas_Int
	nzc_int: Blas_Int = 0
	tryrac_int: Blas_Int = 1
	ldz: Blas_Int = 1

	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.sstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			cast(^f32)&dummy_d[0],
			cast(^f32)&dummy_e[0],
			cast(^f32)&vl,
			cast(^f32)&vu,
			&il_int,
			&iu_int,
			&m_int,
			cast(^f32)&dummy_w[0],
			nil, // Z
			&ldz,
			&nzc_int,
			nil, // isuppz
			&tryrac_int,
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

		lapack.dstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			&dummy_d[0],
			&dummy_e[0],
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			&dummy_w[0],
			nil, // Z
			&ldz,
			&nzc_int,
			nil, // isuppz
			&tryrac_int,
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

		lapack.cstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			cast(^f32)&dummy_d[0],
			cast(^f32)&dummy_e[0],
			cast(^f32)&vl,
			cast(^f32)&vu,
			&il_int,
			&iu_int,
			&m_int,
			cast(^f32)&dummy_w[0],
			nil, // Z
			&ldz,
			&nzc_int,
			nil, // isuppz
			&tryrac_int,
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

		lapack.zstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			&dummy_d[0],
			&dummy_e[0],
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			&dummy_w[0],
			nil, // Z
			&ldz,
			&nzc_int,
			nil, // isuppz
			&tryrac_int,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Compute extended MRRR for f32/c64
m_compute_tridiagonal_extended_mrrr_f32_c64 :: proc(
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
	nzc: int = 0, // Number of eigenvectors to compute (0 = automatic)
	tryrac: bool = true, // Try to achieve high relative accuracy
) -> (
	m: int,
	nzc_out: int,
	info: Info,
	ok: bool, // Number of eigenvalues found// Number of columns in Z actually used
) where T == f32 || T == complex64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)EigenRangeOption.ALL

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		max_cols := nzc > 0 ? nzc : n
		assert(Z.rows >= n && Z.cols >= max_cols, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_AND_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f32 {
		lapack.sstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			isuppz_ptr,
			&tryrac_int,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
		)
	} else when T == complex64 {
		lapack.cstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			isuppz_ptr,
			&tryrac_int,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
		)
	}

	m = int(m_int)
	nzc_out = int(nzc_int)
	return m, nzc_out, info, info == 0
}

// Compute extended MRRR for f64/c128
m_compute_tridiagonal_extended_mrrr_f64_c128 :: proc(
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
	nzc: int = 0, // Number of eigenvectors to compute (0 = automatic)
	tryrac: bool = true, // Try to achieve high relative accuracy
) -> (
	m: int,
	nzc_out: int,
	info: Info,
	ok: bool, // Number of eigenvalues found// Number of columns in Z actually used
) where T == f64 || T == complex128 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)EigenRangeOption.ALL

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		max_cols := nzc > 0 ? nzc : n
		assert(Z.rows >= n && Z.cols >= max_cols, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_AND_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f64 {
		lapack.dstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			isuppz_ptr,
			&tryrac_int,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
		)
	} else when T == complex128 {
		lapack.zstemr_(
			&jobz_c,
			&range_c,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			isuppz_ptr,
			&tryrac_int,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
		)
	}

	m = int(m_int)
	nzc_out = int(nzc_int)
	return m, nzc_out, info, info == 0
}


// ============================================================================
// TRIDIAGONAL QR ITERATION
// ============================================================================
// Classic QR algorithm for tridiagonal eigenproblems

// Pre-allocated QR iteration functions
m_compute_tridiagonal_qr :: proc {
	m_compute_tridiagonal_qr_f32_c64,
	m_compute_tridiagonal_qr_f64_c128,
}

// Query workspace for QR iteration (STEQR)
query_workspace_qr_iteration :: proc($T: typeid, n: int, compz: CompzOption) -> (work_size: int) {
	// STEQR requires 2*n-2 real workspace if eigenvectors are computed
	if compz == .None {
		return 0
	}
	return 2 * n - 2
}

// Compute QR iteration for f32/c64
m_compute_tridiagonal_qr_f32_c64 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []f32 = nil, // Pre-allocated workspace (2*n-2 if compz != None)
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Verify workspace
	if compz != .None && work != nil {
		assert(len(work) >= 2 * n - 2, "Insufficient workspace")
	}

	when T == f32 {
		lapack.ssteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	} else when T == complex64 {
		lapack.csteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	}

	return info, info == 0
}

// Compute QR iteration for f64/c128
m_compute_tridiagonal_qr_f64_c128 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []f64 = nil, // Pre-allocated workspace (2*n-2 if compz != None)
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Verify workspace
	if compz != .None && work != nil {
		assert(len(work) >= 2 * n - 2, "Insufficient workspace")
	}

	when T == f64 {
		lapack.dsteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	} else when T == complex128 {
		lapack.zsteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	}

	return info, info == 0
}
