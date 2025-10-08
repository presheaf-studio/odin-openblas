package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ===================================================================================
// EXTENDED TRIDIAGONAL EIGENVALUE ALGORITHMS
// ===================================================================================
// Advanced eigenvalue algorithms including STEMR, STEQR, and STEGR

// ===================================================================================
// EXTENDED MRRR WITH TRYRAC (STEMR)
// ===================================================================================

// Query workspace for extended MRRR (STEMR)
query_workspace_tridiagonal_stemr :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, iwork_size: int) {
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

// Extended MRRR algorithm with control over accuracy
tridiagonal_extended_mrrr :: proc {
	trid_eigen_extended_mrrr_real,
	trid_eigen_extended_mrrr_complex,
}

// Extended MRRR for f32/f64
trid_eigen_extended_mrrr_real :: proc(
	d: []$T, // Diagonal (modified to eigenvalues)
	e: []T, // Off-diagonal (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	w: []T, // Pre-allocated eigenvalue array
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	nzc: int = 0, // Number of eigenvectors to compute (0 = automatic)
	tryrac: bool = true, // Try to achieve high relative accuracy
) -> (
	m: int,
	nzc_out: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

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
		lapack.sstemr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &m_int, raw_data(w), z_ptr, &ldz, &nzc_int, isuppz_ptr, &tryrac_int, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dstemr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &m_int, raw_data(w), z_ptr, &ldz, &nzc_int, isuppz_ptr, &tryrac_int, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	nzc_out = int(nzc_int)
	return m, nzc_out, info, info == 0
}

// Extended MRRR for c64/c128
trid_eigen_extended_mrrr_complex :: proc(
	d: []$R, // Diagonal (modified to eigenvalues, always real)
	e: []R, // Off-diagonal (destroyed, always real)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	w: []R, // Pre-allocated eigenvalue array (always real)
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []R, // Pre-allocated workspace (real for STEMR)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: R, // Lower bound (if range == VALUE)
	vu: R, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	nzc: int = 0, // Number of eigenvectors to compute (0 = automatic)
	tryrac: bool = true, // Try to achieve high relative accuracy
) -> (
	m: int,
	nzc_out: int,
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

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

	when T == complex64 {
		lapack.cstemr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &m_int, raw_data(w), z_ptr, &ldz, &nzc_int, isuppz_ptr, &tryrac_int, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == complex128 {
		lapack.zstemr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &m_int, raw_data(w), z_ptr, &ldz, &nzc_int, isuppz_ptr, &tryrac_int, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	nzc_out = int(nzc_int)
	return m, nzc_out, info, info == 0
}

// ===================================================================================
// TRIDIAGONAL QR ITERATION (STEQR)
// ===================================================================================

// Query workspace for QR iteration (STEQR)
query_workspace_tridiagonal_qr :: proc($T: typeid, n: int, compz: CompzOption) -> (work_size: int) {
	// STEQR requires 2*n-2 real workspace if eigenvectors are computed
	if compz == .None {
		return 0
	}
	return max(1, 2 * n - 2)
}

// QR iteration for tridiagonal eigenproblems
tridiagonal_qr_iteration :: proc {
	trid_eigen_qr_real,
	trid_eigen_qr_complex,
}

// QR iteration for f32/f64
trid_eigen_qr_real :: proc(
	d: []$T, // Diagonal (modified to eigenvalues on output)
	e: []T, // Off-diagonal (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	work: []T = nil, // Pre-allocated workspace (2*n-2 if compz != None)
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
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
		assert(len(work) >= max(1, 2 * n - 2), "Insufficient workspace")
	}

	when T == f32 {
		lapack.ssteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	} else when T == f64 {
		lapack.dsteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	}

	return info, info == 0
}

// QR iteration for c64/c128
trid_eigen_qr_complex :: proc(
	d: []$R, // Diagonal (modified to eigenvalues on output, always real)
	e: []R, // Off-diagonal (destroyed, always real)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []R = nil, // Pre-allocated workspace (2*n-2 if compz != None)
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
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
		assert(len(work) >= max(1, 2 * n - 2), "Insufficient workspace")
	}

	when T == complex64 {
		lapack.csteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	} else when T == complex128 {
		lapack.zsteqr_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work) if work != nil else nil, &info)
	}

	return info, info == 0
}

// ===================================================================================
// GENERAL TRIDIAGONAL MRRR (STEGR)
// ===================================================================================

// Query workspace for general MRRR algorithm (STEGR)
query_workspace_tridiagonal_stegr :: proc($T: typeid, n: int, jobz: EigenJobOption, range: EigenRangeOption) -> (work_size: int, iwork_size: int, isuppz_size: int) {
	// Query LAPACK for optimal workspace sizes
	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

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

	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)
		abstol_f32 := f32(abstol)

		lapack.sstegr_(
			&jobz_c,
			&range_c,
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
			&jobz_c,
			&range_c,
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
			&jobz_c,
			&range_c,
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
			&jobz_c,
			&range_c,
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
	if jobz == .VALUES_AND_VECTORS {
		isuppz_size = 2 * n // Maximum possible
	} else {
		isuppz_size = 0
	}

	return work_size, iwork_size, isuppz_size
}

// General MRRR algorithm for tridiagonal matrices
tridiagonal_general_mrrr :: proc {
	trid_eigen_general_mrrr_real,
	trid_eigen_general_mrrr_complex,
}

// General MRRR for f32/f64
trid_eigen_general_mrrr_real :: proc(
	d: []$T, // Diagonal (modified to eigenvalues)
	e: []T, // Off-diagonal (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional)
	w: []T, // Pre-allocated eigenvalue array
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance (0 = machine precision)
) -> (
	m: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_AND_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f32 {
		lapack.sstegr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dstegr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	return m, info, info == 0
}

// General MRRR for c64/c128
trid_eigen_general_mrrr_complex :: proc(
	d: []$R, // Diagonal (modified to eigenvalues, always real)
	e: []R, // Off-diagonal (destroyed, always real)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	w: []R, // Pre-allocated eigenvalue array (always real)
	isuppz: []Blas_Int = nil, // Pre-allocated support array (2*max_m)
	work: []R, // Pre-allocated workspace (real for STEGR)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	vl: R, // Lower bound (if range == VALUE)
	vu: R, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: R, // Absolute tolerance (0 = machine precision)
) -> (
	m: int,
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Support array pointer
	isuppz_ptr: ^Blas_Int = nil
	if jobz == .VALUES_AND_VECTORS && isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == complex64 {
		lapack.cstegr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == complex128 {
		lapack.zstegr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	m = int(m_int)
	return m, info, info == 0
}

// ===================================================================================
// INVERSE ITERATION FOR EIGENVECTORS (STEIN)
// ===================================================================================

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
	when is_float(T) || is_complex(T) {
		work_size = 5 * n
		iwork_size = n
		ifail_size = m
	}
	return work_size, iwork_size, ifail_size
}

// Compute eigenvectors using inverse iteration
tridiagonal_inverse_iteration :: proc {
	trid_eigen_inverse_iteration_real,
	trid_eigen_inverse_iteration_complex,
}

// Inverse iteration for f32/f64
trid_eigen_inverse_iteration_real :: proc(
	d: []$T, // Diagonal elements
	e: []T, // Off-diagonal elements
	w: []T, // Eigenvalues
	iblock: []Blas_Int, // Block indices from stebz
	isplit: []Blas_Int, // Split points from stebz
	Z: ^Matrix(T), // Eigenvector matrix (output)
	work: []T, // Pre-allocated workspace (5*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (n)
	ifail: []Blas_Int, // Pre-allocated failed indices (m)
) -> (
	nfailed: int,
	info: Info,
	ok: bool,
) where is_float(T) {
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
	ldz := Z.ld

	when T == f32 {
		lapack.sstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	nfailed = int(info)
	return nfailed, info, info == 0
}

// Inverse iteration for c64/c128
trid_eigen_inverse_iteration_complex :: proc(
	d: []$R, // Diagonal elements (always real)
	e: []R, // Off-diagonal elements (always real)
	w: []R, // Eigenvalues (always real)
	iblock: []Blas_Int, // Block indices from stebz
	isplit: []Blas_Int, // Split points from stebz
	Z: ^Matrix($T), // Eigenvector matrix (output)
	work: []R, // Pre-allocated workspace (5*n, real)
	iwork: []Blas_Int, // Pre-allocated integer workspace (n)
	ifail: []Blas_Int, // Pre-allocated failed indices (m)
) -> (
	nfailed: int,
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
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
	ldz := Z.ld

	when T == complex64 {
		lapack.cstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == complex128 {
		lapack.zstein_(&n_int, raw_data(d), raw_data(e), &m_int, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(Z.data), &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	nfailed = int(info)
	return nfailed, info, info == 0
}
