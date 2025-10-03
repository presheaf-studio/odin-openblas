package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// SYMMETRIC TRIDIAGONAL EIGENVALUE PROBLEMS (ST PREFIX)
// ===================================================================================
// Eigenvalue and eigenvector computation for symmetric tridiagonal matrices
// Non-allocating API with pre-allocated arrays

// ===================================================================================
// SIMPLE EIGENVALUE-ONLY COMPUTATION (STERF)
// ===================================================================================

// Compute eigenvalues only for symmetric tridiagonal matrix (no workspace needed)
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
tridiagonal_eigenvalues_only :: proc(
	d: []$T, // Diagonal elements (modified to eigenvalues on output)
	e: []T, // Off-diagonal elements (destroyed)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	when T == f32 {
		lapack.ssterf_(&n_int, raw_data(d), raw_data(e), &info)
	} else when T == f64 {
		lapack.dsterf_(&n_int, raw_data(d), raw_data(e), &info)
	}

	return info, info == 0
}


// ===================================================================================
// SIMPLE DRIVER (STEV)
// ===================================================================================

// Query workspace for simple eigenvalue/eigenvector computation
query_workspace_tridiagonal_simple :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int) {
	if compute_vectors {
		return max(1, 2 * n - 2)
	}
	return 0
}

// Compute eigenvalues and optionally eigenvectors using simple driver
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
tridiagonal_eigenvalues_simple :: proc(
	d: []$T, // Diagonal elements (modified to eigenvalues on output)
	e: []T, // Off-diagonal elements (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional output)
	work: []T = nil, // Pre-allocated workspace (needed if computing vectors)
	jobz := EigenJobOption.VALUES_ONLY,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
		assert(work != nil && len(work) >= max(1, 2 * n - 2), "Workspace required for eigenvectors")
	}

	jobz_c := cast(u8)jobz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .VALUES_AND_VECTORS {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	work_ptr: ^T = nil
	if work != nil {
		work_ptr = raw_data(work)
	}

	when T == f32 {
		lapack.sstev_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, work_ptr, &info)
	} else when T == f64 {
		lapack.dstev_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, work_ptr, &info)
	}

	return info, info == 0
}


// ===================================================================================
// DIVIDE AND CONQUER DRIVER (STEVD)
// ===================================================================================

// Query workspace for divide-and-conquer computation
query_workspace_tridiagonal_dc :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int, iwork_size: int) {
	n_blas := Blas_Int(n)
	jobz_c := cast(u8)(compute_vectors ? EigenJobOption.VALUES_AND_VECTORS : EigenJobOption.VALUES_ONLY)

	// Query LAPACK for optimal sizes
	work_query: T
	iwork_query: Blas_Int
	lwork := Blas_Int(-1)
	liwork := Blas_Int(-1)
	info: Info
	ldz := Blas_Int(1)

	when T == f32 {
		lapack.sstevd_(&jobz_c, &n_blas, nil, nil, nil, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info)
		return int(work_query), int(iwork_query)
	} else when T == f64 {
		lapack.dstevd_(&jobz_c, &n_blas, nil, nil, nil, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info)
		return int(work_query), int(iwork_query)
	}
}

// Compute eigenvalues/eigenvectors using divide-and-conquer
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
tridiagonal_eigenvalues_dc :: proc(
	d: []$T, // Diagonal elements (modified to eigenvalues on output)
	e: []T, // Off-diagonal elements (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	if jobz == .VALUES_AND_VECTORS {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	jobz_c := cast(u8)jobz
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .VALUES_AND_VECTORS {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sstevd_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dstevd_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}


// ===================================================================================
// MRRR DRIVER (STEVR)
// ===================================================================================

// Query workspace for MRRR computation
query_workspace_tridiagonal_mrrr :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int, iwork_size: int) {
	n_blas := Blas_Int(n)
	jobz_c := cast(u8)(compute_vectors ? EigenJobOption.VALUES_AND_VECTORS : EigenJobOption.VALUES_ONLY)
	range_c := cast(u8)EigenRangeOption.ALL

	// Query LAPACK for optimal sizes
	work_query: T
	iwork_query: Blas_Int
	lwork := Blas_Int(-1)
	liwork := Blas_Int(-1)
	info: Info
	ldz := Blas_Int(1)
	m_dummy: Blas_Int

	when T == f32 {
		vl_dummy: f32 = 0
		vu_dummy: f32 = 0
		il_dummy := Blas_Int(1)
		iu_dummy := Blas_Int(n)
		abstol_dummy: f32 = 0

		lapack.sstevr_(&jobz_c, &range_c, &n_blas, nil, nil, &vl_dummy, &vu_dummy, &il_dummy, &iu_dummy, &abstol_dummy, &m_dummy, nil, nil, &ldz, nil, &work_query, &lwork, &iwork_query, &liwork, &info)
		return int(work_query), int(iwork_query)
	} else when T == f64 {
		vl_dummy: f64 = 0
		vu_dummy: f64 = 0
		il_dummy := Blas_Int(1)
		iu_dummy := Blas_Int(n)
		abstol_dummy: f64 = 0

		lapack.dstevr_(&jobz_c, &range_c, &n_blas, nil, nil, &vl_dummy, &vu_dummy, &il_dummy, &iu_dummy, &abstol_dummy, &m_dummy, nil, nil, &ldz, nil, &work_query, &lwork, &iwork_query, &liwork, &info)
		return int(work_query), int(iwork_query)
	}
}

// Compute eigenvalues/eigenvectors using MRRR
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
tridiagonal_eigenvalues_mrrr :: proc(
	d: []$T, // Diagonal elements (modified)
	e: []T, // Off-diagonal elements (modified)
	w: []T, // Eigenvalues output (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional output)
	isuppz: []Blas_Int = nil, // Support indices (size 2*max(m))
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range := EigenRangeOption.ALL,
	vl: T = 0, // Lower bound (if range == VALUE)
	vu: T = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T = 0, // Absolute tolerance
	jobz := EigenJobOption.VALUES_ONLY,
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	if jobz == .VALUES_AND_VECTORS {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
		max_m := n // Conservative estimate
		assert(isuppz != nil && len(isuppz) >= 2 * max_m, "Support array required for eigenvectors")
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .VALUES_AND_VECTORS {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	isuppz_ptr: ^Blas_Int = nil
	if isuppz != nil {
		isuppz_ptr = raw_data(isuppz)
	}

	when T == f32 {
		lapack.sstevr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dstevr_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, isuppz_ptr, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	num_found = int(m)
	return num_found, info, info == 0
}


// ===================================================================================
// BISECTION AND INVERSE ITERATION (STEVX)
// ===================================================================================

// Query workspace for bisection and inverse iteration
query_workspace_tridiagonal_bisection :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
	// STEVX workspace requirements
	return 5 * n, 5 * n
}

// Compute eigenvalues/eigenvectors using bisection and inverse iteration
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
tridiagonal_eigenvalues_bisection :: proc(
	d: []$T, // Diagonal elements (preserved)
	e: []T, // Off-diagonal elements (preserved)
	w: []T, // Eigenvalues output (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional output)
	work: []T, // Pre-allocated workspace (size 5*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	ifail: []Blas_Int, // Failure indices (size n)
	range := EigenRangeOption.ALL,
	vl: T = 0, // Lower bound (if range == VALUE)
	vu: T = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T = 0, // Absolute tolerance
	jobz := EigenJobOption.VALUES_ONLY,
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Failure array too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(Z != nil && Z.rows >= Blas_Int(n) && Z.cols >= Blas_Int(n), "Eigenvector matrix required")
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .VALUES_AND_VECTORS {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sstevx_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dstevx_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	num_found = int(m)
	return num_found, info, info == 0
}


// ===================================================================================
// EIGENVALUE ANALYSIS UTILITIES
// ===================================================================================

// Helper to analyze eigenvalues after computation
analyze_tridiagonal_eigenvalues :: proc(eigenvalues: []$T) -> (min_val: f64, max_val: f64, condition_number: f64, all_positive: bool) {
	if len(eigenvalues) == 0 {
		return 0, 0, 1, false
	}

	// Eigenvalues are sorted by LAPACK
	when T == f32 {
		min_val = f64(eigenvalues[0])
		max_val = f64(eigenvalues[len(eigenvalues) - 1])
	} else {
		min_val = eigenvalues[0]
		max_val = eigenvalues[len(eigenvalues) - 1]
	}

	all_positive = eigenvalues[0] > 0

	// Compute condition number
	if abs(eigenvalues[0]) > machine_epsilon(T) {
		when T == f32 {
			condition_number = f64(abs(eigenvalues[len(eigenvalues) - 1] / eigenvalues[0]))
		} else {
			condition_number = abs(eigenvalues[len(eigenvalues) - 1] / eigenvalues[0])
		}
	} else {
		condition_number = math.INF_F64
	}

	return min_val, max_val, condition_number, all_positive
}

// Helper to get machine epsilon
machine_epsilon :: proc($T: typeid) -> T {
	when T == f32 {
		return math.F32_EPSILON
	} else when T == f64 {
		return math.F64_EPSILON
	} else {
		#panic("Unsupported type for machine epsilon")
	}
}

// ===================================================================================
// TRIDIAGONAL DIVIDE-AND-CONQUER WITH COMPLEX SUPPORT (STEDC)
// ===================================================================================

// Query workspace for complex divide-and-conquer computation
query_workspace_tridiagonal_stedc :: proc($T: typeid, n: int, compz: CompzOption) -> (work_size: int, rwork_size: int, iwork_size: int) {
	n_int := Blas_Int(n)
	compz_c := cast(u8)compz

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.sstedc_(
			&compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		return int(work_query), 0, int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.dstedc_(
			&compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
		)

		return int(work_query), 0, int(iwork_query)
	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		lrwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.cstedc_(
			&compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)

		return int(real(work_query)), int(rwork_query), int(iwork_query)
	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		lrwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.zstedc_(
			&compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)

		return int(real(work_query)), int(rwork_query), int(iwork_query)
	}
}

// Compute eigenvalues/eigenvectors using divide-and-conquer for all types
tridiagonal_eigenvalues_stedc :: proc {
	tridiagonal_eigenvalues_stedc_f32_f64,
	tridiagonal_eigenvalues_stedc_c64_c128,
}

// Compute eigenvalues/eigenvectors using divide-and-conquer for f32/f64
tridiagonal_eigenvalues_stedc_f32_f64 :: proc(
	d: []$R, // Diagonal (modified to eigenvalues on output, always real)
	e: []R, // Off-diagonal (destroyed, always real)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	rwork: []R = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where (T == f32 && R == f32) || (T == f64 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		assert(len(work) > 0, "Workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.sstedc_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		assert(len(work) > 0, "Workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.dstedc_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// Compute eigenvalues/eigenvectors using divide-and-conquer for c64/c128
tridiagonal_eigenvalues_stedc_c64_c128 :: proc(
	d: []$R, // Diagonal (modified to eigenvalues on output, always real)
	e: []R, // Off-diagonal (destroyed, always real)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	rwork: []R, // Pre-allocated real workspace (required for complex)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := cast(u8)compz
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == complex64 {
		assert(len(work) > 0, "Workspace required")
		assert(len(rwork) > 0, "Real workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.cstedc_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when T == complex128 {
		assert(len(work) > 0, "Workspace required")
		assert(len(rwork) > 0, "Real workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.zstedc_(&compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// EIGENVALUE SELECTION BY BISECTION (STEBZ)
// ===================================================================================

// Eigenvalue ordering option
EigenvalueOrder :: enum u8 {
	BLOCKS = 'B', // Order eigenvalues by blocks
	ENTIRE = 'E', // Order eigenvalues for entire matrix
}

// Query workspace for eigenvalue bisection
query_workspace_eigenvalue_bisection :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
	return 4 * n, 3 * n
}

// Compute selected eigenvalues using bisection
// Note: Symmetric tridiagonal eigenvalues are always real, so this only supports f32/f64
eigenvalue_bisection :: proc(
	d: []$T, // Diagonal elements
	e: []T, // Off-diagonal elements
	w: []T, // Eigenvalues output
	iblock: []Blas_Int, // Block indices output
	isplit: []Blas_Int, // Split points output
	work: []T, // Pre-allocated workspace (size 4*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 3*n)
	range := EigenRangeOption.ALL,
	order := EigenvalueOrder.ENTIRE,
	vl: T = 0, // Lower bound (if range == VALUE)
	vu: T = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T = 0, // Absolute tolerance
) -> (
	num_found: int,
	num_splits: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(iblock) >= n, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")
	assert(len(iwork) >= 3 * n, "Insufficient integer workspace")

	range_c := cast(u8)range
	order_c := cast(u8)order
	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	nsplit: Blas_Int

	when T == f32 {
		lapack.sstebz_(&range_c, &order_c, &n_int, &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, raw_data(d), raw_data(e), &m, &nsplit, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dstebz_(&range_c, &order_c, &n_int, &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, raw_data(d), raw_data(e), &m, &nsplit, raw_data(w), raw_data(iblock), raw_data(isplit), raw_data(work), raw_data(iwork), &info)
	}

	num_found = int(m)
	num_splits = int(nsplit)
	return num_found, num_splits, info, info == 0
}
