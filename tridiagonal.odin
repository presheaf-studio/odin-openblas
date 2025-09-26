package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SIMPLE TRIDIAGONAL EIGENVALUE COMPUTATION
// ============================================================================
compute_tridiagonal_eigenvalues :: proc {
	compute_tridiagonal_eigenvalues_f32,
	compute_tridiagonal_eigenvalues_f64,
	compute_tridiagonal_eigenvalues_from_matrix,
}

compute_tridiagonal_eigenvectors :: proc {
	compute_tridiagonal_eigenvectors_f32,
	compute_tridiagonal_eigenvectors_f64,
	compute_tridiagonal_eigenvectors_c64,
	compute_tridiagonal_eigenvectors_c128,
}


compute_tridiagonal_dc :: proc {
	compute_tridiagonal_dc_f32,
	compute_tridiagonal_dc_f64,
	compute_tridiagonal_dc_c64,
	compute_tridiagonal_dc_c128,
}

// Helper to analyze eigenvalues after computation
analyze_eigenvalues :: proc(
	eigenvalues: []$T,
) -> (
	min_val: f64,
	max_val: f64,
	condition_number: f64,
	all_positive: bool,
) {
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

// Compute tridiagonal eigenvalues for f32
compute_tridiagonal_eigenvalues_f32 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (destroyed)
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	// Call LAPACK
	lapack.ssterf_(&n_int, raw_data(d), raw_data(e), &info)

	return info, info == 0
}

// Compute tridiagonal eigenvalues for f64
compute_tridiagonal_eigenvalues_f64 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (destroyed)
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	// Call LAPACK
	lapack.dsterf_(&n_int, raw_data(d), raw_data(e), &info)

	return info, info == 0
}

// Compute eigenvalues from tridiagonal matrix (extracts d,e internally)
// Works for both real and complex matrices since Hermitian tridiagonal has real elements
compute_tridiagonal_eigenvalues_from_matrix :: proc(
	M: ^Matrix($T), // Tridiagonal matrix (not modified)
	d: []$U, // Pre-allocated diagonal output (becomes eigenvalues)
	e: []U, // Pre-allocated off-diagonal workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := M.rows
	assert(M.rows == M.cols, "Matrix must be square")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	// FIXME: review diagonal
	// Extract diagonal and off-diagonal from matrix
	for i in 0 ..< n {
		when is_float(T) {
			d[i] = M.data[i * M.ld + i]
			if i < n - 1 {
				// Can extract from upper or lower diagonal
				e[i] = M.data[i * M.ld + i + 1] // Upper diagonal
				// e[i] = M.data[(i + 1) * M.ld + i] // Lower diagonal
			}
			return compute_tridiagonal_eigenvalues_f32(d, e)
		} else when is_complex(T) {
			// For complex Hermitian tridiagonal, diagonal is real part
			d[i] = real(M.data[i * M.ld + i])
			if i < n - 1 {
				// Off-diagonal is also real for Hermitian tridiagonal
				e[i] = real(M.data[i * M.ld + i + 1]) // Upper diagonal
				// e[i] = real(M.data[(i + 1) * M.ld + i]) // Lower diagonal
			}
			return compute_tridiagonal_eigenvalues_f64(d, e)
		}
	}

}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - SIMPLE DRIVER
// ============================================================================

// Query workspace for simple eigenvalue/eigenvector computation
query_workspace_tridiagonal_eigenvectors :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
) -> (
	work_size: int,
) {
	if compute_vectors {
		return 2 * n - 2
	}
	return 0
}

// Compute tridiagonal eigenvalues and optionally eigenvectors for f32
compute_tridiagonal_eigenvectors_f32 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (destroyed)
	Z: ^Matrix(f32) = nil, // Eigenvectors (optional output)
	work: []f32 = nil, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compute_vectors := Z != nil
	if compute_vectors {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		work_size := query_workspace_tridiagonal_eigenvectors(f32, n, true)
		assert(work == nil || len(work) >= work_size, "Insufficient workspace")
	}

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)

	n_int := Blas_Int(n)

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if compute_vectors {
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	lapack.sstev_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info,
		1,
	)

	return info, info == 0
}

// Compute tridiagonal eigenvalues and optionally eigenvectors for f64
compute_tridiagonal_eigenvectors_f64 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (destroyed)
	Z: ^Matrix(f64) = nil, // Eigenvectors (optional output)
	work: []f64 = nil, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compute_vectors := Z != nil
	if compute_vectors {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		work_size := query_workspace_tridiagonal_eigenvectors(f64, n, true)
		assert(work == nil || len(work) >= work_size, "Insufficient workspace")
	}

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)

	n_int := Blas_Int(n)

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if compute_vectors {
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	// Call LAPACK
	lapack.dstev_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info,
		1,
	)

	return info, info == 0
}

// Complex versions just call the real versions (Hermitian tridiagonal has real eigenvalues)
compute_tridiagonal_eigenvectors_c64 :: proc(
	d: []f32, // Real diagonal (modified to eigenvalues on output)
	e: []f32, // Real off-diagonal (destroyed)
	Z: ^Matrix(complex64) = nil, // Eigenvectors (optional output)
	work: []f32 = nil, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	// For complex Hermitian, we need a real matrix for eigenvectors
	// then convert to complex if needed
	if Z != nil {
		zptr := cast([^]f32)cast(rawptr)&Z.data[0]
		// Create real eigenvector matrix
		Z_real := Matrix(f32) {
			data = zptr[:2 * len(Z.data)], // Reinterpret complex as pairs of reals
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2, // Complex has twice the ld
		}
		return compute_tridiagonal_eigenvectors_f32(d, e, &Z_real, work)
	}
	return compute_tridiagonal_eigenvectors_f32(d, e, nil, work)
}

compute_tridiagonal_eigenvectors_c128 :: proc(
	d: []f64, // Real diagonal (modified to eigenvalues on output)
	e: []f64, // Real off-diagonal (destroyed)
	Z: ^Matrix(complex128) = nil, // Eigenvectors (optional output)
	work: []f64 = nil, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	// For complex Hermitian, we need a real matrix for eigenvectors
	// then convert to complex if needed
	if Z != nil {
		zptr := cast([^]f64)cast(rawptr)&Z.data[0]
		// Create real eigenvector matrix
		Z_real := Matrix(f64) {
			data = zptr[:2 * len(Z.data)], // Reinterpret complex as pairs of reals
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2, // Complex has twice the ld
		}
		return compute_tridiagonal_eigenvectors_f64(d, e, &Z_real, work)
	}
	return compute_tridiagonal_eigenvectors_f64(d, e, nil, work)
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - DIVIDE AND CONQUER DRIVER
// ============================================================================

// Query workspace for divide and conquer eigenvalue/eigenvector computation
query_workspace_tridiagonal_dc :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
) -> (
	work_size: int,
	iwork_size: int,
) {
	n_blas := Blas_Int(n)

	// Create dummy arrays for query
	d_dummy: T
	e_dummy: T
	z_dummy: T
	work_query: T
	iwork_query: Blas_Int
	lwork := Blas_Int(-1)
	liwork := Blas_Int(-1)
	info: Info
	ldz := Blas_Int(1)

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)

	when T == f32 {
		lapack.sstevd_(
			jobz_cstring,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&z_dummy,
			&ldz,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dstevd_(
			jobz_cstring,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&z_dummy,
			&ldz,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
		)
	}

	return int(work_query), int(iwork_query)
}

// Compute tridiagonal eigenvalues/eigenvectors using divide and conquer for f32
compute_tridiagonal_dc_f32 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (destroyed)
	Z: ^Matrix(f32) = nil, // Eigenvectors (optional output)
	work: []f32, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compute_vectors := Z != nil
	if compute_vectors {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	// Verify workspace  TODO: WHEN OB_ASSERT CLAUSE
	work_size, iwork_size := query_workspace_tridiagonal_dc(f32, n, compute_vectors)
	assert(len(work) >= work_size, "Insufficient workspace")
	assert(len(iwork) >= iwork_size, "Insufficient integer workspace")

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)

	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if compute_vectors {
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	// Call LAPACK
	lapack.sstevd_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
	)

	ok = info == 0
	return info, ok
}

// Compute tridiagonal eigenvalues/eigenvectors using divide and conquer for f64
compute_tridiagonal_dc_f64 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (destroyed)
	Z: ^Matrix(f64) = nil, // Eigenvectors (optional output)
	work: []f64, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compute_vectors := Z != nil
	if compute_vectors {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	// Verify workspace
	work_size, iwork_size := query_workspace_tridiagonal_dc(f64, n, compute_vectors)
	assert(len(work) >= work_size, "Insufficient workspace")
	assert(len(iwork) >= iwork_size, "Insufficient integer workspace")

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)

	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if compute_vectors {
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	// Call LAPACK
	lapack.dstevd_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
	)

	ok = info == 0
	return info, ok
}

// Complex versions
compute_tridiagonal_dc_c64 :: proc(
	d: []f32, // Real diagonal (modified to eigenvalues on output)
	e: []f32, // Real off-diagonal (destroyed)
	Z: ^Matrix(complex64) = nil, // Eigenvectors (optional output)
	work: []f32, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	// For complex Hermitian, convert eigenvector matrix if needed
	if Z != nil {
		zptr := cast([^]f32)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f32) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_dc_f32(d, e, &Z_real, work, iwork)
	}
	return compute_tridiagonal_dc_f32(d, e, nil, work, iwork)
}

compute_tridiagonal_dc_c128 :: proc(
	d: []f64, // Real diagonal (modified to eigenvalues on output)
	e: []f64, // Real off-diagonal (destroyed)
	Z: ^Matrix(complex128) = nil, // Eigenvectors (optional output)
	work: []f64, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	// For complex Hermitian, convert eigenvector matrix if needed
	if Z != nil {
		zptr := cast([^]f64)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f64) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_dc_f64(d, e, &Z_real, work, iwork)
	}
	return compute_tridiagonal_dc_f64(d, e, nil, work, iwork)
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - MRRR DRIVER
// ============================================================================

// Query workspace for MRRR eigenvalue/eigenvector computation
query_workspace_tridiagonal_mrrr :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
	range: EigenRangeOption = .ALL,
) -> (
	work_size: int,
	iwork_size: int,
) {
	n_blas := Blas_Int(n)

	// Create dummy arrays for query
	d_dummy: T
	e_dummy: T
	vl_dummy: T
	vu_dummy: T
	il_dummy := Blas_Int(1)
	iu_dummy := Blas_Int(1)
	abstol_dummy: T
	m_dummy: Blas_Int
	w_dummy: T
	z_dummy: T
	ldz := Blas_Int(1)
	isuppz_dummy: Blas_Int
	work_query: T
	iwork_query: Blas_Int
	lwork := Blas_Int(-1)
	liwork := Blas_Int(-1)
	info: Info

	jobz_cstring := eigen_job_to_cstring(compute_vectors ? .VALUES_VECTORS : .VALUES_ONLY)
	range_cstring := eigen_range_to_cstring(range)

	when T == f32 {
		lapack.sstevr_(
			jobz_cstring,
			range_cstring,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&vl_dummy,
			&vu_dummy,
			&il_dummy,
			&iu_dummy,
			&abstol_dummy,
			&m_dummy,
			&w_dummy,
			&z_dummy,
			&ldz,
			&isuppz_dummy,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dstevr_(
			jobz_cstring,
			range_cstring,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&vl_dummy,
			&vu_dummy,
			&il_dummy,
			&iu_dummy,
			&abstol_dummy,
			&m_dummy,
			&w_dummy,
			&z_dummy,
			&ldz,
			&isuppz_dummy,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
	}

	return int(work_query), int(iwork_query)
}

// Compute tridiagonal eigenvalues/eigenvectors using MRRR for f32
compute_tridiagonal_mrrr_f32 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (modified)
	w: []f32, // Pre-allocated eigenvalues output
	Z: ^Matrix(f32) = nil, // Eigenvectors (optional output)
	isuppz: []Blas_Int = nil, // Pre-allocated support arrays (size 2*max_eigenvalues)
	work: []f32, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range: EigenRangeOption = .ALL,
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) > 0, "Work array required")
	assert(len(iwork) > 0, "Integer work array required")

	jobz := Z != nil ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(len(work))
	liwork_int := Blas_Int(len(iwork))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]

		// Support arrays required for eigenvectors
		max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
		assert(
			isuppz != nil && len(isuppz) >= 2 * max_m,
			"Support array required for eigenvectors",
		)
	}

	// Call LAPACK
	lapack.sstevr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info,
		len(jobz_cstring),
		len(range_cstring),
	)

	num_found = int(m)
	ok = info == 0
	return num_found, info, ok
}

// Compute tridiagonal eigenvalues/eigenvectors using MRRR for f64
compute_tridiagonal_mrrr_f64 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (modified)
	w: []f64, // Pre-allocated eigenvalues output
	Z: ^Matrix(f64) = nil, // Eigenvectors (optional output)
	isuppz: []Blas_Int = nil, // Pre-allocated support arrays (size 2*max_eigenvalues)
	work: []f64, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range: EigenRangeOption = .ALL,
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) > 0, "Work array required")
	assert(len(iwork) > 0, "Integer work array required")

	jobz := Z != nil ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(len(work))
	liwork_int := Blas_Int(len(iwork))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]

		// Support arrays required for eigenvectors
		max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
		assert(
			isuppz != nil && len(isuppz) >= 2 * max_m,
			"Support array required for eigenvectors",
		)
	}

	// Call LAPACK
	lapack.dstevr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info,
		len(jobz_cstring),
		len(range_cstring),
	)

	num_found = int(m)
	ok = info == 0
	return num_found, info, ok
}

// Compute tridiagonal eigenvalues/eigenvectors using MRRR for c64
compute_tridiagonal_mrrr_c64 :: proc(
	d: []f32, // Diagonal (real, modified to eigenvalues)
	e: []f32, // Off-diagonal (real, modified)
	w: []f32, // Pre-allocated eigenvalues output (real)
	Z: ^Matrix(complex64) = nil, // Complex eigenvectors (optional output)
	isuppz: []Blas_Int = nil, // Pre-allocated support arrays
	work: []f32, // Pre-allocated workspace (real)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range: EigenRangeOption = .ALL,
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	if Z != nil {
		// Create real eigenvector matrix that overlays the complex matrix
		zptr := cast([^]f32)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f32) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_mrrr_f32(
			d,
			e,
			w,
			&Z_real,
			isuppz,
			work,
			iwork,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	} else {
		return compute_tridiagonal_mrrr_f32(
			d,
			e,
			w,
			nil,
			isuppz,
			work,
			iwork,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	}
}

// Compute tridiagonal eigenvalues/eigenvectors using MRRR for c128
compute_tridiagonal_mrrr_c128 :: proc(
	d: []f64, // Diagonal (real, modified to eigenvalues)
	e: []f64, // Off-diagonal (real, modified)
	w: []f64, // Pre-allocated eigenvalues output (real)
	Z: ^Matrix(complex128) = nil, // Complex eigenvectors (optional output)
	isuppz: []Blas_Int = nil, // Pre-allocated support arrays
	work: []f64, // Pre-allocated workspace (real)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range: EigenRangeOption = .ALL,
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	if Z != nil {
		// Create real eigenvector matrix that overlays the complex matrix
		zptr := cast([^]f64)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f64) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_mrrr_f64(
			d,
			e,
			w,
			&Z_real,
			isuppz,
			work,
			iwork,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	} else {
		return compute_tridiagonal_mrrr_f64(
			d,
			e,
			w,
			nil,
			isuppz,
			work,
			iwork,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	}
}


// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - BISECTION AND INVERSE ITERATION
// ============================================================================

// Query workspace for bisection and inverse iteration eigenvalue/eigenvector computation
query_workspace_tridiagonal_bisection :: proc(
	$T: typeid,
	n: int,
) -> (
	work_size: int,
	iwork_size: int,
) {
	// STEVX requires:
	// work: 5*n for real types
	// iwork: 5*n
	return 5 * n, 5 * n
}

// Compute tridiagonal eigenvalues/eigenvectors using bisection for f32
compute_tridiagonal_bisection_f32 :: proc(
	d: []f32, // Diagonal (preserved in a copy)
	e: []f32, // Off-diagonal (preserved in a copy)
	w: []f32, // Pre-allocated eigenvalues output
	Z: ^Matrix(f32) = nil, // Eigenvectors (optional output)
	work: []f32, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices
	range: EigenRangeOption = .ALL,
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Insufficient failure array")

	jobz := Z != nil ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	// Call LAPACK
	lapack.sstevx_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		len(jobz_cstring),
		len(range_cstring),
	)

	num_found = int(m)
	ok = info == 0
	return num_found, info, ok
}

// Compute tridiagonal eigenvalues/eigenvectors using bisection for f64
compute_tridiagonal_bisection_f64 :: proc(
	d: []f64, // Diagonal (preserved in a copy)
	e: []f64, // Off-diagonal (preserved in a copy)
	w: []f64, // Pre-allocated eigenvalues output
	Z: ^Matrix(f64) = nil, // Eigenvectors (optional output)
	work: []f64, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices
	range: EigenRangeOption = .ALL,
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Insufficient failure array")

	jobz := Z != nil ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = &Z.data[0]
	}

	// Call LAPACK
	lapack.dstevx_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		len(jobz_cstring),
		len(range_cstring),
	)

	num_found = int(m)
	ok = info == 0
	return num_found, info, ok
}

// Compute tridiagonal eigenvalues/eigenvectors using bisection for c64
compute_tridiagonal_bisection_c64 :: proc(
	d: []f32, // Diagonal (real, preserved)
	e: []f32, // Off-diagonal (real, preserved)
	w: []f32, // Pre-allocated eigenvalues output (real)
	Z: ^Matrix(complex64) = nil, // Complex eigenvectors (optional output)
	work: []f32, // Pre-allocated workspace (real)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices
	range: EigenRangeOption = .ALL,
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	if Z != nil {
		// Create real eigenvector matrix that overlays the complex matrix
		zptr := cast([^]f32)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f32) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_bisection_f32(
			d,
			e,
			w,
			&Z_real,
			work,
			iwork,
			ifail,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	} else {
		return compute_tridiagonal_bisection_f32(
			d,
			e,
			w,
			nil,
			work,
			iwork,
			ifail,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	}
}

// Compute tridiagonal eigenvalues/eigenvectors using bisection for c128
compute_tridiagonal_bisection_c128 :: proc(
	d: []f64, // Diagonal (real, preserved)
	e: []f64, // Off-diagonal (real, preserved)
	w: []f64, // Pre-allocated eigenvalues output (real)
	Z: ^Matrix(complex128) = nil, // Complex eigenvectors (optional output)
	work: []f64, // Pre-allocated workspace (real)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices
	range: EigenRangeOption = .ALL,
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) {
	if Z != nil {
		// Create real eigenvector matrix that overlays the complex matrix
		zptr := cast([^]f64)cast(rawptr)&Z.data[0]
		Z_real := Matrix(f64) {
			data = zptr[:2 * len(Z.data)],
			rows = Z.rows,
			cols = Z.cols,
			ld   = Z.ld * 2,
		}
		return compute_tridiagonal_bisection_f64(
			d,
			e,
			w,
			&Z_real,
			work,
			iwork,
			ifail,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	} else {
		return compute_tridiagonal_bisection_f64(
			d,
			e,
			w,
			nil,
			work,
			iwork,
			ifail,
			range,
			vl,
			vu,
			il,
			iu,
			abstol,
		)
	}
}

// ============================================================================
// SYMMETRIC MATRIX CONDITION NUMBER ESTIMATION
// ============================================================================

// Query workspace for symmetric condition number estimation
query_workspace_symmetric_condition :: proc(
	$T: typeid,
	n: int,
) -> (
	work_size: int,
	iwork_size: int,
) {
	when T == f32 || T == f64 {
		// Real types need work and iwork
		return 2 * n, n
	} else when T == complex64 || T == complex128 {
		// Complex types only need work
		return 2 * n, 0
	}
}

// Estimate symmetric matrix condition number for f32/c64
m_estimate_symmetric_condition_f32_c64 :: proc(
	A: ^Matrix($T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: $RealType, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: RealType,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64,
	RealType == f32 {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")

	when T == f32 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
		assert(len(iwork) >= n, "Insufficient integer workspace")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	anorm_val := anorm
	rcond_val: f32

	when T == f32 {
		lapack.ssycon_(
			uplo_c,
			&n_int,
			raw_data(A.data),
			&lda,
			raw_data(ipiv),
			&anorm_val,
			&rcond_val,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.csycon_(
			uplo_c,
			&n_int,
			cast(^lapack.complex)raw_data(A.data),
			&lda,
			raw_data(ipiv),
			&anorm_val,
			&rcond_val,
			cast(^lapack.complex)raw_data(work),
			&info,
			len(uplo_c),
		)
	}

	rcond = rcond_val
	ok = info == 0
	return rcond, info, ok
}

// Estimate symmetric matrix condition number for f64/c128
m_estimate_symmetric_condition_f64_c128 :: proc(
	A: ^Matrix($T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: $RealType, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: RealType,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128,
	RealType == f64 {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")

	when T == f64 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
		assert(len(iwork) >= n, "Insufficient integer workspace")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	anorm_val := anorm
	rcond_val: f64

	when T == f64 {
		lapack.dsycon_(
			uplo_c,
			&n_int,
			raw_data(A.data),
			&lda,
			raw_data(ipiv),
			&anorm_val,
			&rcond_val,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zsycon_(
			uplo_c,
			&n_int,
			cast(^lapack.doublecomplex)raw_data(A.data),
			&lda,
			raw_data(ipiv),
			&anorm_val,
			&rcond_val,
			cast(^lapack.doublecomplex)raw_data(work),
			&info,
			len(uplo_c),
		)
	}

	rcond = rcond_val
	ok = info == 0
	return rcond, info, ok
}

// Proc group for symmetric condition estimation
m_estimate_symmetric_condition :: proc {
	m_estimate_symmetric_condition_f32_c64,
	m_estimate_symmetric_condition_f64_c128,
}
