package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SIMPLE TRIDIAGONAL EIGENVALUE COMPUTATION
// ============================================================================

// Compute eigenvalues only using STERF (simple, no workspace)


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

// Compute eigenvalues only (S/D STERF - no workspace needed)
m_compute_tridiagonal_eigenvalues_only :: proc(
	d: []$T, // Diagonal (modified to eigenvalues on output)
	e: []T, // Off-diagonal (destroyed)
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
			return m_compute_eigenvalues_only_f32_f64(d, e)
		} else when is_complex(T) {
			// For complex Hermitian tridiagonal, diagonal is real part
			d[i] = real(M.data[i * M.ld + i])
			if i < n - 1 {
				// Off-diagonal is also real for Hermitian tridiagonal
				e[i] = real(M.data[i * M.ld + i + 1]) // Upper diagonal
				// e[i] = real(M.data[(i + 1) * M.ld + i]) // Lower diagonal
			}
			return m_compute_eigenvalues_only_f32_f64(d, e)
		}
	}

}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - SIMPLE DRIVER (STEV)
// ============================================================================

// Query workspace for eigenvalue/eigenvector computation (STEV)
query_workspace_eigenvalues_tridiagonal_vectors :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int) {
	if compute_vectors {
		return 2 * n - 2
	}
	return 0
}

// Compute eigenvalues and optionally eigenvectors (STEV)
m_compute_eigenvalues_tridiagonal_vectors :: proc(
	d: []$T, // Diagonal (modified to eigenvalues on output)
	e: []T, // Off-diagonal (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvectors (optional output)
	work: []T = nil, // Pre-allocated workspace (2*n-2 if computing vectors)
	jobz := EigenJobOption.VALUES_ONLY,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(Z != nil && int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		assert(work != nil && len(work) >= 2 * n - 2, "Insufficient workspace for eigenvectors")
	}

	jobz_c := cast(u8)jobz
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
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

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - DIVIDE AND CONQUER DRIVER
// ============================================================================

// Query workspace for divide and conquer eigenvalue/eigenvector computation (STEVD)
query_workspace_tridiagonal_eigenvalues_all_dc :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int, iwork_size: int) {
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

	jobz_c := cast(u8)compute_vectors ? .VALUES_AND_VECTORS : .VALUES_ONLY

	when T == f32 {
		lapack.sstevd_(&jobz_c, &n_blas, &d_dummy, &e_dummy, &z_dummy, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info)
	} else when T == f64 {
		lapack.dstevd_(&jobz_c, &n_blas, &d_dummy, &e_dummy, &z_dummy, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info)
	}

	return int(work_query), int(iwork_query)
}

// Compute eigenvalues/eigenvectors using divide and conquer (STEVD)
m_compute_tridiagonal_eigenvalues_all_dc :: proc(
	d: []$T, // Diagonal (modified to eigenvalues on output)
	e: []T, // Off-diagonal (destroyed)
	Z: ^Matrix(T) = nil, // Eigenvectors (optional output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compute_vectors := Z != nil
	if compute_vectors {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	// Verify workspace  TODO: WHEN OB_ASSERT CLAUSE
	work_size, iwork_size := query_workspace_dc(T, n, compute_vectors)
	assert(len(work) >= work_size, "Insufficient workspace")
	assert(len(iwork) >= iwork_size, "Insufficient integer workspace")

	jobz_c := cast(u8)compute_vectors ? .VALUES_AND_VECTORS : .VALUES_ONLY

	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if compute_vectors {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Call LAPACK
	when T == f32 {
		lapack.sstevd_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dstevd_(&jobz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - MRRR DRIVER (STEVR)
// ============================================================================
// Note: This uses STEVR for symmetric tridiagonal from reduction
// For general tridiagonal MRRR, use tridiagonal_mrrr.odin (STEGR)

// Query workspace for MRRR eigenvalue/eigenvector computation (STEVR)
query_workspace_tridiagonal_symmetric_mrrr :: proc($T: typeid, n: int, compute_vectors: bool) -> (work_size: int, iwork_size: int) {
	// Query LAPACK for optimal workspace sizes
	n_blas := Blas_Int(n)
	jobz_c := cast(u8)compute_vectors ? .VALUES_AND_VECTORS : .VALUES_ONLY
	range_c := cast(u8)EigenRangeOption.ALL

	// Dummy values for workspace query
	d_dummy: T
	e_dummy: T
	vl_dummy: T = 0
	vu_dummy: T = 0
	il_dummy := Blas_Int(1)
	iu_dummy := Blas_Int(n)
	abstol_dummy: T = 0
	m_dummy: Blas_Int
	ldz := Blas_Int(1)

	lwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.sstevr_(
			&jobz_c,
			&range_c,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&vl_dummy,
			&vu_dummy,
			&il_dummy,
			&iu_dummy,
			&abstol_dummy,
			&m_dummy,
			nil, // w
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

		lapack.dstevr_(
			&jobz_c,
			&range_c,
			&n_blas,
			&d_dummy,
			&e_dummy,
			&vl_dummy,
			&vu_dummy,
			&il_dummy,
			&iu_dummy,
			&abstol_dummy,
			&m_dummy,
			nil, // w
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

	return work_size, iwork_size
}

// Compute eigenvalues/eigenvectors using MRRR for symmetric tridiagonal (STEVR)
m_compute_eigenvalues_tridiagonal_symmetric_mrrr :: proc(
	d: []$T, // Diagonal (modified to eigenvalues on output)
	e: []T, // Off-diagonal (modified)
	w: []T, // Pre-allocated eigenvalues output
	Z: ^Matrix(T) = nil, // Eigenvectors (optional output)
	isuppz: []Blas_Int = nil, // Pre-allocated support arrays (size 2*max_eigenvalues)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	range := EigenRangeOption.ALL,
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) > 0, "Work array required")
	assert(len(iwork) > 0, "Integer work array required")

	jobz := Z != nil ? EigenJobOption.VALUES_AND_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_c := cast(u8)jobz
	range_c := cast(u8)EigenRangeOption.ALL

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
	z_ptr: rawptr = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)

		// Support arrays required for eigenvectors
		max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
		assert(isuppz != nil && len(isuppz) >= 2 * max_m, "Support array required for eigenvectors")
	}

	// Call LAPACK
	when T == f32 {
		lapack.sstevr_(
			&jobz_c,
			&range_c,
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
		)
	} else when T == f64 {
		lapack.dstevr_(
			&jobz_c,
			&range_c,
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
		)
	}

	num_found = int(m)
	return num_found, info, info == 0
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - BISECTION AND INVERSE ITERATION
// ============================================================================

// Query workspace for bisection and inverse iteration (STEVX)
query_workspace_tridiagonal_bisection :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
	// STEVX requires:
	// work: 5*n for real types
	// iwork: 5*n
	return 5 * n, 5 * n
}

// Compute eigenvalues/eigenvectors using bisection and inverse iteration (STEVX)
m_compute_eigenvalues_tridiagonal_bisection :: proc(
	d: []$T, // Diagonal (preserved in a copy)
	e: []T, // Off-diagonal (preserved in a copy)
	w: []T, // Pre-allocated eigenvalues output
	Z: ^Matrix(T) = nil, // Eigenvectors (optional output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices
	range: EigenRangeOption = .ALL,
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance
) -> (
	num_found: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(work) >= 5 * n, "Insufficient workspace")
	assert(len(iwork) >= 5 * n, "Insufficient integer workspace")
	assert(len(ifail) >= n, "Insufficient failure array")

	jobz := Z != nil ? EigenJobOption.VALUES_AND_VECTORS : EigenJobOption.VALUES_ONLY
	jobz_c := cast(u8)jobz
	range_c := cast(u8)EigenRangeOption.ALL

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if Z != nil {
		assert(int(Z.rows) >= n, "Eigenvector matrix too small")
		assert(len(w) >= n, "Eigenvalue array too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	// Call LAPACK
	when T == f32 {
		lapack.sstevx_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dstevx_(&jobz_c, &range_c, &n_int, raw_data(d), raw_data(e), &vl_val, &vu_val, &il_int, &iu_int, &abstol_val, &m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	num_found = int(m)
	return num_found, info, info == 0
}

// ============================================================================
// SYMMETRIC MATRIX CONDITION NUMBER ESTIMATION
// ============================================================================

// Query workspace for symmetric condition number estimation
query_workspace_tridiagonal_symmetric_condition :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
	when is_float(T) {
		// Real types need work and iwork
		return 2 * n, n
	} else when T == complex64 || T == complex128 {
		// Complex types only need work
		return 2 * n, 0
	}
}

// Estimate symmetric matrix condition number for f32/c64
m_estimate_tridiagonal_symmetric_condition_f32_c64 :: proc(
	A: ^Matrix($T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: f32, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f32,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")

	when T == f32 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
		assert(len(iwork) >= n, "Insufficient integer workspace")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
	}

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	anorm_val := anorm
	rcond_val: f32

	when T == f32 {
		lapack.ssycon_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), &anorm_val, &rcond_val, raw_data(work), raw_data(iwork), &info)
	} else when T == complex64 {
		lapack.csycon_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), &anorm_val, &rcond_val, raw_data(work), &info)
	}

	rcond = rcond_val
	return rcond, info, info == 0
}

// Estimate symmetric matrix condition number for f64/c128
m_estimate_symmetric_condition_f64_c128 :: proc(
	A: ^Matrix($T), // Factored matrix from sytrf
	ipiv: []Blas_Int, // Pivot indices from sytrf
	anorm: f64, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: f64,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= n, "Pivot array too small")

	when T == f64 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
		assert(len(iwork) >= n, "Insufficient integer workspace")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Insufficient workspace")
	}

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	anorm_val := anorm

	when T == f64 {
		lapack.dsycon_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), &anorm_val, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == complex128 {
		lapack.zsycon_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(ipiv), &anorm_val, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}
