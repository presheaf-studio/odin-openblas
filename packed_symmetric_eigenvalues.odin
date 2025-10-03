package openblas

import lapack "./f77"

// ===================================================================================
// PACKED SYMMETRIC EIGENVALUE OPERATIONS
// ===================================================================================
//
// This file provides eigenvalue routines for symmetric/Hermitian matrices stored
// in packed format. Includes:
// - Standard eigenvalue problems (spev/dspev/chpev/zhpev)
// - Eigenvalues with divide-and-conquer (spevd/dspevd/chpevd/zhpevd)
// - Selective eigenvalue computation (spevx/dspevx/chpevx/zhpevx)
// - Generalized eigenvalue problems (spgv/dspgv/chpgv/zhpgv)
// - Tridiagonal reduction (sptrd/dsptrd/chptrd/zhptrd)
// - Generalized reductions (spgst/dspgst/chpgst/zhpgst)

// ===================================================================================
// STANDARD EIGENVALUE PROBLEMS
// ===================================================================================

// Compute all eigenvalues and optionally eigenvectors
eigenvalues_packed_symmetric :: proc {
	eigenvalues_packed_symmetric_real,
	eigenvalues_packed_hermitian_complex,
}

// Real symmetric packed eigenvalues (f32/f64)
eigenvalues_packed_symmetric_real :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (size 3*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Workspace too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)

	when T == f32 {
		lapack.sspev_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &info)
	} else when T == f64 {
		lapack.dspev_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed eigenvalues (complex64/complex128)
eigenvalues_packed_hermitian_complex :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size 3*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	jobz: EigenJobOption, // Compute eigenvectors?
	uplo: MatrixRegion, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 3 * n, "Real workspace too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)

	when T == complex64 {
		lapack.chpev_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zhpev_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// DIVIDE-AND-CONQUER EIGENVALUE SOLVER
// ===================================================================================

// Compute eigenvalues using divide-and-conquer (faster for large matrices)
eigenvalues_packed_symmetric_dc :: proc {
	eigenvalues_packed_symmetric_dc_real,
	eigenvalues_packed_hermitian_dc_complex,
}

// Real symmetric packed eigenvalues with divide-and-conquer (f32/f64)
eigenvalues_packed_symmetric_dc_real :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (query with lwork=-1)
	iwork: []Blas_Int, // Pre-allocated integer workspace (query with liwork=-1)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	lwork: int, // Size of work array
	liwork: int, // Size of iwork array
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	lwork_blas := Blas_Int(lwork)
	liwork_blas := Blas_Int(liwork)

	when T == f32 {
		lapack.sspevd_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(iwork), &liwork_blas, &info)
	} else when T == f64 {
		lapack.dspevd_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(iwork), &liwork_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed eigenvalues with divide-and-conquer (complex64/complex128)
eigenvalues_packed_hermitian_dc_complex :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (query with lwork=-1)
	rwork: []Real, // Pre-allocated real workspace (query with lrwork=-1)
	iwork: []Blas_Int, // Pre-allocated integer workspace (query with liwork=-1)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	lwork: int, // Size of work array
	lrwork: int, // Size of rwork array
	liwork: int, // Size of iwork array
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	lwork_blas := Blas_Int(lwork)
	lrwork_blas := Blas_Int(lrwork)
	liwork_blas := Blas_Int(liwork)

	when T == complex64 {
		lapack.chpevd_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(rwork), &lrwork_blas, raw_data(iwork), &liwork_blas, &info)
	} else when T == complex128 {
		lapack.zhpevd_(&jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(rwork), &lrwork_blas, raw_data(iwork), &liwork_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// SELECTIVE EIGENVALUE COMPUTATION
// ===================================================================================

// Compute selected eigenvalues and eigenvectors
eigenvalues_packed_symmetric_select :: proc {
	eigenvalues_packed_symmetric_select_real,
	eigenvalues_packed_hermitian_select_complex,
}

// Real symmetric packed selective eigenvalues (f32/f64)
eigenvalues_packed_symmetric_select_real :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×m] where m = number found
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	work: []T, // Pre-allocated workspace (size 8*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	vl, vu: T, // Range bounds (if range = .VALUE)
	il, iu: int, // Index bounds (if range = .INDEX)
	abstol: T, // Absolute tolerance for eigenvalues
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	range: EigenRangeOption = .ALL, // Which eigenvalues to compute
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	m: int,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failure array too small")
	assert(len(work) >= 8 * n, "Workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	range_c := u8(range)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	il_blas := Blas_Int(il)
	iu_blas := Blas_Int(iu)
	m_blas := Blas_Int(0)

	when T == f32 {
		lapack.sspevx_(&jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dspevx_(&jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	m = int(m_blas)
	ok = (info == 0)
	return m, info, ok
}

// Complex Hermitian packed selective eigenvalues (complex64/complex128)
eigenvalues_packed_hermitian_select_complex :: proc(
	AP: []$T, // Packed matrix (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []T, // Pre-allocated eigenvectors [n×m] where m = number found
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size 7*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	vl, vu: Real, // Range bounds (if range = .VALUE)
	il, iu: int, // Index bounds (if range = .INDEX)
	abstol: Real, // Absolute tolerance for eigenvalues
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	range: EigenRangeOption = .ALL, // Which eigenvalues to compute
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	m: int,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failure array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	jobz_c := u8(jobz)
	range_c := u8(range)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	il_blas := Blas_Int(il)
	iu_blas := Blas_Int(iu)
	m_blas := Blas_Int(0)

	when T == complex64 {
		lapack.chpevx_(&jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	} else when T == complex128 {
		lapack.zhpevx_(&jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	}

	m = int(m_blas)
	ok = (info == 0)
	return m, info, ok
}

// ===================================================================================
// TRIDIAGONAL REDUCTION
// ===================================================================================

// Reduce packed symmetric/Hermitian matrix to tridiagonal form
reduce_packed_to_tridiagonal :: proc {
	reduce_packed_symmetric_to_tridiagonal_real,
	reduce_packed_hermitian_to_tridiagonal_complex,
}

// Real symmetric packed to tridiagonal reduction (f32/f64)
reduce_packed_symmetric_to_tridiagonal_real :: proc(
	AP: []$T, // Packed matrix (modified to orthogonal matrix)
	D: []T, // Pre-allocated diagonal elements (size n)
	E: []T, // Pre-allocated off-diagonal elements (size n-1)
	tau: []T, // Pre-allocated elementary reflectors (size n-1)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(D) >= n, "Diagonal array too small")
	if n > 1 {
		assert(len(E) >= n - 1, "Off-diagonal array too small")
		assert(len(tau) >= n - 1, "Tau array too small")
	}

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.ssptrd_(&uplo_c, &n_blas, raw_data(AP), raw_data(D), raw_data(E), raw_data(tau), &info)
	} else when T == f64 {
		lapack.dsptrd_(&uplo_c, &n_blas, raw_data(AP), raw_data(D), raw_data(E), raw_data(tau), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed to tridiagonal reduction (complex64/complex128)
reduce_packed_hermitian_to_tridiagonal_complex :: proc(
	AP: []$T, // Packed matrix (modified to unitary matrix)
	D: []$Real, // Pre-allocated diagonal elements (size n, always real)
	E: []Real, // Pre-allocated off-diagonal elements (size n-1, always real)
	tau: []T, // Pre-allocated elementary reflectors (size n-1)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(D) >= n, "Diagonal array too small")
	if n > 1 {
		assert(len(E) >= n - 1, "Off-diagonal array too small")
		assert(len(tau) >= n - 1, "Tau array too small")
	}

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chptrd_(&uplo_c, &n_blas, raw_data(AP), raw_data(D), raw_data(E), raw_data(tau), &info)
	} else when T == complex128 {
		lapack.zhptrd_(&uplo_c, &n_blas, raw_data(AP), raw_data(D), raw_data(E), raw_data(tau), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS
// ===================================================================================

// Solve generalized eigenvalue problem A*x = lambda*B*x
eigenvalues_packed_generalized :: proc {
	eigenvalues_packed_generalized_real,
	eigenvalues_packed_hermitian_generalized_complex,
}

// Real symmetric packed generalized eigenvalues (f32/f64)
eigenvalues_packed_generalized_real :: proc(
	AP: []$T, // Packed matrix A (destroyed on output)
	BP: []T, // Packed matrix B (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (size 3*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)

	when T == f32 {
		lapack.sspgv_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &info)
	} else when T == f64 {
		lapack.dspgv_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed generalized eigenvalues (complex64/complex128)
eigenvalues_packed_hermitian_generalized_complex :: proc(
	AP: []$T, // Packed matrix A (destroyed on output)
	BP: []T, // Packed matrix B (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size 3*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 3 * n, "Real workspace too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)

	when T == complex64 {
		lapack.chpgv_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zhpgv_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// GENERALIZED REDUCTION
// ===================================================================================

// Reduce generalized eigenproblem to standard form
reduce_packed_generalized :: proc {
	reduce_packed_generalized_real,
	reduce_packed_hermitian_generalized_complex,
}

// Real symmetric packed generalized reduction (f32/f64)
reduce_packed_generalized_real :: proc(
	AP: []$T, // Packed matrix A (modified on output)
	BP: []T, // Packed Cholesky factor of B (from spptrf)
	n: int, // Matrix dimension
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	itype_blas := Blas_Int(itype)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.sspgst_(&itype_blas, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &info)
	} else when T == f64 {
		lapack.dspgst_(&itype_blas, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed generalized reduction (complex64/complex128)
reduce_packed_hermitian_generalized_complex :: proc(
	AP: []$T, // Packed matrix A (modified on output)
	BP: []T, // Packed Cholesky factor of B (from cpptrf/zpptrf)
	n: int, // Matrix dimension
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	itype_blas := Blas_Int(itype)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chpgst_(&itype_blas, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &info)
	} else when T == complex128 {
		lapack.zhpgst_(&itype_blas, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS - DIVIDE-AND-CONQUER
// ===================================================================================

// Solve generalized eigenvalue problem A*x = lambda*B*x using divide-and-conquer
eigenvalues_packed_generalized_dc :: proc {
	eigenvalues_packed_generalized_dc_real,
	eigenvalues_packed_hermitian_generalized_dc_complex,
}

// Real symmetric packed generalized eigenvalues with divide-and-conquer (f32/f64)
eigenvalues_packed_generalized_dc_real :: proc(
	AP: []$T, // Packed matrix A (destroyed on output)
	BP: []T, // Packed matrix B (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []T, // Pre-allocated workspace (query with lwork=-1)
	iwork: []Blas_Int, // Pre-allocated integer workspace (query with liwork=-1)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	lwork: int, // Size of work array
	liwork: int, // Size of iwork array
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	lwork_blas := Blas_Int(lwork)
	liwork_blas := Blas_Int(liwork)

	when T == f32 {
		lapack.sspgvd_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(iwork), &liwork_blas, &info)
	} else when T == f64 {
		lapack.dspgvd_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(iwork), &liwork_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex Hermitian packed generalized eigenvalues with divide-and-conquer (complex64/complex128)
eigenvalues_packed_hermitian_generalized_dc_complex :: proc(
	AP: []$Cmplx, // Packed matrix A (destroyed on output)
	BP: []Cmplx, // Packed matrix B (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []Cmplx, // Pre-allocated eigenvectors [n×n] (if jobz = .VALUES_AND_VECTORS)
	work: []Cmplx, // Pre-allocated workspace (query with lwork=-1)
	rwork: []Real, // Pre-allocated real workspace (query with lrwork=-1)
	iwork: []Blas_Int, // Pre-allocated integer workspace (query with liwork=-1)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	lwork: int, // Size of work array
	lrwork: int, // Size of rwork array
	liwork: int, // Size of iwork array
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	lwork_blas := Blas_Int(lwork)
	lrwork_blas := Blas_Int(lrwork)
	liwork_blas := Blas_Int(liwork)

	when Cmplx == complex64 {
		lapack.chpgvd_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(rwork), &lrwork_blas, raw_data(iwork), &liwork_blas, &info)
	} else when Cmplx == complex128 {
		lapack.zhpgvd_(&itype_blas, &jobz_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), &lwork_blas, raw_data(rwork), &lrwork_blas, raw_data(iwork), &liwork_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS - SELECTIVE
// ===================================================================================

// Solve generalized eigenvalue problem A*x = lambda*B*x for selected eigenvalues
eigenvalues_packed_generalized_select :: proc {
	eigenvalues_packed_generalized_select_real,
	eigenvalues_packed_hermitian_generalized_select_complex,
}

// Real symmetric packed generalized selective eigenvalues (f32/f64)
eigenvalues_packed_generalized_select_real :: proc(
	AP: []$T, // Packed matrix A (destroyed on output)
	BP: []T, // Packed matrix B (destroyed on output)
	W: []T, // Pre-allocated eigenvalues (size n)
	Z: []T, // Pre-allocated eigenvectors [n×m] where m = number found
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	work: []T, // Pre-allocated workspace (size 8*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	vl, vu: T, // Range bounds (if range = .VALUE)
	il, iu: int, // Index bounds (if range = .INDEX)
	abstol: T, // Absolute tolerance for eigenvalues
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	range: EigenRangeOption = .ALL, // Which eigenvalues to compute
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	m: int,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failure array too small")
	assert(len(work) >= 8 * n, "Workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	range_c := u8(range)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	il_blas := Blas_Int(il)
	iu_blas := Blas_Int(iu)
	m_blas := Blas_Int(0)

	when T == f32 {
		lapack.sspgvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dspgvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	m = int(m_blas)
	ok = (info == 0)
	return m, info, ok
}

// Complex Hermitian packed generalized selective eigenvalues (complex64/complex128)
eigenvalues_packed_hermitian_generalized_select_complex :: proc(
	AP: []$Cmplx, // Packed matrix A (destroyed on output)
	BP: []Cmplx, // Packed matrix B (destroyed on output)
	W: []$Real, // Pre-allocated eigenvalues (size n, always real)
	Z: []Cmplx, // Pre-allocated eigenvectors [n×m] where m = number found
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size 7*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	n: int, // Matrix dimension
	ldz: int, // Leading dimension of Z
	vl, vu: Real, // Range bounds (if range = .VALUE)
	il, iu: int, // Index bounds (if range = .INDEX)
	abstol: Real, // Absolute tolerance for eigenvalues
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption = .VALUES_ONLY, // Compute eigenvectors?
	range: EigenRangeOption = .ALL, // Which eigenvalues to compute
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	m: int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(BP)), "BP array too small")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failure array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	if jobz == .VALUES_AND_VECTORS {
		assert(len(Z) >= n * ldz, "Eigenvector array too small")
		assert(ldz >= n, "Leading dimension too small")
	}

	itype_blas := Blas_Int(itype)
	jobz_c := u8(jobz)
	range_c := u8(range)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	ldz_blas := Blas_Int(ldz)
	il_blas := Blas_Int(il)
	iu_blas := Blas_Int(iu)
	m_blas := Blas_Int(0)

	when Cmplx == complex64 {
		lapack.chpgvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	} else when Cmplx == complex128 {
		lapack.zhpgvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_blas, raw_data(AP), raw_data(BP), &vl, &vu, &il_blas, &iu_blas, &abstol, &m_blas, raw_data(W), raw_data(Z), &ldz_blas, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	}

	m = int(m_blas)
	ok = (info == 0)
	return m, info, ok
}
