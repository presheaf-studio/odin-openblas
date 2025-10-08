package openblas

import lapack "./f77"

// ===================================================================================
// PACKED SYMMETRIC LINEAR SOLVERS
// ===================================================================================
//
// This file provides linear solvers for symmetric matrices stored in packed format.
// Includes:
// - Bunch-Kaufman factorization (sptrf/dsptrf/csptrf/zsptrf, chptrf/zhptrf)
// - Linear system solvers (spsv/dspsv/cspsv/zspsv, chpsv/zhpsv)
// - Expert solvers with condition estimation (spsvx/dspsvx/cspsvx/zspsvx, chpsvx/zhpsvx)
// - Triangular solvers (sptrs/dsptrs/csptrs/zsptrs, chptrs/zhptrs)
// - Matrix inversion (sptri/dsptri/csptri/zsptri, chptri/zhptri)
// - Condition number estimation (spcon/dspcon/cspcon/zspcon, chpcon/zhpcon)
// - Iterative refinement (sprfs/dsprfs/csprfs/zsprfs, chprfs/zhprfs)

// ===================================================================================
// FACTORIZATION ROUTINES
// ===================================================================================

// Compute Bunch-Kaufman factorization of symmetric packed matrix
// Supports all types: f32, f64, complex64, complex128
pack_sym_bk_factorize :: proc(
	AP: []$T, // Packed matrix (modified to L*D*L^T factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.ssptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dsptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.csptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zsptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	}

	return info, info == 0
}

// Compute Bunch-Kaufman factorization of Hermitian packed matrix
pack_herm_factorize :: proc(
	AP: []$T, // Packed matrix (modified to L*D*L^H factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zhptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	}

	return info, info == 0
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Solve linear system using packed matrix (combined factorization + solve)
// Supports all types: f32, f64, complex64, complex128
pack_sym_bk_solve :: proc(
	AP: []$T, // Packed matrix (modified to factorization)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.sspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.cspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// Solve linear system using packed Hermitian matrix
pack_herm_solve :: proc(
	AP: []$T, // Packed matrix (modified to factorization)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == complex64 {
		lapack.chpsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zhpsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIANGULAR SOLVERS (using factorization)
// ===================================================================================

// Solve using existing factorization
// Supports all types: f32, f64, complex64, complex128
pack_sym_bk_solve_factorized :: proc(
	AP: []$T, // Factored packed matrix (from sptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pivot indices from factorization
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.ssptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dsptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.csptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zsptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// Solve using existing Hermitian factorization
pack_herm_solve_factorized :: proc(
	AP: []$T, // Factored packed matrix (from chptrf/zhptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pivot indices from factorization
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == complex64 {
		lapack.chptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zhptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================

// Invert packed symmetric matrix using factorization
pack_sym_bk_invert :: proc(
	AP: []$T, // Factored packed matrix (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Pre-allocated workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.ssptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == f64 {
		lapack.dsptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == complex64 {
		lapack.csptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zsptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	}

	return info, info == 0
}

// Invert packed Hermitian matrix using factorization
pack_herm_invert :: proc(
	AP: []$T, // Factored packed matrix (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Pre-allocated workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zhptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number using factorization
pack_sym_bk_condition :: proc {
	pack_sym_bk_condition_real,
	pack_sym_bk_condition_complex,
}

// Real symmetric packed condition estimation (f32/f64)
pack_sym_bk_condition_real :: proc(
	AP: []$T, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	anorm: T, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.sspcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dspcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Complex symmetric packed condition estimation (complex64/complex128)
pack_sym_bk_condition_complex :: proc(
	AP: []$T, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	anorm: $Real, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.cspcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zspcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// Estimate condition number for Hermitian packed matrix
pack_herm_condition :: proc(
	AP: []$T, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	anorm: $Real, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chpcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zhpcon_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// EXPERT SOLVERS (with condition estimation and error bounds)
// ===================================================================================

// Query workspace for expert solver
query_workspace_pack_sym_bk_solve_expert :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// Real types: work = 3*n, iwork = n, no rwork
		work_size = 3 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, no iwork, rwork = n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = n
	}
	return
}

// Expert solver for packed symmetric matrices
pack_sym_bk_solve_expert :: proc {
	pack_sym_bk_solve_expert_real,
	pack_sym_bk_solve_expert_complex,
}

// Real symmetric packed expert solver (f32/f64)
pack_sym_bk_solve_expert_real :: proc(
	AP: []$T, // Original packed matrix
	AFP: []T, // Pre-allocated factored packed matrix (in/out)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (in/out)
	B: []T, // Right-hand side [n×nrhs]
	X: []T, // Pre-allocated solution matrix [n×nrhs]
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	fact: FactorizationOption = .Equilibrate,
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	fact_c := u8(fact)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == f32 {
		lapack.sspsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dspsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Complex symmetric packed expert solver (complex64/complex128)
pack_sym_bk_solve_expert_complex :: proc(
	AP: []$T, // Original packed matrix
	AFP: []T, // Pre-allocated factored packed matrix (in/out)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (in/out)
	B: []T, // Right-hand side [n×nrhs]
	X: []T, // Pre-allocated solution matrix [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	fact: FactorizationOption = .Equilibrate,
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	fact_c := u8(fact)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == complex64 {
		lapack.cspsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zspsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// Expert solver for packed Hermitian matrices
pack_herm_solve_expert :: proc(
	AP: []$T, // Original packed Hermitian matrix
	AFP: []T, // Pre-allocated factored packed matrix (in/out)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (in/out)
	B: []T, // Right-hand side [n×nrhs]
	X: []T, // Pre-allocated solution matrix [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	fact: FactorizationOption = .Equilibrate,
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	fact_c := u8(fact)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == complex64 {
		lapack.chpsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zhpsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Refine solution using iterative refinement for packed symmetric matrices
pack_sym_bk_refine :: proc {
	pack_sym_bk_refine_real,
	pack_sym_bk_refine_complex,
}

// Real symmetric packed iterative refinement (f32/f64)
pack_sym_bk_refine_real :: proc(
	AP: []$T, // Original packed matrix
	AFP: []T, // Factored packed matrix (from sptrf)
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == f32 {
		lapack.ssprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dsprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex symmetric packed iterative refinement (complex64/complex128)
pack_sym_bk_refine_complex :: proc(
	AP: []$T, // Original packed matrix
	AFP: []T, // Factored packed matrix (from csptrf/zsptrf)
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == complex64 {
		lapack.csprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zsprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// Refine solution for Hermitian packed matrices
pack_herm_refine :: proc(
	AP: []$T, // Original packed Hermitian matrix
	AFP: []T, // Factored packed matrix (from chptrf/zhptrf)
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == complex64 {
		lapack.chprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zhprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(ipiv), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}
