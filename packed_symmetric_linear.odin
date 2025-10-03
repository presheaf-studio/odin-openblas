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
factorize_packed_symmetric :: proc {
	factorize_packed_symmetric_real,
	factorize_packed_symmetric_complex,
}

// Real symmetric packed factorization (f32/f64)
factorize_packed_symmetric_real :: proc(
	AP: []$T, // Packed matrix (modified to L*D*L^T factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.ssptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dsptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex symmetric packed factorization (complex64/complex128)
// Note: These are symmetric (not Hermitian) factorizations
factorize_packed_symmetric_complex :: proc(
	AP: []$T, // Packed matrix (modified to L*D*L^T factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.csptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zsptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Compute Bunch-Kaufman factorization of Hermitian packed matrix
factorize_packed_hermitian :: proc {
	factorize_packed_hermitian_complex,
}

// Complex Hermitian packed factorization (complex64/complex128)
factorize_packed_hermitian_complex :: proc(
	AP: []$T, // Packed matrix (modified to L*D*L^H factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.chptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zhptrf_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Solve linear system using packed matrix (combined factorization + solve)
solve_packed_symmetric :: proc {
	solve_packed_symmetric_real,
	solve_packed_symmetric_complex,
}

// Real symmetric packed solver (f32/f64)
solve_packed_symmetric_real :: proc(
	AP: []$T, // Packed matrix (modified to factorization)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
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
	}

	ok = (info == 0)
	return info, ok
}

// Complex symmetric packed solver (complex64/complex128)
solve_packed_symmetric_complex :: proc(
	AP: []$T, // Packed matrix (modified to factorization)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	// Note: These are for complex symmetric, not Hermitian
	when T == complex64 {
		lapack.cspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zspsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// Solve linear system using packed Hermitian matrix
solve_packed_hermitian :: proc {
	solve_packed_hermitian_complex,
}

// Complex Hermitian packed solver (complex64/complex128)
solve_packed_hermitian_complex :: proc(
	AP: []$T, // Packed matrix (modified to factorization)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
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

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// TRIANGULAR SOLVERS (using factorization)
// ===================================================================================

// Solve using existing factorization
solve_triangular_packed_symmetric :: proc {
	solve_triangular_packed_symmetric_real,
	solve_triangular_packed_symmetric_complex,
}

// Real symmetric triangular solver (f32/f64)
solve_triangular_packed_symmetric_real :: proc(
	AP: []$T, // Factored packed matrix (from sptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pivot indices from factorization
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
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
	}

	ok = (info == 0)
	return info, ok
}

// Complex symmetric triangular solver (complex64/complex128)
solve_triangular_packed_symmetric_complex :: proc(
	AP: []$T, // Factored packed matrix (from csptrf/zsptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pivot indices from factorization
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == complex64 {
		lapack.csptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zsptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(ipiv), raw_data(B), &ldb_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// Solve using existing Hermitian factorization
solve_triangular_packed_hermitian :: proc {
	solve_triangular_packed_hermitian_complex,
}

// Complex Hermitian triangular solver (complex64/complex128)
solve_triangular_packed_hermitian_complex :: proc(
	AP: []$T, // Factored packed matrix (from chptrf/zhptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	ipiv: []Blas_Int, // Pivot indices from factorization
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
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

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================

// Invert packed symmetric matrix using factorization
invert_packed_symmetric :: proc {
	invert_packed_symmetric_real,
	invert_packed_symmetric_complex,
}

// Real symmetric packed inversion (f32/f64)
invert_packed_symmetric_real :: proc(
	AP: []$T, // Factored packed matrix (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Pre-allocated workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.ssptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == f64 {
		lapack.dsptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex symmetric packed inversion (complex64/complex128)
invert_packed_symmetric_complex :: proc(
	AP: []$T, // Factored packed matrix (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Pre-allocated workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == complex64 {
		lapack.csptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zsptri_(&uplo_c, &n_blas, raw_data(AP), raw_data(ipiv), raw_data(work), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Invert packed Hermitian matrix using factorization
invert_packed_hermitian :: proc {
	invert_packed_hermitian_complex,
}

// Complex Hermitian packed inversion (complex64/complex128)
invert_packed_hermitian_complex :: proc(
	AP: []$T, // Factored packed matrix (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Pre-allocated workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
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

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number using factorization
estimate_condition_packed_symmetric :: proc {
	estimate_condition_packed_symmetric_real,
	estimate_condition_packed_symmetric_complex,
}

// Real symmetric packed condition estimation (f32/f64)
estimate_condition_packed_symmetric_real :: proc(
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
) where T == f32 || T == f64 {
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

	ok = (info == 0)
	return rcond, info, ok
}

// Complex symmetric packed condition estimation (complex64/complex128)
estimate_condition_packed_symmetric_complex :: proc(
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

	ok = (info == 0)
	return rcond, info, ok
}

// Estimate condition number for Hermitian packed matrix
estimate_condition_packed_hermitian :: proc {
	estimate_condition_packed_hermitian_complex,
}

// Complex Hermitian packed condition estimation (complex64/complex128)
estimate_condition_packed_hermitian_complex :: proc(
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

	ok = (info == 0)
	return rcond, info, ok
}

// ===================================================================================
// EXPERT SOLVERS (with condition estimation and error bounds)
// ===================================================================================

// Query workspace for expert solver
query_workspace_solve_expert_packed_symmetric :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
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
solve_expert_packed_symmetric :: proc {
	solve_expert_packed_symmetric_real,
	solve_expert_packed_symmetric_complex,
}

// Real symmetric packed expert solver (f32/f64)
solve_expert_packed_symmetric_real :: proc(
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
) where T == f32 || T == f64 {
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

	ok = (info == 0)
	return rcond, info, ok
}

// Complex symmetric packed expert solver (complex64/complex128)
solve_expert_packed_symmetric_complex :: proc(
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

	ok = (info == 0)
	return rcond, info, ok
}

// Expert solver for packed Hermitian matrices
solve_expert_packed_hermitian :: proc {
	solve_expert_packed_hermitian_complex,
}

// Complex Hermitian packed expert solver (complex64/complex128)
solve_expert_packed_hermitian_complex :: proc(
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

	ok = (info == 0)
	return rcond, info, ok
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Refine solution using iterative refinement for packed symmetric matrices
refine_packed_symmetric :: proc {
	refine_packed_symmetric_real,
	refine_packed_symmetric_complex,
}

// Real symmetric packed iterative refinement (f32/f64)
refine_packed_symmetric_real :: proc(
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
) where T == f32 || T == f64 {
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

	ok = (info == 0)
	return info, ok
}

// Complex symmetric packed iterative refinement (complex64/complex128)
refine_packed_symmetric_complex :: proc(
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

	ok = (info == 0)
	return info, ok
}

// Refine solution for Hermitian packed matrices
refine_packed_hermitian :: proc {
	refine_packed_hermitian_complex,
}

// Complex Hermitian packed iterative refinement (complex64/complex128)
refine_packed_hermitian_complex :: proc(
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

	ok = (info == 0)
	return info, ok
}
