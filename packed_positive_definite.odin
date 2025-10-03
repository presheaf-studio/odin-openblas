package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED POSITIVE DEFINITE MATRIX OPERATIONS
// ===================================================================================
//
// This file provides specialized operations for positive definite matrices stored
// in packed format. Positive definite matrices allow for Cholesky factorization,
// which is more efficient and numerically stable than general symmetric factorization.
//
// Includes:
// - Cholesky factorization (pptrf/dpptrf/cpptrf/zpptrf)
// - Linear system solvers (ppsv/dppsv/cppsv/zppsv)
// - Expert solvers (ppsvx/dppsvx/cppsvx/zppsvx)
// - Triangular solvers (pptrs/dpptrs/cpptrs/zpptrs)
// - Matrix inversion (pptri/dpptri/cpptri/zpptri)
// - Condition number estimation (ppcon/dppcon/cppcon/zppcon)
// - Iterative refinement (pprfs/dpprfs/cpprfs/zpprfs)
// - Equilibration (ppequ/dppequ/cppequ/zppequ)

// ===================================================================================
// CHOLESKY FACTORIZATION
// ===================================================================================

// Compute Cholesky factorization A = L*L^T or A = U^T*U
cholesky_factorize_packed :: proc(
	AP: []$T, // Packed matrix (modified to Cholesky factor)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.spptrf_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == f64 {
		lapack.dpptrf_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == complex64 {
		lapack.cpptrf_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == complex128 {
		lapack.zpptrf_(&uplo_c, &n_blas, raw_data(AP), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Solve positive definite system using packed storage (combined factorization + solve)
solve_packed_positive_definite :: proc(
	AP: []$T, // Packed matrix (modified to Cholesky factor)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.sppsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dppsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.cppsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zppsv_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// EXPERT SOLVERS WITH CONDITION ESTIMATION
// ===================================================================================

// Expert solver with equilibration and condition estimation
solve_packed_positive_definite_expert :: proc {
	solve_packed_positive_definite_expert_real,
	solve_packed_positive_definite_expert_complex,
}

// Real positive definite packed expert solver (f32/f64)
solve_packed_positive_definite_expert_real :: proc(
	AP: []$T, // Packed matrix (preserved)
	AFP: []T, // Pre-allocated factorization array (n*(n+1)/2)
	B: []T, // RHS vectors [n×nrhs] (preserved)
	X: []T, // Pre-allocated solution array [n×nrhs]
	S: []T, // Pre-allocated scaling factors (size n)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	fact: FactorizationOption = .Equilibrate, // Factorization option
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	equed: EquilibrationState,
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(S) >= n, "Scaling array too small")
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
	equed_c := u8(EquilibrationState.None)

	when T == f32 {
		lapack.sppsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), &equed_c, raw_data(S), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dppsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), &equed_c, raw_data(S), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	equed = EquilibrationState(equed_c)
	ok = (info == 0)
	return equed, rcond, info, ok
}

// Complex positive definite packed expert solver (complex64/complex128)
solve_packed_positive_definite_expert_complex :: proc(
	AP: []$T, // Packed matrix (preserved)
	AFP: []T, // Pre-allocated factorization array (n*(n+1)/2)
	B: []T, // RHS vectors [n×nrhs] (preserved)
	X: []T, // Pre-allocated solution array [n×nrhs]
	S: []$Real, // Pre-allocated scaling factors (size n)
	ferr: []Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	fact: FactorizationOption, // Factorization option
	uplo: MatrixRegion, // Storage format
) -> (
	equed: EquilibrationState,
	rcond: Real,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(S) >= n, "Scaling array too small")
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
	equed_c := u8(EquilibrationState.None)

	when T == complex64 {
		lapack.cppsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), &equed_c, raw_data(S), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zppsvx_(&fact_c, &uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), &equed_c, raw_data(S), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	equed = EquilibrationState(equed_c)
	ok = (info == 0)
	return equed, rcond, info, ok
}

// ===================================================================================
// TRIANGULAR SOLVERS (using Cholesky factorization)
// ===================================================================================

// Solve using existing Cholesky factorization
solve_triangular_packed_cholesky :: proc(
	AP: []$T, // Cholesky factorization (from pptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.spptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dpptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.cpptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.zpptrs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================

// Invert positive definite packed matrix using Cholesky factorization
invert_packed_cholesky :: proc(
	AP: []$T, // Cholesky factor (modified to inverse)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.spptri_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == f64 {
		lapack.dpptri_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == complex64 {
		lapack.cpptri_(&uplo_c, &n_blas, raw_data(AP), &info)
	} else when T == complex128 {
		lapack.zpptri_(&uplo_c, &n_blas, raw_data(AP), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number using Cholesky factorization
estimate_condition_packed_cholesky :: proc {
	estimate_condition_packed_cholesky_real,
	estimate_condition_packed_cholesky_complex,
}

// Real packed Cholesky condition estimation (f32/f64)
estimate_condition_packed_cholesky_real :: proc(
	AP: []$T, // Cholesky factor
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
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.sppcon_(&uplo_c, &n_blas, raw_data(AP), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dppcon_(&uplo_c, &n_blas, raw_data(AP), &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// Complex packed Cholesky condition estimation (complex64/complex128)
estimate_condition_packed_cholesky_complex :: proc(
	AP: []$Cmplx, // Cholesky factor
	anorm: $Real, // 1-norm of original matrix
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion, // Storage format
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when Cmplx == complex64 {
		lapack.cppcon_(&uplo_c, &n_blas, raw_data(AP), &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zppcon_(&uplo_c, &n_blas, raw_data(AP), &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Iterative refinement for packed positive definite systems
refine_packed_cholesky :: proc {
	refine_packed_cholesky_real,
// refine_packed_cholesky_complex, // TODO: Not yet implemented
}

// Real packed Cholesky iterative refinement (f32/f64)
refine_packed_cholesky_real :: proc(
	AP: []$T, // Original packed matrix
	AFP: []T, // Cholesky factorization
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
		lapack.spprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Complex packed Cholesky iterative refinement (complex64/complex128)
refine_packed_cholesky_complex :: proc {
	refine_packed_cholesky_complex64,
	refine_packed_cholesky_complex128,
}

refine_packed_cholesky_complex64 :: proc(
	AP: []complex64, // Original packed matrix
	AFP: []complex64, // Cholesky factorization
	B: []complex64, // Original RHS [n×nrhs]
	X: []complex64, // Solution (refined on output) [n×nrhs]
	ferr: []f32, // Pre-allocated forward error bounds (size nrhs)
	berr: []f32, // Pre-allocated backward error bounds (size nrhs)
	work: []complex64, // Pre-allocated workspace (size 2*n)
	rwork: []f32, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
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

	lapack.cpprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)

	ok = (info == 0)
	return info, ok
}

refine_packed_cholesky_complex128 :: proc(
	AP: []complex128, // Original packed matrix
	AFP: []complex128, // Cholesky factorization
	B: []complex128, // Original RHS [n×nrhs]
	X: []complex128, // Solution (refined on output) [n×nrhs]
	ferr: []f64, // Pre-allocated forward error bounds (size nrhs)
	berr: []f64, // Pre-allocated backward error bounds (size nrhs)
	work: []complex128, // Pre-allocated workspace (size 2*n)
	rwork: []f64, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) {
	assert(validate_packed_storage(n, len(AP)), "AP array too small")
	assert(validate_packed_storage(n, len(AFP)), "AFP array too small")
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

	lapack.zpprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// EQUILIBRATION
// ===================================================================================

// Compute equilibration scaling factors for positive definite matrices
equilibrate_packed_positive_definite :: proc {
	equilibrate_packed_positive_definite_real,
// equilibrate_packed_positive_definite_complex, // TODO: Not yet implemented
}

// Real packed positive definite equilibration (f32/f64)
equilibrate_packed_positive_definite_real :: proc(
	AP: []$T, // Packed matrix
	S: []T, // Pre-allocated scaling factors (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	scond: T,
	amax: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.sppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	} else when T == f64 {
		lapack.dppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	}

	ok = (info == 0)
	return scond, amax, info, ok
}

// Complex packed positive definite equilibration (complex64/complex128)
equilibrate_packed_positive_definite_complex :: proc {
	equilibrate_packed_positive_definite_complex64,
	equilibrate_packed_positive_definite_complex128,
}

equilibrate_packed_positive_definite_complex64 :: proc(
	AP: []complex64, // Packed matrix
	S: []f32, // Pre-allocated scaling factors (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool,
) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	lapack.cppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)

	ok = (info == 0)
	return scond, amax, info, ok
}

equilibrate_packed_positive_definite_complex128 :: proc(
	AP: []complex128, // Packed matrix
	S: []f64, // Pre-allocated scaling factors (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool,
) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	lapack.zppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)

	ok = (info == 0)
	return scond, amax, info, ok
}

// ===================================================================================
// UTILITY FUNCTIONS
// ===================================================================================

// Check if matrix is positive definite by attempting Cholesky factorization
is_positive_definite_packed :: proc(
	AP: []$T, // Packed matrix (preserved)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
	allocator := context.allocator,
) -> bool {
	// Make a copy for testing
	AP_copy := make([]T, len(AP), allocator)
	defer delete(AP_copy, allocator)
	copy(AP_copy, AP)

	// Try Cholesky factorization
	info, ok := cholesky_factorize_packed(AP_copy, n, uplo)
	return ok
}

// Compute determinant from Cholesky factorization
determinant_packed_cholesky :: proc(
	AP: []$T, // Cholesky factor
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> T where T == f32 || T == f64 {
	det := T(1)

	// Product of diagonal elements (squared for Cholesky factor)
	for i in 0 ..< n {
		diag := packed_diagonal_element(AP, n, i, uplo)
		det *= diag * diag // L*L^T, so diagonal contributes twice
	}

	return det
}

// Compute log determinant (more numerically stable for large matrices)
log_determinant_packed_cholesky :: proc(
	AP: []$T, // Cholesky factor
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> T where T == f32 || T == f64 {
	log_det := T(0)

	// Sum of log diagonal elements (doubled for Cholesky factor)
	for i in 0 ..< n {
		diag := packed_diagonal_element(AP, n, i, uplo)
		if diag > 0 {
			log_det += 2 * math.ln(diag) // L*L^T, so log(diag) contributes twice
		} else {
			return T(-math.INF_F64) // Matrix is not positive definite
		}
	}

	return log_det
}
