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
pack_sym_cholesky :: proc(
	AP: []$T, // Packed matrix (modified to Cholesky factor)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
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


	return info, info == 0
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Solve positive definite system using packed storage (combined factorization + solve)
pack_sym_solve :: proc(
	AP: []$T, // Packed matrix (modified to Cholesky factor)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
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


	return info, info == 0
}

// ===================================================================================
// EXPERT SOLVERS WITH CONDITION ESTIMATION
// ===================================================================================

// Expert solver with equilibration and condition estimation
pack_sym_solve_expert :: proc {
	pack_sym_solve_expert_real,
	pack_sym_solve_expert_complex,
}

// Real positive definite packed expert solver (f32/f64)
pack_sym_solve_expert_real :: proc(
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
) where is_float(T) {
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

	return equed, rcond, info, info == 0
}

// Complex positive definite packed expert solver (complex64/complex128)
pack_sym_solve_expert_complex :: proc(
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

	return equed, rcond, info, info == 0
}

// ===================================================================================
// TRIANGULAR SOLVERS (using Cholesky factorization)
// ===================================================================================

// Solve using existing Cholesky factorization
pack_sym_solve_factorized :: proc(
	AP: []$T, // Cholesky factorization (from pptrf)
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
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

	return info, info == 0
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================

// Invert positive definite packed matrix using Cholesky factorization
pack_sym_cholesky_invert :: proc(
	AP: []$T, // Cholesky factor (modified to inverse)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
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

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number using Cholesky factorization
pack_sym_condition :: proc {
	pack_sym_condition_real,
	pack_sym_condition_complex,
}

// Real packed Cholesky condition estimation (f32/f64)
pack_sym_condition_real :: proc(
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
) where is_float(T) {
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

	return rcond, info, info == 0
}

// Complex packed Cholesky condition estimation (complex64/complex128)
pack_sym_condition_complex :: proc(
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
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
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

	return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Iterative refinement for packed positive definite systems
pack_sym_refine :: proc {
	pack_sym_refine_real,
	pack_sym_refine_complex,
}

// Real packed Cholesky iterative refinement (f32/f64)
pack_sym_refine_real :: proc(
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
) where is_float(T) {
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

	return info, info == 0
}

// Complex packed Cholesky iterative refinement (complex64/complex128)
pack_sym_refine_complex :: proc(
	AP: []$Cmplx, // Original packed matrix
	AFP: []Cmplx, // Cholesky factorization
	B: []Cmplx, // Original RHS [n×nrhs]
	X: []Cmplx, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
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

	when Cmplx == complex64 {
		lapack.cpprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpprfs_(&uplo_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(AFP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// EQUILIBRATION
// ===================================================================================

// Compute equilibration scaling factors for positive definite matrices
pack_sym_equilibrate :: proc {
	pack_sym_equilibrate_real,
	pack_sym_equilibrate_complex,
}

// Real packed positive definite equilibration (f32/f64)
pack_sym_equilibrate_real :: proc(
	AP: []$T, // Packed matrix
	S: []T, // Pre-allocated scaling factors (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	scond: T,
	amax: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.sppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	} else when T == f64 {
		lapack.dppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Complex packed positive definite equilibration (complex64/complex128)
pack_sym_equilibrate_complex :: proc(
	AP: []$Cmplx, // Packed matrix
	S: []$Real, // Pre-allocated scaling factors (size n)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> (
	scond: Real,
	amax: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when Cmplx == complex64 {
		lapack.cppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	} else when Cmplx == complex128 {
		lapack.zppequ_(&uplo_c, &n_blas, raw_data(AP), raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// UTILITY FUNCTIONS
// ===================================================================================

// Check if matrix is positive definite by attempting Cholesky factorization
pack_sym_is_positive_definite :: proc(
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
	info, ok := pack_sym_cholesky(AP_copy, n, uplo)
	return ok
}

// Compute determinant from Cholesky factorization
pack_sym_determinant :: proc(
	AP: []$T, // Cholesky factor
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> T where is_float(T) {
	det := T(1)

	// Product of diagonal elements (squared for Cholesky factor)
	for i in 0 ..< n {
		diag := pack_sym_diagonal_get(AP, n, i, uplo)
		det *= diag * diag // L*L^T, so diagonal contributes twice
	}

	return det
}

// Compute log determinant (more numerically stable for large matrices)
pack_sym_log_determinant :: proc(
	AP: []$T, // Cholesky factor
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Storage format
) -> T where is_float(T) {
	log_det := T(0)

	// Sum of log diagonal elements (doubled for Cholesky factor)
	for i in 0 ..< n {
		diag := pack_sym_diagonal_get(AP, n, i, uplo)
		if diag > 0 {
			log_det += 2 * math.ln(diag) // L*L^T, so log(diag) contributes twice
		} else {
			return T(-math.INF_F64) // Matrix is not positive definite
		}
	}

	return log_det
}
