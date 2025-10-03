package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX OPERATIONS
//
// This file provides operations specifically for positive definite banded matrices:
// - PB-class LAPACK routines (Cholesky factorization, solvers, refinement)
// - Condition number estimation for positive definite banded matrices
// - Equilibration and scaling operations
//
// These routines handle symmetric/Hermitian positive definite banded matrices
// using specialized algorithms that are more efficient than general banded methods.
// ===================================================================================

// ===================================================================================
// PROC GROUPS FOR OVERLOADING
// ===================================================================================

// Cholesky factorization for positive definite banded matrices
banded_cholesky :: proc {
	banded_cholesky_real,
	banded_cholesky_complex,
}

// Solve using pre-computed Cholesky factorization
banded_solve_cholesky :: proc {
	banded_solve_cholesky_real,
	banded_solve_cholesky_complex,
}

// Simple driver: factor and solve in one call
banded_solve_pd :: proc {
	banded_solve_pd_real,
	banded_solve_pd_complex,
}

// Expert driver for positive definite banded systems
banded_solve_pd_expert :: proc {
	banded_solve_pd_expert_real,
	banded_solve_pd_expert_complex,
}

// Iterative refinement for positive definite banded systems
banded_refine_pd :: proc {
	banded_refine_pd_real,
	banded_refine_pd_complex,
}

// Condition number estimation for positive definite banded matrices
banded_condition_pd :: proc {
	banded_condition_pd_real,
	banded_condition_pd_complex,
}

// Split Cholesky factorization
banded_split_cholesky :: proc {
	banded_split_cholesky_real,
	banded_split_cholesky_complex,
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query workspace for Cholesky factorization (no workspace needed)
query_workspace_banded_cholesky :: proc($T: typeid, n, kd: int) -> (work: int, rwork: int, iwork: int) {
	return 0, 0, 0 // PBTRF requires no workspace
}

// Query workspace for condition number estimation
query_workspace_banded_condition_pd :: proc($T: typeid, n: int) -> (work: int, rwork: int, iwork: int) {
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for iterative refinement
query_workspace_banded_refine_pd :: proc($T: typeid, n: int) -> (work: int, rwork: int, iwork: int) {
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for expert solver
query_workspace_banded_solve_pd_expert :: proc($T: typeid, n: int) -> (work: int, rwork: int, iwork: int) {
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query array sizes for expert solver
query_result_sizes_banded_solve_pd_expert :: proc(n, nrhs: int) -> (S_size: int, X_rows: int, X_cols: int, ferr_size: int, berr_size: int) {
	return n, n, nrhs, nrhs, nrhs
}

// Query array sizes for equilibration
query_result_sizes_banded_equilibrate_pd :: proc(n: int) -> (S_size: int) {
	return n
}

// Query array sizes for refinement
query_result_sizes_banded_refine_pd :: proc(nrhs: int) -> (ferr_size: int, berr_size: int) {
	return nrhs, nrhs
}

// ===================================================================================
// CHOLESKY FACTORIZATION (PBTRF)
// ===================================================================================

// Cholesky factorization of positive definite banded matrix (real version)
banded_cholesky_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with Cholesky factor)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == f64 {
		lapack.dpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// Cholesky factorization of positive definite banded matrix (complex version)
banded_cholesky_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - overwritten with Cholesky factor)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when Cmplx == complex128 {
		lapack.zpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SOLVE WITH CHOLESKY FACTORIZATION (PBTRS)
// ===================================================================================

// Solve using pre-computed Cholesky factorization (real version)
banded_solve_cholesky_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Cholesky factorization from banded_cholesky
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve using pre-computed Cholesky factorization (complex version)
banded_solve_cholesky_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Cholesky factorization from banded_cholesky
	B: ^Matrix(Cmplx), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when Cmplx == complex128 {
		lapack.zpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SIMPLE DRIVER (PBSV)
// ===================================================================================

// Solve positive definite banded linear system: factor and solve in one call (real version)
banded_solve_pd_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with Cholesky factor)
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve positive definite banded linear system: factor and solve in one call (complex version)
banded_solve_pd_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - overwritten with Cholesky factor)
	B: ^Matrix(Cmplx), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when Cmplx == complex128 {
		lapack.zpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT DRIVER (PBSVX)
// ===================================================================================

// Expert solve for positive definite banded system (real version)
banded_solve_pd_expert_real :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output)
	AFB: ^BandedMatrix(T), // Factored matrix (input/output)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	S: []T, // Scaling factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Solution matrix (output)
	rcond: ^T, // Reciprocal condition number (output)
	ferr: []T, // Forward error bounds (output)
	berr: []T, // Backward error bounds (output)
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(S) >= int(n), "Scaling array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Expert solve for positive definite banded system (complex version)
banded_solve_pd_expert_complex :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output)
	AFB: ^BandedMatrix(Cmplx), // Factored matrix (input/output)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	S: []$Real, // Scaling factors (input/output)
	B: ^Matrix(Cmplx), // Right-hand side (input/output)
	X: ^Matrix(Cmplx), // Solution matrix (output)
	rcond: ^Real, // Reciprocal condition number (output)
	ferr: []Real, // Forward error bounds (output)
	berr: []Real, // Backward error bounds (output)
	work: []Cmplx, // Workspace
	rwork: []Real, // Real workspace
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(S) >= int(n), "Scaling array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (PBRFS)
// ===================================================================================

// Iterative refinement for positive definite banded systems (real version)
banded_refine_pd_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Original banded matrix
	AFB: ^BandedMatrix(T), // Factorized matrix from banded_cholesky
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Iterative refinement for positive definite banded systems (complex version)
banded_refine_pd_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Original banded matrix
	AFB: ^BandedMatrix(Cmplx), // Factorized matrix from banded_cholesky
	B: ^Matrix(Cmplx), // Right-hand side matrix
	X: ^Matrix(Cmplx), // Solution matrix (input/output)
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// SPLIT CHOLESKY FACTORIZATION (PBSTF)
// ===================================================================================

// Split Cholesky factorization for positive definite banded matrix (real version)
// Computes split factor S from L^T*L where L = S*S^T
banded_split_cholesky_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == f64 {
		lapack.dpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// Split Cholesky factorization for positive definite banded matrix (complex version)
// Computes split factor S from L^H*L where L = S*S^H
banded_split_cholesky_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output)
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when Cmplx == complex128 {
		lapack.zpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}
// ===================================================================================
// CONDITION NUMBER ESTIMATION (PBCON)
// ===================================================================================

// Estimate the reciprocal condition number of a positive definite banded matrix (real version)
banded_condition_pd_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Cholesky factorization from banded_cholesky
	anorm: T, // 1-norm of the original matrix (before factorization)
	rcond: ^T, // Reciprocal condition number estimate (output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	// Validate inputs
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	uplo_c := cast(u8)uplo
	anorm_val := anorm

	when T == f32 {
		lapack.spbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm_val, rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm_val, rcond, raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Estimate the reciprocal condition number of a positive definite banded matrix (complex version)
banded_condition_pd_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Cholesky factorization from banded_cholesky
	anorm: $Real, // 1-norm of the original matrix (before factorization)
	rcond: ^Real, // Reciprocal condition number estimate (output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	// Validate inputs
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	uplo_c := cast(u8)uplo
	anorm_val := anorm

	when Cmplx == complex64 {
		lapack.cpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm_val, rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm_val, rcond, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}
