package openblas

import lapack "./f77"
import "base:intrinsics"


// ===================================================================================
// TRIANGULAR LINEAR SOLVERS AND MATRIX INVERSION
// ===================================================================================
//
// This file provides linear algebra operations for full-storage triangular matrices:
// - Triangular system solving (TRTRS)
// - Triangular matrix inversion (TRTRI)
// - Iterative refinement for triangular systems (TRRFS)
//
// All functions use the non-allocating API pattern with pre-allocated arrays.

// ===================================================================================
// TRIANGULAR SYSTEM SOLVING (TRTRS)
// ===================================================================================

// Solve triangular system Ax = b or AX = B
// Supports all numeric types: f32, f64, complex64, complex128
tri_solve :: proc(
	A: []$T, // Triangular matrix [n×n]
	B: []T, // RHS matrix/vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	lda, ldb: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
	assert(len(B) >= n * nrhs, "B array too small for column-major storage")
	assert(len(B) >= ldb * nrhs, "B array too small for given leading dimension")
	assert(ldb >= max(1, n), "Leading dimension ldb too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	lda_blas := Blas_Int(lda)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.strtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtrtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctrtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztrtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIANGULAR MATRIX INVERSION (TRTRI)
// ===================================================================================

// Invert triangular matrix in-place
// Supports all numeric types: f32, f64, complex64, complex128
tri_invert :: proc(
	A: []$T, // Triangular matrix (modified to inverse) [n×n]
	n: int, // Matrix dimension
	lda: int, // Leading dimension
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")

	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)

	when T == f32 {
		lapack.strtri_(&uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &info)
	} else when T == f64 {
		lapack.dtrtri_(&uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &info)
	} else when T == complex64 {
		lapack.ctrtri_(&uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &info)
	} else when T == complex128 {
		lapack.ztrtri_(&uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (TRRFS)
// ===================================================================================

// Iterative refinement for triangular systems
tri_refine :: proc {
	tri_refine_real,
	tri_refine_complex,
}

// Real triangular iterative refinement (f32/f64)
tri_refine_real :: proc(
	A: []$T, // Triangular matrix [n×n]
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	lda, ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	lda_blas := Blas_Int(lda)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == f32 {
		lapack.strrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtrrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex triangular iterative refinement (complex64/complex128)
tri_refine_complex :: proc(
	A: []$Cmplx, // Triangular matrix [n×n]
	B: []Cmplx, // Original RHS [n×nrhs]
	X: []Cmplx, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	lda, ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	lda_blas := Blas_Int(lda)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when Cmplx == complex64 {
		lapack.ctrrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztrrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace size for triangular refinement
query_workspace_tri_refine :: proc(A: Triangular($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := A.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork (per n and nrhs)
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork (per n)
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Solve single triangular system (single RHS)
tri_solve_vector :: proc(
	A: []$T,
	b: []T, // Vector (modified to solution)
	n: int,
	lda: int,
	uplo: MatrixRegion = .Upper,
	trans: TransposeMode = .None,
	diag: DiagonalType = .NonUnit,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	return tri_solve(A, b, n, 1, lda, max(1, n), uplo, trans, diag)
}

// Check if triangular matrix is invertible (non-singular)
is_invertible_triangular :: proc(A: []$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> bool where is_float(T) || is_complex(T) {
	if diag == .Unit {
		return true // Unit triangular matrices are always invertible
	}

	// Check diagonal elements for zeros
	for i in 0 ..< n {
		diag_elem := A[i + i * lda]
		when is_complex(T) {
			if abs(diag_elem) == 0 {
				return false
			}
		} else {
			if diag_elem == 0 {
				return false
			}
		}
	}
	return true
}

// ===================================================================================
// BANDED TRIANGULAR SYSTEM SOLVING (TBTRS)
// ===================================================================================

// Solve banded triangular system Ax = b or AX = B
// AB is stored in banded format with kd super/sub-diagonals
tri_solve_banded :: proc(
	AB: []$T, // Banded triangular matrix [ldab×n]
	B: []T, // RHS matrix/vectors (modified to solution) [n×nrhs]
	n, kd, nrhs: int, // Matrix dimension, bandwidth, and number of RHS
	ldab, ldb: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(ldab >= kd + 1, "Leading dimension ldab too small for bandwidth")
	assert(len(AB) >= ldab * n, "AB array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(ldb >= max(1, n), "Leading dimension ldb too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	kd_blas := Blas_Int(kd)
	nrhs_blas := Blas_Int(nrhs)
	ldab_blas := Blas_Int(ldab)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.stbtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtbtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctbtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztbtrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED TRIANGULAR CONDITION NUMBER ESTIMATION (TBCON)
// ===================================================================================

// Estimate condition number of banded triangular matrix
tri_condition_banded :: proc {
	tri_condition_banded_real,
	tri_condition_banded_complex,
}

// Real banded triangular condition number estimation (f32/f64)
tri_condition_banded_real :: proc(
	AB: []$T, // Banded triangular matrix [ldab×n]
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, kd: int, // Matrix dimension and bandwidth
	ldab: int, // Leading dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(ldab >= kd + 1, "Leading dimension ldab too small for bandwidth")
	assert(len(AB) >= ldab * n, "AB array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	kd_blas := Blas_Int(kd)
	ldab_blas := Blas_Int(ldab)

	when T == f32 {
		lapack.stbcon_(&norm_c, &uplo_c, &diag_c, &n_blas, &kd_blas, raw_data(AB), &ldab_blas, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtbcon_(&norm_c, &uplo_c, &diag_c, &n_blas, &kd_blas, raw_data(AB), &ldab_blas, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Complex banded triangular condition number estimation (complex64/complex128)
tri_condition_banded_complex :: proc(
	AB: []$Cmplx, // Banded triangular matrix [ldab×n]
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []$Real, // Pre-allocated real workspace (size n)
	n, kd: int, // Matrix dimension and bandwidth
	ldab: int, // Leading dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(ldab >= kd + 1, "Leading dimension ldab too small for bandwidth")
	assert(len(AB) >= ldab * n, "AB array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	kd_blas := Blas_Int(kd)
	ldab_blas := Blas_Int(ldab)

	when Cmplx == complex64 {
		lapack.ctbcon_(&norm_c, &uplo_c, &diag_c, &n_blas, &kd_blas, raw_data(AB), &ldab_blas, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztbcon_(&norm_c, &uplo_c, &diag_c, &n_blas, &kd_blas, raw_data(AB), &ldab_blas, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// BANDED TRIANGULAR ITERATIVE REFINEMENT (TBRFS)
// ===================================================================================

// Iterative refinement for banded triangular systems
tri_refine_banded :: proc {
	tri_refine_banded_real,
	tri_refine_banded_complex,
}

// Real banded triangular iterative refinement (f32/f64)
tri_refine_banded_real :: proc(
	AB: []$T, // Banded triangular matrix [ldab×n]
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, kd, nrhs: int, // Matrix dimension, bandwidth, and number of RHS
	ldab, ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(ldab >= kd + 1, "Leading dimension ldab too small for bandwidth")
	assert(len(AB) >= ldab * n, "AB array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	kd_blas := Blas_Int(kd)
	nrhs_blas := Blas_Int(nrhs)
	ldab_blas := Blas_Int(ldab)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == f32 {
		lapack.stbrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtbrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex banded triangular iterative refinement (complex64/complex128)
tri_refine_banded_complex :: proc(
	AB: []$Cmplx, // Banded triangular matrix [ldab×n]
	B: []Cmplx, // Original RHS [n×nrhs]
	X: []Cmplx, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, kd, nrhs: int, // Matrix dimension, bandwidth, and number of RHS
	ldab, ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(ldab >= kd + 1, "Leading dimension ldab too small for bandwidth")
	assert(len(AB) >= ldab * n, "AB array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")
	assert(len(X) >= n * nrhs || len(X) >= ldx * nrhs, "X array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	kd_blas := Blas_Int(kd)
	nrhs_blas := Blas_Int(nrhs)
	ldab_blas := Blas_Int(ldab)
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when Cmplx == complex64 {
		lapack.ctbrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztbrfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &kd_blas, &nrhs_blas, raw_data(AB), &ldab_blas, raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS FOR BANDED TRIANGULAR
// ===================================================================================

// Query workspace size for banded triangular condition estimation
query_workspace_tri_condition_banded :: proc(AB: Triangular($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := AB.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork (per n)
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork (per n)
	}
}

// Query workspace size for banded triangular refinement
query_workspace_tri_refine_banded :: proc(AB: Triangular($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := AB.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork (per n)
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork (per n)
	}
}
