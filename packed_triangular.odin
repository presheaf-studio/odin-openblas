package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// PACKED TRIANGULAR MATRIX BASE TYPES AND OPERATIONS
// ===================================================================================
//
// This file provides the base types and core operations for triangular matrices
// stored in packed format. Triangular matrices store only the upper or lower triangle,
// requiring n*(n+1)/2 elements instead of n² for an n×n matrix.
//
// For conversion operations between packed and full formats, see:
// - packed_triangular_conversion.odin
//
// Includes:
// - Triangular solvers (tptrs/dtptrs/ctptrs/ztptrs)
// - Matrix inversion (tptri/dtptri/ctptri/ztptri)
// - Condition number estimation (tpcon/dtpcon/ctpcon/ztpcon)
// - Iterative refinement (tprfs/dtprfs/ctprfs/ztprfs)

// ===================================================================================
// BASE TYPES
// ===================================================================================

// Packed triangular matrix type
PackedTriangular :: struct($T: typeid) {
	data: []T, // Packed storage array (n*(n+1)/2 elements)
	n:    int, // Matrix dimension
	uplo: MatrixRegion, // Storage region (Upper or Lower)
	diag: DiagonalType, // Diagonal type (Unit or NonUnit)
}

// ===================================================================================
// PACKED MATRIX CREATION AND MANAGEMENT
// ===================================================================================

// Create empty PackedTriangular matrix
// Supports all numeric types: f32, f64, complex64, complex128
pack_tri_make :: proc($T: typeid, n: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit, allocator := context.allocator) -> PackedTriangular(T) where is_float(T) || is_complex(T) {
	size := pack_tri_storage_size(n)
	data := make([]T, size, allocator)

	return PackedTriangular(T){data = data, n = n, uplo = uplo, diag = diag}
}

// Delete PackedTriangular matrix
// Supports all numeric types: f32, f64, complex64, complex128
pack_tri_delete :: proc(packed: ^PackedTriangular($T), allocator := context.allocator) where is_float(T) || is_complex(T) {
	delete(packed.data, allocator)
	packed.data = nil
	packed.n = 0
}

// ===================================================================================
// TRIANGULAR MATRIX PROPERTIES
// ===================================================================================

// Calculate packed storage size for triangular matrix
pack_tri_storage_size :: proc(n: int) -> int {
	return n * (n + 1) / 2
}


// ===================================================================================
// TRIANGULAR SOLVERS
// ===================================================================================

// Solve triangular system using packed storage
solve_packed_triangular :: proc(
	AP: []$T, // Triangular packed matrix
	B: []T, // RHS vectors (modified to solution) [n×nrhs]
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb: int, // Leading dimension of B
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(B) >= n * nrhs || len(B) >= ldb * nrhs, "B array too small")

	uplo_c := u8(uplo)
	trans_c := u8(trans)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	nrhs_blas := Blas_Int(nrhs)
	ldb_blas := Blas_Int(ldb)

	when T == f32 {
		lapack.stptrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtptrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctptrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztptrs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================

// Invert triangular packed matrix
invert_packed_triangular :: proc(
	AP: []$T, // Triangular packed matrix (modified to inverse)
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")

	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.stptri_(&uplo_c, &diag_c, &n_blas, raw_data(AP), &info)
	} else when T == f64 {
		lapack.dtptri_(&uplo_c, &diag_c, &n_blas, raw_data(AP), &info)
	} else when T == complex64 {
		lapack.ctptri_(&uplo_c, &diag_c, &n_blas, raw_data(AP), &info)
	} else when T == complex128 {
		lapack.ztptri_(&uplo_c, &diag_c, &n_blas, raw_data(AP), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number of real triangular packed matrix (f32/f64)
estimate_condition_packed_triangular_real :: proc(
	AP: []$T, // Triangular packed matrix
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n: int, // Matrix dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.stpcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(AP), &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtpcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(AP), &rcond, raw_data(work), raw_data(iwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// Estimate condition number of complex triangular packed matrix (complex64/complex128)
estimate_condition_packed_triangular_complex :: proc(
	AP: []$Cmplx, // Triangular packed matrix
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []$Real, // Pre-allocated real workspace (size n)
	n: int, // Matrix dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)

	when Cmplx == complex64 {
		lapack.ctpcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(AP), &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztpcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(AP), &rcond, raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// Overloaded proc group for condition number estimation
estimate_condition_packed_triangular :: proc {
	estimate_condition_packed_triangular_real,
	estimate_condition_packed_triangular_complex,
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Iterative refinement for real triangular packed systems (f32/f64)
refine_packed_triangular_real :: proc(
	AP: []$T, // Triangular packed matrix
	B: []T, // Original RHS [n×nrhs]
	X: []T, // Solution (refined on output) [n×nrhs]
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
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
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when T == f32 {
		lapack.stprfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtprfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Iterative refinement for complex triangular packed systems (complex64/complex128)
refine_packed_triangular_complex :: proc(
	AP: []$Cmplx, // Triangular packed matrix
	B: []Cmplx, // Original RHS [n×nrhs]
	X: []Cmplx, // Solution (refined on output) [n×nrhs]
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	n, nrhs: int, // Matrix dimension and number of RHS
	ldb, ldx: int, // Leading dimensions
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	trans: TransposeMode = .None, // Transpose operation
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
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
	ldb_blas := Blas_Int(ldb)
	ldx_blas := Blas_Int(ldx)

	when Cmplx == complex64 {
		lapack.ctprfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztprfs_(&uplo_c, &trans_c, &diag_c, &n_blas, &nrhs_blas, raw_data(AP), raw_data(B), &ldb_blas, raw_data(X), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return info, ok
}

// Overloaded proc group for iterative refinement
refine_packed_triangular :: proc {
	refine_packed_triangular_real,
	refine_packed_triangular_complex,
}


// ===================================================================================
// UTILITY FUNCTIONS
// ===================================================================================

// Check if packed triangular matrix is well-conditioned
is_well_conditioned_triangular :: proc(
	AP: []$T, // Triangular packed matrix
	n: int, // Matrix dimension
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
	allocator := context.allocator,
) -> bool where is_float(T) {
	when T == f32 {
		work := make([]f32, 3 * n, allocator)
		defer delete(work, allocator)
		iwork := make([]Blas_Int, n, allocator)
		defer delete(iwork, allocator)

		rcond, info, ok := estimate_condition_packed_triangular_real(AP, work, iwork, n, .OneNorm, uplo, diag)
		return ok && rcond > T(1e-6) // Reasonable threshold for single precision
	} else when T == f64 {
		work := make([]f64, 3 * n, allocator)
		defer delete(work, allocator)
		iwork := make([]Blas_Int, n, allocator)
		defer delete(iwork, allocator)

		rcond, info, ok := estimate_condition_packed_triangular_real(AP, work, iwork, n, .OneNorm, uplo, diag)
		return ok && rcond > T(1e-12) // Reasonable threshold for double precision
	} else {
		return false
	}
}
