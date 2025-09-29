package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:strings"

// ===================================================================================
// AUXILIARY AND UTILITY FUNCTIONS
// Helper routines for matrix operations, norms, and utilities
// ===================================================================================

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// Estimate reciprocal condition number of factored matrices
// ===================================================================================

// Estimate reciprocal condition number of general matrix
// Matrix A must be factored (e.g., by LU decomposition)
// Query workspace for condition estimation
query_workspace_condition_estimate :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// Real types: work = 4*n, iwork = n, no rwork
		work_size = 4 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, no iwork, rwork = 2*n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = 2 * n
	}
	return
}

// Estimate condition number for f32/f64
m_condition_estimate_f32_f64 :: proc(
	A: ^Matrix($T), // Factored matrix (from getrf)
	anorm: T, // Norm of original matrix
	work: []T, // Pre-allocated workspace (size 4*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	norm: MatrixNorm = .OneNorm,
) -> (
	rcond: T,
	info: Info,
	ok: bool, // Reciprocal condition number
) where is_float(T) {
	n := A.cols
	assert(len(work) >= 4 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	n_int := Blas_Int(n)
	lda := A.ld

	// Convert norm type to char
	norm_c := cast(u8)norm

	anorm_copy := anorm

	when T == f32 {
		lapack.sgecon_(&norm_c, &n_int, raw_data(A.data), &lda, &anorm_copy, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	} else when T == f64 {
		lapack.dgecon_(&norm_c, &n_int, raw_data(A.data), &lda, &anorm_copy, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	}

	return rcond, info, info == 0
}

// Estimate condition number for complex64/complex128
m_condition_estimate_c64_c128 :: proc(
	A: ^Matrix($T), // Factored matrix (from getrf)
	anorm: $R, // Norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size 2*n)
	norm: MatrixNorm = .OneNorm,
) -> (
	rcond: R,
	info: Info,
	ok: bool, // Reciprocal condition number
) where is_complex(T),
	R == real_type_of(T) {
	n := A.cols
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 2 * n, "Real workspace too small")

	n_int := Blas_Int(n)
	lda := A.ld

	// Convert norm type to char
	norm_c := cast(u8)norm

	anorm_copy := anorm

	when T == complex64 {
		lapack.cgecon_(&norm_c, &n_int, cast(^lapack.complex)raw_data(A.data), &lda, &anorm_copy, &rcond, cast(^lapack.complex)raw_data(work), raw_data(rwork), &info, 1)
	} else when T == complex128 {
		lapack.zgecon_(&norm_c, &n_int, cast(^lapack.doublecomplex)raw_data(A.data), &lda, &anorm_copy, &rcond, cast(^lapack.doublecomplex)raw_data(work), raw_data(rwork), &info, 1)
	}
	return rcond, info, info == 0
}

// Procedure group for condition estimation
m_condition_estimate :: proc {
	m_condition_estimate_f32_f64,
	m_condition_estimate_c64_c128,
}

// Helper function to check if matrix is well-conditioned
// Returns true if reciprocal condition number > threshold (default 1E-6)
m_is_well_conditioned :: proc(rcond: $T, threshold: T) -> bool where is_float(T) {
	return rcond > threshold
}

// ===================================================================================
// MATRIX EQUILIBRATION
// Compute row and column scale factors to improve matrix conditioning
// ===================================================================================

// Compute row and column scale factors for general matrix equilibration
// Scale factors R and C are chosen so that R*A*C has rows and columns with similar norms

// Compute equilibration scale factors for f32/f64
m_equilibrate_f32_f64 :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info,
	ok: bool, // Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where is_float(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := A.ld

	when T == f32 {
		lapack.sgeequ_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequ_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Compute equilibration scale factors for complex64/complex128
m_equilibrate_c64_c128 :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []$Real, // Pre-allocated row scale factors (size m)
	C: []Real, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: Real,
	colcnd: Real,
	amax: Real,
	info: Info,
	ok: bool, // Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where is_complex(T),
	R == real_type_of(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := A.ld

	when T == complex64 {
		lapack.cgeequ_(&m_int, &n_int, cast(^lapack.complex)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == complex128 {
		lapack.zgeequ_(&m_int, &n_int, cast(^lapack.doublecomplex)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Procedure group for equilibration
m_equilibrate :: proc {
	m_equilibrate_f32_f64,
	m_equilibrate_c64_c128,
}

// Improved equilibration with better algorithm (LAPACK 3.x)
// More robust handling of over/underflow

// Compute improved equilibration scale factors for f32/f64
m_equilibrate_improved_f32_f64 :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info,
	ok: bool, // Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where is_float(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := A.ld

	when T == f32 {
		lapack.sgeequb_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequb_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Compute improved equilibration scale factors for complex64/complex128
m_equilibrate_improved_c64_c128 :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []$Real, // Pre-allocated row scale factors (size m)
	C: []Real, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: Real,
	colcnd: Real,
	amax: Real,
	info: Info,
	ok: bool, // Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where is_complex(T),
	R == real_type_of(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := A.ld

	when T == complex64 {
		lapack.cgeequb_(&m_int, &n_int, cast(^lapack.complex)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == complex128 {
		lapack.zgeequb_(&m_int, &n_int, cast(^lapack.doublecomplex)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Procedure group for improved equilibration
m_equilibrate_improved :: proc {
	m_equilibrate_improved_f32_f64,
	m_equilibrate_improved_c64_c128,
}

// Apply equilibration scale factors to a matrix
// Computes A_scaled = R * A * C where R and C are diagonal scaling matrices
m_apply_equilibration_real :: proc(
	A: ^Matrix($T),
	R: []T, // Row scale factors (from m_equilibrate)
	C: []T, // Column scale factors (from m_equilibrate)
) where is_float(T) {
	m := A.rows
	n := A.cols

	// Scale rows
	for i in 0 ..< m {
		for j in 0 ..< n {
			// Column-major indexing
			A.data[j * A.ld + i] *= R[i]
		}
	}

	// Scale columns
	for j in 0 ..< n {
		for i in 0 ..< m {
			// Column-major indexing
			A.data[j * A.ld + i] *= C[j]
		}
	}
}

m_apply_equilibration_complex :: proc(
	A: ^Matrix($T),
	R: []$S, // Real row scale factors (from m_equilibrate)
	C: []S, // Real column scale factors (from m_equilibrate)
) where (T == complex64 && S == f32) || (T == complex128 && S == f64) {
	m := A.rows
	n := A.cols

	// Scale rows
	for i in 0 ..< m {
		row_scale := T(complex(R[i], 0))
		for j in 0 ..< n {
			// Column-major indexing
			A.data[j * A.ld + i] *= row_scale
		}
	}

	// Scale columns
	for j in 0 ..< n {
		col_scale := T(complex(C[j], 0))
		for i in 0 ..< m {
			// Column-major indexing
			A.data[j * A.ld + i] *= col_scale
		}
	}
}

// Check if matrix needs equilibration
// Returns true if equilibration would significantly improve conditioning
// default threshold 0.1
m_needs_equilibration :: proc(rowcnd, colcnd: $T, threshold: T) -> bool where is_float(T) {
	return rowcnd < threshold || colcnd < threshold
}

// ===================================================================================
