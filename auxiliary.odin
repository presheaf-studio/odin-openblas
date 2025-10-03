package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:strings"

// ===================================================================================
// AUXILIARY AND UTILITY FUNCTIONS
// Helper routines for matrix operations, norms, and utilities
// ===================================================================================

LapackVersion :: struct {
	major: int,
	minor: int,
	patch: int,
}

// Get LAPACK version
get_lapack_version :: proc() -> LapackVersion {
	major, minor, patch: Blas_Int
	lapack.ilaver_(&major, &minor, &patch)
	return LapackVersion{major = int(major), minor = int(minor), patch = int(patch)}
}

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
m_condition_estimate_real :: proc(
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
	lda := Blas_Int(A.ld)

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
m_condition_estimate_complex :: proc(
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
	lda := Blas_Int(A.ld)

	// Convert norm type to char
	norm_c := cast(u8)norm

	anorm_copy := anorm

	when T == complex64 {
		lapack.cgecon_(&norm_c, &n_int, cast(^complex64)raw_data(A.data), &lda, &anorm_copy, &rcond, cast(^complex64)raw_data(work), raw_data(rwork), &info, 1)
	} else when T == complex128 {
		lapack.zgecon_(&norm_c, &n_int, cast(^complex128)raw_data(A.data), &lda, &anorm_copy, &rcond, cast(^complex128)raw_data(work), raw_data(rwork), &info, 1)
	}
	return rcond, info, info == 0
}

// Procedure group for condition estimation
m_condition_estimate :: proc {
	m_condition_estimate_real,
	m_condition_estimate_complex,
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
m_equilibrate_real :: proc(
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
	lda := Blas_Int(A.ld)

	when T == f32 {
		lapack.sgeequ_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequ_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Compute equilibration scale factors for complex64/complex128
m_equilibrate_complex :: proc(
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
	Real == real_type_of(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)

	when T == complex64 {
		lapack.cgeequ_(&m_int, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == complex128 {
		lapack.zgeequ_(&m_int, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Procedure group for equilibration
m_equilibrate :: proc {
	m_equilibrate_real,
	m_equilibrate_complex,
}

// Improved equilibration with better algorithm (LAPACK 3.x)
// More robust handling of over/underflow

// Compute improved equilibration scale factors for f32/f64
m_equilibrate_improved_real :: proc(
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
	lda := Blas_Int(A.ld)

	when T == f32 {
		lapack.sgeequb_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequb_(&m_int, &n_int, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Compute improved equilibration scale factors for complex64/complex128
m_equilibrate_improved_complex :: proc(
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
	Real == real_type_of(T) {
	m := A.rows
	n := A.cols
	assert(len(R) >= m, "Row scale factors array too small")
	assert(len(C) >= n, "Column scale factors array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)

	when T == complex64 {
		lapack.cgeequb_(&m_int, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == complex128 {
		lapack.zgeequb_(&m_int, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	ok = info == 0
	return rowcnd, colcnd, amax, info, ok
}

// Procedure group for improved equilibration
m_equilibrate_improved :: proc {
	m_equilibrate_improved_real,
	m_equilibrate_improved_complex,
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
// MATRIX COPY OPERATIONS (LACPY)
// ===================================================================================

// Copy matrix or submatrix
m_copy_matrix :: proc {
	m_copy_matrix_f32,
	m_copy_matrix_f64,
	m_copy_matrix_c64,
	m_copy_matrix_c128,
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32), // Source matrix
	B: ^Matrix(f32), // Destination matrix
	region := MatrixRegion.Full, // Region to copy (Full, Upper, Lower)
) {
	// Validate matrices
	assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	uplo_c := cast(u8)region

	lapack.slacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64), // Source matrix
	B: ^Matrix(f64), // Destination matrix
	region := MatrixRegion.Full, // Region to copy (Full, Upper, Lower)
) {
	// Validate matrices
	assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	uplo_c := cast(u8)region

	lapack.dlacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (c64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64), // Source matrix
	B: ^Matrix(complex64), // Destination matrix
	region := MatrixRegion.Full, // Region to copy (Full, Upper, Lower)
) {
	// Validate matrices
	assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	uplo_c := cast(u8)region

	lapack.clacpy_(&uplo_c, &m, &n, cast(^complex64)raw_data(A.data), &lda, cast(^complex64)raw_data(B.data), &ldb, 1)
}

// Copy matrix (c128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128), // Source matrix
	B: ^Matrix(complex128), // Destination matrix
	region := MatrixRegion.Full, // Region to copy (Full, Upper, Lower)
) {
	// Validate matrices
	assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	uplo_c := cast(u8)region

	lapack.zlacpy_(&uplo_c, &m, &n, cast(^complex128)raw_data(A.data), &lda, cast(^complex128)raw_data(B.data), &ldb, 1)
}

// ===================================================================================
// COMPLEX VECTOR CONJUGATION (LACGV)
// ===================================================================================

// Conjugate complex vector proc group
v_conjugate :: proc {
	v_conjugate_c64,
	v_conjugate_c128,
}

// Conjugate complex vector (c64)
// Computes X[i] = conj(X[i]) for i = 0, incx, 2*incx, ..., (n-1)*incx
v_conjugate_c64 :: proc(X: []complex64, incx: int = 1) {
	if len(X) == 0 do return

	n := Blas_Int(len(X))
	incx_int := Blas_Int(incx)

	lapack.clacgv_(&n, cast(^complex64)raw_data(X), &incx_int)
}

// Conjugate complex vector (c128)
// Computes X[i] = conj(X[i]) for i = 0, incx, 2*incx, ..., (n-1)*incx
v_conjugate_c128 :: proc(X: []complex128, incx: int = 1) {
	if len(X) == 0 do return

	n := Blas_Int(len(X))
	incx_int := Blas_Int(incx)

	lapack.zlacgv_(&n, cast(^complex128)raw_data(X), &incx_int)
}

// ===================================================================================
// COMPLEX × REAL MATRIX MULTIPLICATION (LACRM)
// ===================================================================================

// Complex × real matrix multiplication proc group
// Computes C = A * B where A is complex and B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64,
	m_multiply_complex_real_c128,
}

// Complex × real matrix multiplication (c64)
// Computes C = A * B where A is complex(m×n), B is real(n×n), C is complex(m×n)
// rwork must be at least size 2*m*n
m_multiply_complex_real_c64 :: proc(
	A: ^Matrix(complex64), // Complex matrix (m × n)
	B: ^Matrix(f32), // Real matrix (n × n)
	C: ^Matrix(complex64), // Complex result (m × n)
	rwork: []f32, // Real workspace (size >= 2*m*n)
) {
	m := A.rows
	n := A.cols
	assert(B.rows == n && B.cols == n, "B must be n×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= 2 * m * n, "rwork array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	lapack.clacrm_(&m_int, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(B.data), &ldb, cast(^complex64)raw_data(C.data), &ldc, raw_data(rwork))
}

// Complex × real matrix multiplication (c128)
// Computes C = A * B where A is complex(m×n), B is real(n×n), C is complex(m×n)
// rwork must be at least size 2*m*n
m_multiply_complex_real_c128 :: proc(
	A: ^Matrix(complex128), // Complex matrix (m × n)
	B: ^Matrix(f64), // Real matrix (n × n)
	C: ^Matrix(complex128), // Complex result (m × n)
	rwork: []f64, // Real workspace (size >= 2*m*n)
) {
	m := A.rows
	n := A.cols
	assert(B.rows == n && B.cols == n, "B must be n×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= 2 * m * n, "rwork array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	lapack.zlacrm_(&m_int, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(B.data), &ldb, cast(^complex128)raw_data(C.data), &ldc, raw_data(rwork))
}

// ===================================================================================
// REAL × COMPLEX MATRIX MULTIPLICATION (LARCM)
// ===================================================================================

// Real × complex matrix multiplication proc group
// Computes C = A * B where A is real and B is complex
m_multiply_real_complex :: proc {
	m_multiply_real_complex_c64,
	m_multiply_real_complex_c128,
}

// Real × complex matrix multiplication (c64)
// Computes C = A * B where A is real(m×m), B is complex(m×n), C is complex(m×n)
// rwork must be at least size 2*m*n
m_multiply_real_complex_c64 :: proc(
	A: ^Matrix(f32), // Real matrix (m × m)
	B: ^Matrix(complex64), // Complex matrix (m × n)
	C: ^Matrix(complex64), // Complex result (m × n)
	rwork: []f32, // Real workspace (size >= 2*m*n)
) {
	m := A.rows
	n := B.cols
	assert(A.cols == m, "A must be m×m")
	assert(B.rows == m, "B must be m×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= 2 * m * n, "rwork array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	lapack.clarcm_(&m_int, &n_int, raw_data(A.data), &lda, cast(^complex64)raw_data(B.data), &ldb, cast(^complex64)raw_data(C.data), &ldc, raw_data(rwork))
}

// Real × complex matrix multiplication (c128)
// Computes C = A * B where A is real(m×m), B is complex(m×n), C is complex(m×n)
// rwork must be at least size 2*m*n
m_multiply_real_complex_c128 :: proc(
	A: ^Matrix(f64), // Real matrix (m × m)
	B: ^Matrix(complex128), // Complex matrix (m × n)
	C: ^Matrix(complex128), // Complex result (m × n)
	rwork: []f64, // Real workspace (size >= 2*m*n)
) {
	m := A.rows
	n := B.cols
	assert(A.cols == m, "A must be m×m")
	assert(B.rows == m, "B must be m×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= 2 * m * n, "rwork array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	lapack.zlarcm_(&m_int, &n_int, raw_data(A.data), &lda, cast(^complex128)raw_data(B.data), &ldb, cast(^complex128)raw_data(C.data), &ldc, raw_data(rwork))
}

// ===================================================================================
// RECIPROCAL CONDITION NUMBERS FOR EIGENVALUES/SINGULAR VALUES (DISNA)
// ===================================================================================

// Reciprocal condition numbers for eigenvalues/singular values proc group
compute_reciprocal_condition_numbers :: proc {
	compute_reciprocal_condition_numbers_f32,
	compute_reciprocal_condition_numbers_f64,
}

// Job type for DISNA
DisnaJob :: enum u8 {
	Eigenvalues   = 'E', // Compute condition numbers for eigenvalues
	SingularLeft  = 'L', // Compute condition numbers for left singular vectors
	SingularRight = 'R', // Compute condition numbers for right singular vectors
}

// Compute reciprocal condition numbers for eigenvalues/singular values (f32)
// For eigenvalue problems: SEP[i] = |lambda[i] - lambda[i+1]| (gap between eigenvalues)
// For singular value problems: SEP[i] = min(D[i], D[i-1]) (minimum singular value gap)
compute_reciprocal_condition_numbers_f32 :: proc(
	D: []f32, // Eigenvalues or singular values
	SEP: []f32, // Reciprocal condition numbers (output)
	job: DisnaJob = .Eigenvalues,
	m: int = 0, // Rows (for SVD, 0 for eigenvalue)
	n: int = 0, // Cols (for SVD, 0 for eigenvalue)
) -> (
	info: Info,
	ok: bool,
) {
	// For eigenvalue problems, m and n are not used (set to 0)
	// For SVD, m and n are matrix dimensions
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)

	if job == .Eigenvalues {
		// For eigenvalue problems, m is not used, but n should be size of D
		n_int = Blas_Int(len(D))
	}

	assert(len(SEP) >= len(D), "SEP array too small")

	job_c := cast(u8)job

	lapack.sdisna_(&job_c, &m_int, &n_int, raw_data(D), raw_data(SEP), &info, 1)

	return info, info == 0
}

// Compute reciprocal condition numbers for eigenvalues/singular values (f64)
// For eigenvalue problems: SEP[i] = |lambda[i] - lambda[i+1]| (gap between eigenvalues)
// For singular value problems: SEP[i] = min(D[i], D[i-1]) (minimum singular value gap)
compute_reciprocal_condition_numbers_f64 :: proc(
	D: []f64, // Eigenvalues or singular values
	SEP: []f64, // Reciprocal condition numbers (output)
	job: DisnaJob = .Eigenvalues,
	m: int = 0, // Rows (for SVD, 0 for eigenvalue)
	n: int = 0, // Cols (for SVD, 0 for eigenvalue)
) -> (
	info: Info,
	ok: bool,
) {
	// For eigenvalue problems, m and n are not used (set to 0)
	// For SVD, m and n are matrix dimensions
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)

	if job == .Eigenvalues {
		// For eigenvalue problems, m is not used, but n should be size of D
		n_int = Blas_Int(len(D))
	}

	assert(len(SEP) >= len(D), "SEP array too small")

	job_c := cast(u8)job

	lapack.ddisna_(&job_c, &m_int, &n_int, raw_data(D), raw_data(SEP), &info, 1)

	return info, info == 0
}
