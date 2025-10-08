package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"
import "core:slice"

// ===================================================================================
// MATRIX INITIALIZATION, SORTING, AND UTILITY OPERATIONS
// ===================================================================================

// Initialize matrix to specified values
initialize_matrix :: proc {
	initialize_matrix_real,
	initialize_matrix_complex,
}

// Sort vector elements
v_sort :: proc {
	v_sort_real,
}

// Sum of squares computation
v_sum_of_squares :: proc {
	v_sum_of_squares_f32_c64,
	v_sum_of_squares_f64_c128,
}

// ===================================================================================
// MATRIX INITIALIZATION
// ===================================================================================
// Sets diagonal elements to beta, off-diagonal elements to alpha

// Initialize matrix (real: f32/f64)
initialize_matrix_real :: proc(
	A: ^Matrix($T),
	alpha: T, // Off-diagonal value
	beta: T, // Diagonal value
	region := MatrixRegion.Full,
) where is_float(T) {
	assert(A.data != nil, "Matrix data cannot be empty")

	uplo_c := cast(u8)region
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	alpha_val := alpha
	beta_val := beta

	when T == f32 {
		lapack.slaset_(&uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda)
	} else when T == f64 {
		lapack.dlaset_(&uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda)
	}
}

// Initialize matrix (complex: c64/c128)
initialize_matrix_complex :: proc(
	A: ^Matrix($T),
	alpha: T, // Off-diagonal value
	beta: T, // Diagonal value
	region := MatrixRegion.Full,
) where is_complex(T) {
	assert(A.data != nil, "Matrix data cannot be empty")

	uplo_c := cast(u8)region
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	alpha_val := alpha
	beta_val := beta

	when T == complex64 {
		lapack.claset_(&uplo_c, &m, &n, cast(^complex64)&alpha_val, cast(^complex64)&beta_val, cast(^complex64)raw_data(A.data), &lda)
	} else when T == complex128 {
		lapack.zlaset_(&uplo_c, &m, &n, cast(^complex128)&alpha_val, cast(^complex128)&beta_val, cast(^complex128)raw_data(A.data), &lda)
	}
}

// ===================================================================================
// VECTOR SORTING
// ===================================================================================

// Sort vector elements (real: f32/f64)
v_sort_real :: proc(D: Vector($T), direction := SortDirection.Increasing) -> (success: bool, info: Info) where is_float(T) {
	// Validate input
	if len(D.data) == 0 {return true, 0}

	id_c := cast(u8)direction
	n := Blas_Int(D.size)
	info_val: Info

	when T == f32 {
		lapack.slasrt_(&id_c, &n, raw_data(D.data), &info_val)
	} else when T == f64 {
		lapack.dlasrt_(&id_c, &n, raw_data(D.data), &info_val)
	}

	return info_val == 0, info_val
}

// ===================================================================================
// SUM OF SQUARES COMPUTATION
// ===================================================================================

// Sum of squares computation (f32/c64)
// Updates scale and sumsq such that scale^2 * sumsq = sum of squares of X
v_sum_of_squares_f32_c64 :: proc(
	X: ^Vector($T),
	scale: ^f32, // In/out: scaling factor
	sumsq: ^f32, // In/out: sum of squares
) where T == f32 || T == complex64 {
	if len(X.data) == 0 {return}

	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	when T == f32 {
		lapack.slassq_(&n, &X.data[X.offset], &incx, scale, sumsq)
	} else when T == complex64 {
		lapack.classq_(&n, cast(^complex64)&X.data[X.offset], &incx, scale, sumsq)
	}
}

// Sum of squares computation (f64/c128)
// Updates scale and sumsq such that scale^2 * sumsq = sum of squares of X
v_sum_of_squares_f64_c128 :: proc(
	X: ^Vector($T),
	scale: ^f64, // In/out: scaling factor
	sumsq: ^f64, // In/out: sum of squares
) where T == f64 || T == complex128 {
	// Validate input
	if len(X.data) == 0 {return}

	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	when T == f64 {
		lapack.dlassq_(&n, &X.data[X.offset], &incx, scale, sumsq)
	} else when T == complex128 {
		lapack.zlassq_(&n, cast(^complex128)&X.data[X.offset], &incx, scale, sumsq)
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Initialize matrix to identity
m_identity :: proc(A: ^Matrix($T)) -> bool where is_float(T) || is_complex(T) {
	// Only works for square matrices
	assert(A.rows == A.cols, "Identity matrix must be square")

	m_initialize_matrix(A, 0, 1, .Full)
	return true
}

// Compute vector norm using sum of squares
v_norm :: proc(X: ^Vector($T)) -> f64 where is_float(T) || is_complex(T) {
	when T == f32 || T == complex64 {
		scale := f32(1.0)
		sumsq := f32(0.0)
		v_sum_of_squares_f32_c64(X, &scale, &sumsq)
		return f64(scale) * math.sqrt(f64(sumsq))
	} else when T == f64 || T == complex128 {
		scale := f64(1.0)
		sumsq := f64(0.0)
		v_sum_of_squares_f64_c128(X, &scale, &sumsq)
		return scale * math.sqrt(sumsq)
	}
}

// Copy matrix data
copy_matrix :: proc(src: ^Matrix($T), dst: ^Matrix(T)) {
	min_rows := min(src.rows, dst.rows)
	min_cols := min(src.cols, dst.cols)
	for j in 0 ..< min_cols {
		for i in 0 ..< min_rows {
			matrix_set(dst, i, j, matrix_get(src, i, j))
		}
	}
}

// Find maximum value in array
max_array :: proc(arr: []$T) -> f64 {
	if len(arr) == 0 {
		return 0.0
	}
	max_val := f64(arr[0])
	for val in arr[1:] {
		max_val = max(max_val, f64(val))
	}
	return max_val
}

// ===================================================================================
// MATRIX ROW/COLUMN PERMUTATION (LAPMR/LAPMT)
// ===================================================================================

// Permute rows of matrix
permute_rows :: proc {
	permute_rows_real,
	permute_rows_complex,
}

// Permute columns of matrix
permute_columns :: proc {
	permute_columns_real,
	permute_columns_complex,
}

// Permute rows of matrix (real: f32/f64)
permute_rows_real :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_float(T) {
	// Validate input
	assert(len(K) >= int(A.rows), "Permutation array too small")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	ldx := Blas_Int(A.ld)
	forwrd: Blas_Int = 1 if forward else 0

	when T == f32 {
		lapack.slapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	}
}

// Permute rows of matrix (complex: c64/c128)
permute_rows_complex :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_complex(T) {
	// Validate input
	assert(len(K) >= int(A.rows), "Permutation array too small")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	ldx := Blas_Int(A.ld)
	forwrd: Blas_Int = 1 if forward else 0

	when T == complex64 {
		lapack.clapmr_(&forwrd, &m, &n, cast(^complex64)raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmr_(&forwrd, &m, &n, cast(^complex128)raw_data(A.data), &ldx, raw_data(K))
	}
}

// Permute columns of matrix (real: f32/f64)
permute_columns_real :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_float(T) {
	// Validate input
	assert(len(K) >= int(A.cols), "Permutation array too small")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	ldx := Blas_Int(A.ld)
	forwrd: Blas_Int = 1 if forward else 0

	when T == f32 {
		lapack.slapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	}
}

// Permute columns of matrix (complex: c64/c128)
permute_columns_complex :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_complex(T) {
	// Validate input
	assert(len(K) >= int(A.cols), "Permutation array too small")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	ldx := Blas_Int(A.ld)
	forwrd: Blas_Int = 1 if forward else 0

	when T == complex64 {
		lapack.clapmt_(&forwrd, &m, &n, cast(^complex64)raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmt_(&forwrd, &m, &n, cast(^complex128)raw_data(A.data), &ldx, raw_data(K))
	}
}
