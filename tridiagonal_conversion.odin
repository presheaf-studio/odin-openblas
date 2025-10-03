package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// TRIDIAGONAL MATRIX CONVERSION AND UTILITIES
// ===================================================================================
// Functions to convert various matrix types TO tridiagonal form and utilities
// for working with tridiagonal matrices.

// ===================================================================================
// DENSE TO TRIDIAGONAL CONVERSION (SYTRD/HETRD)
// ===================================================================================

// Query workspace for dense to tridiagonal reduction
query_workspace_dense_to_tridiagonal :: proc($T: typeid, n: int) -> (work_size: int) {
	// Query LAPACK for optimal workspace sizes
	n_blas := Blas_Int(n)
	uplo_c := cast(u8)MatrixRegion.Upper
	lwork := QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrd_(&uplo_c, &n_blas, nil, &n_blas, nil, nil, nil, &work_query, &lwork, &info)
		return int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrd_(&uplo_c, &n_blas, nil, &n_blas, nil, nil, nil, &work_query, &lwork, &info)
		return int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.chetrd_(&uplo_c, &n_blas, nil, &n_blas, nil, nil, nil, &work_query, &lwork, &info)
		return int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zhetrd_(&uplo_c, &n_blas, nil, &n_blas, nil, nil, nil, &work_query, &lwork, &info)
		return int(real(work_query))
	}
}

// Convert dense symmetric/Hermitian matrix to tridiagonal form (SYTRD/HETRD)
dense_to_tridiagonal :: proc {
	dense_to_tridiagonal_f32_f64,
	dense_to_tridiagonal_c64_c128,
}

// Convert dense symmetric matrix to tridiagonal form for f32/f64
dense_to_tridiagonal_f32_f64 :: proc(
	A: ^Matrix($T), // Symmetric matrix (modified on output to orthogonal matrix)
	d: []T, // Diagonal elements output (size n)
	e: []T, // Off-diagonal elements output (size n-1)
	tau: []T, // Householder reflectors output (size n-1)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1 || n <= 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dsytrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Convert dense Hermitian matrix to tridiagonal form for c64/c128
dense_to_tridiagonal_c64_c128 :: proc(
	A: ^Matrix($T), // Hermitian matrix (modified on output to unitary matrix)
	d: []$R, // Diagonal elements output (real, size n)
	e: []R, // Off-diagonal elements output (real, size n-1)
	tau: []T, // Householder reflectors output (size n-1)
	work: []T, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1 || n <= 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.chetrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zhetrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED TO TRIDIAGONAL CONVERSION (SBTRD/HBTRD)
// ===================================================================================

// Query workspace for banded to tridiagonal reduction
query_workspace_banded_to_tridiagonal :: proc($T: typeid, n: int) -> (work_size: int) where is_float(T) || is_complex(T) {
	// SBTRD/HBTRD requires n workspace for real types, none for complex
	when is_float(T) {
		return n
	} else {
		return 0 // Complex types don't need workspace
	}
}

// Convert banded symmetric/Hermitian matrix to tridiagonal form (SBTRD/HBTRD)
banded_to_tridiagonal :: proc {
	banded_to_tridiagonal_f32_f64,
	banded_to_tridiagonal_c64_c128,
}

// Convert banded symmetric matrix to tridiagonal form for f32/f64
banded_to_tridiagonal_f32_f64 :: proc(
	AB: ^BandedMatrix($T), // Banded symmetric matrix (modified)
	d: []T, // Diagonal elements output (size n)
	e: []T, // Off-diagonal elements output (size n-1)
	Q: ^Matrix(T) = nil, // Orthogonal matrix (optional output)
	work: []T = nil, // Pre-allocated workspace (size n)
	uplo := MatrixRegion.Upper,
	vect := VectorOption.NO_VECTORS,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := AB.rows
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	if vect == .FORM_VECTORS {
		assert(Q != nil && Q.rows >= n && Q.cols >= n, "Q matrix required for vector computation")
		assert(work != nil && len(work) >= n, "Workspace required for vector computation")
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	kd_int := Blas_Int(AB.ku) // For symmetric banded, kd = ku = kl
	ldab := AB.ldab

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if Q != nil {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	work_ptr: ^T = nil
	if work != nil {
		work_ptr = raw_data(work)
	}

	when T == f32 {
		lapack.ssbtrd_(&vect_c, &uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, work_ptr, &info)
	} else when T == f64 {
		lapack.dsbtrd_(&vect_c, &uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, work_ptr, &info)
	}

	return info, info == 0
}

// Convert banded Hermitian matrix to tridiagonal form for c64/c128
banded_to_tridiagonal_c64_c128 :: proc(
	AB: ^BandedMatrix($T), // Banded Hermitian matrix (modified)
	d: []$R, // Diagonal elements output (real, size n)
	e: []R, // Off-diagonal elements output (real, size n-1)
	Q: ^Matrix(T) = nil, // Unitary matrix (optional output)
	uplo := MatrixRegion.Upper,
	vect := VectorOption.NO_VECTORS,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && R == f32) || (T == complex128 && R == f64) {
	n := AB.rows
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	if vect == .FORM_VECTORS {
		assert(Q != nil && Q.rows >= n && Q.cols >= n, "Q matrix required for vector computation")
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	kd_int := Blas_Int(AB.ku) // For Hermitian banded, kd = ku = kl
	ldab := AB.ldab

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if Q != nil {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	when T == complex64 {
		lapack.chbtrd_(&vect_c, &uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, nil, &info)
	} else when T == complex128 {
		lapack.zhbtrd_(&vect_c, &uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, nil, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIDIAGONAL MATRIX UTILITIES
// ===================================================================================

// Extract tridiagonal elements from a Tridiagonal into separate arrays
extract_tridiagonal_arrays :: proc(
	tm: ^Tridiagonal($T),
	d: []T, // Diagonal output (size n)
	e: []T, // Off-diagonal output (size n-1)
) {
	n := int(tm.n)
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	// Copy diagonal
	slice.copy(d[:n], tm.d)

	// Copy off-diagonal (use dl for consistency)
	if n > 1 {
		slice.copy(e[:n - 1], tm.dl)
	}
}

// Create Tridiagonal from separate arrays
create_tridiagonal :: proc(
	d: []$T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	allocator := context.allocator,
) -> Tridiagonal(T) {
	n := len(d)
	tm := make_tridiagonal(n, T, allocator)

	slice.copy(tm.d, d)
	if n > 1 {
		slice.copy(tm.dl, e)
		slice.copy(tm.du, e) // For general tridiagonal, du = dl
	}

	return tm
}

// Create symmetric Tridiagonal from diagonal and off-diagonal arrays
create_symmetric_tridiagonal :: proc(
	d: []$T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	allocator := context.allocator,
) -> Tridiagonal(T) {
	n := len(d)
	tm := make_symmetric_tridiagonal(n, T, allocator)

	slice.copy(tm.d, d)
	if n > 1 {
		slice.copy(tm.dl, e)
		// For symmetric real matrices, du shares storage with dl
		// For Hermitian complex matrices, du is conjugate of dl
		when is_complex(T) {
			for i in 0 ..< len(e) {
				tm.du[i] = conj(e[i])
			}
		}
	}

	return tm
}

// Compute matrix norm for tridiagonal matrix
tridiagonal_norm :: proc(tm: ^Tridiagonal($T), norm_type: MatrixNorm) -> f64 where is_float(T) || is_complex(T) {
	n := int(tm.n)
	if n == 0 do return 0

	switch norm_type {
	case .OneNorm, .InfinityNorm:
		// For tridiagonal, 1-norm = infinity-norm
		max_sum: f64 = 0

		// First row/column: |d[0]| + |du[0]| (if n > 1)
		when is_float(T) {
			max_sum = f64(abs(tm.d[0]))
			if n > 1 do max_sum += f64(abs(tm.du[0]))
		} else {
			max_sum = abs(tm.d[0])
			if n > 1 do max_sum += abs(tm.du[0])
		}

		// Middle rows/columns: |dl[i-1]| + |d[i]| + |du[i]|
		for i in 1 ..< n - 1 {
			when is_float(T) {
				sum := f64(abs(tm.dl[i - 1]) + abs(tm.d[i]) + abs(tm.du[i]))
			} else {
				sum := abs(tm.dl[i - 1]) + abs(tm.d[i]) + abs(tm.du[i])
			}
			max_sum = max(max_sum, sum)
		}

		// Last row/column: |dl[n-2]| + |d[n-1]| (if n > 1)
		if n > 1 {
			when is_float(T) {
				sum := f64(abs(tm.dl[n - 2]) + abs(tm.d[n - 1]))
			} else {
				sum := abs(tm.dl[n - 2]) + abs(tm.d[n - 1])
			}
			max_sum = max(max_sum, sum)
		}

		return max_sum

	case .FrobeniusNorm:
		// Sum of squares of all elements
		sum_squares: f64 = 0

		// Diagonal elements
		for val in tm.d {
			when is_float(T) {
				sum_squares += f64(val * val)
			} else {
				sum_squares += real(val * conj(val))
			}
		}

		// Off-diagonal elements (count both upper and lower)
		for val in tm.dl {
			when is_float(T) {
				sum_squares += 2 * f64(val * val) // Count both dl and du
			} else {
				sum_squares += 2 * real(val * conj(val))
			}
		}

		return math.sqrt(sum_squares)

	case .MaxNorm:
		// Maximum absolute value
		max_val: f64 = 0

		for val in tm.d {
			when is_float(T) {
				max_val = max(max_val, f64(abs(val)))
			} else {
				max_val = max(max_val, abs(val))
			}
		}

		for val in tm.dl {
			when is_float(T) {
				max_val = max(max_val, f64(abs(val)))
			} else {
				max_val = max(max_val, abs(val))
			}
		}

		for val in tm.du {
			when is_float(T) {
				max_val = max(max_val, f64(abs(val)))
			} else {
				max_val = max(max_val, abs(val))
			}
		}

		return max_val
	}

	return 0
}
