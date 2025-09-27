package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC MATRIX UTILITIES
// ============================================================================
// Utility functions for symmetric matrix operations

// ============================================================================
// SYMMETRIC MATRIX ROW/COLUMN SWAPPING
// ============================================================================
// Functions to swap rows and columns while maintaining symmetry

// Swap rows and corresponding columns in symmetric matrix
m_swap_symmetric_rows :: proc(
	a: ^Matrix($T),
	i1: int, // First index (0-based)
	i2: int, // Second index (0-based)
	uplo := MatrixRegion.Upper,
) -> (
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(i1 >= 0 && i1 < n, "First index out of range")
	assert(i2 >= 0 && i2 < n, "Second index out of range")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	i1_int := Blas_Int(i1 + 1)
	i2_int := Blas_Int(i2 + 1)

	when T == f32 {
		lapack.ssyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == f64 {
		lapack.dsyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == complex64 {
		lapack.csyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == complex128 {
		lapack.zsyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	}

	return true
}

// ============================================================================
// SYMMETRIC TRIDIAGONALIZATION
// ============================================================================
// Reduces symmetric matrices to tridiagonal form

// Transformation type for 2-stage algorithms
TransformationType :: enum {
	NO_VECTORS, // 'N' - Do not compute transformation matrix
	VECTORS, // 'V' - Compute transformation matrix
}

// Query workspace for symmetric tridiagonalization
query_workspace_tridiagonalize_symmetric :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	}

	return work_size
}

// Reduce symmetric matrix to tridiagonal form
m_tridiagonalize_symmetric :: proc(
	a: ^Matrix($T), // Symmetric matrix (modified in place)
	d: []T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	tau: []T, // Householder reflector scalars (size n-1)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_(&uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrd_(&uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// Query workspace for 2-stage symmetric tridiagonalization
query_workspace_tridiagonalize_symmetric_2stage :: proc($T: typeid, n: int, vect := TransformationType.NO_VECTORS, uplo := MatrixRegion.Upper) -> (work_size: int, hous2_size: int) where T == f32 || T == f64 {
	n_int := Blas_Int(n)
	vect_c := transformation_type_to_char(vect)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	lhous2 := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		hous2_query: f32
		lapack.ssytrd_2stage_(&vect_c, &uplo_c, &n_int, nil, &lda, nil, nil, nil, &hous2_query, &lhous2, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
		hous2_size = int(hous2_query)
	} else when T == f64 {
		work_query: f64
		hous2_query: f64
		lapack.dsytrd_2stage_(&vect_c, &uplo_c, &n_int, nil, &lda, nil, nil, nil, &hous2_query, &lhous2, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
		hous2_size = int(hous2_query)
	}

	return work_size, hous2_size
}

// Reduce symmetric matrix to tridiagonal form using 2-stage algorithm
m_tridiagonalize_symmetric_2stage :: proc(
	a: ^Matrix($T), // Symmetric matrix (modified in place)
	d: []T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	tau: []T, // Householder reflector scalars (size n-1)
	hous2: []T, // Additional Householder vectors
	work: []T, // Workspace
	vect := TransformationType.NO_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(hous2) > 0, "HOUS2 array required")
	assert(len(work) > 0, "Workspace required")

	vect_c := transformation_type_to_char(vect)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lhous2 := Blas_Int(len(hous2))
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_2stage_(&vect_c, &uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(hous2), &lhous2, raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsytrd_2stage_(&vect_c, &uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(hous2), &lhous2, raw_data(work), &lwork, &info, 1, 1)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC MATRIX FACTORIZATION (BUNCH-KAUFMAN)
// ============================================================================
// Bunch-Kaufman factorization of symmetric indefinite matrices

// Query workspace for symmetric factorization (Bunch-Kaufman)
query_workspace_factorize_symmetric :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using Bunch-Kaufman pivoting
m_factorize_symmetric :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Aasen's algorithm for symmetric indefinite matrices

// Query workspace for Aasen symmetric factorization
query_workspace_factorize_symmetric_aasen :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using Aasen's algorithm
m_factorize_symmetric_aasen :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// 2-STAGE AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Two-stage Aasen algorithm for improved performance on large matrices

// Query workspace for 2-stage Aasen symmetric factorization
query_workspace_factorize_symmetric_aasen_2stage :: proc(
	$T: typeid,
	n: int,
	nb: int, // Block size for band matrix
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ltb := Blas_Int(max(1, 4 * n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using 2-stage Aasen's algorithm
m_factorize_symmetric_aasen_2stage :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	tb: ^Matrix(T), // Band matrix storage (4*n x nb)
	ipiv: []Blas_Int, // First stage pivot indices (size n)
	ipiv2: []Blas_Int, // Second stage pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(tb.rows >= 4 * n, "Band matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ltb := Blas_Int(tb.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// RK (BOUNDED BUNCH-KAUFMAN) FACTORIZATION
// ============================================================================
// Bounded Bunch-Kaufman factorization with additional E factor

// Query workspace for RK (bounded Bunch-Kaufman) symmetric factorization
query_workspace_factorize_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using RK (bounded Bunch-Kaufman) algorithm
m_factorize_symmetric_rk :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	e: []T, // E factor from RK factorization (size n)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// ROOK PIVOTING FACTORIZATION
// ============================================================================
// Rook pivoting factorization for enhanced numerical stability

// Query workspace for symmetric matrix factorization using rook pivoting
query_workspace_factorize_symmetric_rook :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using rook pivoting algorithm
m_factorize_symmetric_rook :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// SYMMETRIC MATRIX INVERSION
// ============================================================================
// Inversion of symmetric indefinite matrices using factorizations

// ============================================================================
// STANDARD SYMMETRIC MATRIX INVERSION
// ============================================================================
// Standard inversion using Bunch-Kaufman factorization

// Query workspace for symmetric matrix inversion
query_workspace_invert_symmetric :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRI requires workspace of size n
	work_size = n
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Symmetric matrix inversion using factorization
m_invert_symmetric :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.ssytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == complex64 {
		lapack.csytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// IMPROVED SYMMETRIC MATRIX INVERSION (SYTRI2)
// ============================================================================
// Improved inversion algorithm with better cache efficiency

// Query workspace for improved symmetric matrix inversion
query_workspace_invert_symmetric_improved :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Improved symmetric matrix inversion using factorization
m_invert_symmetric_improved :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// BLOCK-BASED SYMMETRIC MATRIX INVERSION (SYTRI2X)
// ============================================================================
// Block-based inversion algorithm for improved performance

// Query workspace for block-based symmetric matrix inversion
query_workspace_invert_symmetric_blocked :: proc($T: typeid, n: int, nb: int = 64) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRI2X requires workspace of size nb*(n+nb)
	effective_nb := nb
	if effective_nb <= 0 {
		effective_nb = 64 // Default block size
	}
	work_size = effective_nb * (n + effective_nb)
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Block-based symmetric matrix inversion using factorization
m_invert_symmetric_blocked :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace (size nb*(n+nb))
	uplo := MatrixRegion.Upper,
	nb: int = 64, // Block size
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= nb * (n + nb), "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	nb_int := Blas_Int(nb)

	when T == f32 {
		lapack.ssytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == f64 {
		lapack.dsytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == complex64 {
		lapack.csytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == complex128 {
		lapack.zsytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// RK INVERSION WITH E FACTOR (SYTRI_3)
// ============================================================================
// Inversion using RK factorization with E factor

// Query workspace for RK inversion with E factor
query_workspace_invert_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// RK inversion with E factor using factorization
m_invert_symmetric_rk :: proc(
	a: ^Matrix($T), // RK factorized matrix (inverted in place)
	e: []T, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// SYMMETRIC SYSTEM SOLUTION
// ============================================================================
// Solving symmetric systems using factorizations

// System solution result structure
SolutionResult :: struct($T: typeid) {
	solution_successful: bool,
	is_singular:         bool,
}

// ============================================================================
// STANDARD SYMMETRIC SYSTEM SOLUTION (SYTRS)
// ============================================================================
// Standard solution using Bunch-Kaufman factorization

// Symmetric system solution using factorization
m_solve_symmetric :: proc(
	a: ^Matrix($T), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// IMPROVED SYMMETRIC SYSTEM SOLUTION (SYTRS2)
// ============================================================================
// Improved solution algorithm with workspace usage

// Query workspace for improved symmetric system solution
query_workspace_solve_symmetric_improved :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRS2 requires workspace of size n
	work_size = n
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Improved symmetric system solution using factorization
m_solve_symmetric_improved :: proc(
	a: ^Matrix($T), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // RHS on input, solution on output
	work: []T, // Workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == complex64 {
		lapack.csytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// RK SYSTEM SOLUTION WITH E FACTOR (SYTRS_3)
// ============================================================================
// System solution using RK factorization with E factor

// RK system solution with E factor using factorization
m_solve_symmetric_rk :: proc(
	a: ^Matrix($T), // RK factorized matrix
	e: []T, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// AASEN SYSTEM SOLUTION (SYTRS_AA)
// ============================================================================
// System solution using Aasen factorization

// Query workspace for Aasen system solution
query_workspace_solve_symmetric_aasen :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(n)
	ldb := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Aasen system solution using factorization
m_solve_symmetric_aasen :: proc(
	a: ^Matrix($T), // Aasen factorized matrix
	ipiv: []Blas_Int, // Pivot indices from Aasen factorization
	b: ^Matrix(T), // RHS on input, solution on output
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// 2-STAGE AASEN SYSTEM SOLUTION (SYTRS_AA_2STAGE)
// ============================================================================
// System solution using 2-stage Aasen factorization

// 2-stage Aasen system solution using factorization
m_solve_symmetric_aasen_2stage :: proc(
	a: ^Matrix($T), // 2-stage Aasen factorized matrix
	tb: ^Matrix(T), // Band matrix from 2-stage factorization
	ipiv: []Blas_Int, // First stage pivot indices
	ipiv2: []Blas_Int, // Second stage pivot indices
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ltb := Blas_Int(tb.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// ROOK PIVOTING SYSTEM SOLUTION (SYTRS_ROOK)
// ============================================================================
// System solution using rook pivoting factorization

// Rook pivoting system solution using factorization
m_solve_symmetric_rook :: proc(
	a: ^Matrix($T), // Rook factorized matrix
	ipiv: []Blas_Int, // Pivot indices from rook factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// TRIANGULAR BAND MATRIX CONDITION NUMBER
// ============================================================================
// Condition number estimation for triangular band matrices

// Diagonal type for triangular matrices
DiagonalType :: enum {
	NON_UNIT, // 'N' - non-unit diagonal
	UNIT, // 'U' - unit diagonal
}

// Convert diagonal type to character
diagonal_type_to_char :: proc(diag: DiagonalType) -> u8 {
	switch diag {
	case .NON_UNIT:
		return 'N'
	case .UNIT:
		return 'U'
	}
	return 'N'
}

// Helper function to convert TransposeMode to char
transpose_mode_to_char :: proc(t: TransposeMode) -> u8 {
	switch t {
	case .None:
		return 'N'
	case .Transpose:
		return 'T'
	case .ConjugateTranspose:
		return 'C'
	}
	return 'N'
}

// Helper function to convert RFPTranspose to char
rfp_transpose_to_char :: proc(trans: RFPTranspose) -> u8 {
	switch trans {
	case .NORMAL:
		return 'N'
	case .TRANSPOSE:
		return 'T'
	case .CONJUGATE:
		return 'C'
	}
	return 'N'
}

// Query workspace for triangular band condition number - real types
query_workspace_condition_triangular_band_real :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Real types need: work = 3*n, iwork = n
	work_size = 3 * n
	iwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	return
}

// Query workspace for triangular band condition number - complex types
query_workspace_condition_triangular_band_complex :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == complex64 || T == complex128 {
	// Complex types need: work = 2*n, rwork = n
	work_size = 2 * n
	rwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	return
}

// Triangular band condition number for real types
m_condition_triangular_band_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix in packed storage
	work: []T, // Workspace (size 3*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	norm := NormType.ONE_NORM,
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	norm_c: c.char = norm == .ONE_NORM ? '1' : 'I'
	uplo_c := matrix_region_to_char(uplo)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)

	when T == f32 {
		lapack.stbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	}

	return rcond, info, info == 0
}

// Triangular band condition number for complex types
m_condition_triangular_band_c64_c128 :: proc(
	ab: ^Matrix($T), // Band matrix in packed storage
	work: []T, // Complex workspace (size 2*n)
	rwork: []$R, // Real workspace (size n)
	norm := NormType.ONE_NORM,
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c: c.char = norm == .ONE_NORM ? '1' : 'I'
	uplo_c := matrix_region_to_char(uplo)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)

	when T == complex64 {
		lapack.ctbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	}

	return rcond, info, info == 0
}

// Procedure group for triangular band condition number
m_condition_triangular_band :: proc {
	m_condition_triangular_band_f32_f64,
	m_condition_triangular_band_c64_c128,
}

// ============================================================================
// TRIANGULAR BAND ERROR BOUNDS AND REFINEMENT
// ============================================================================
// Error bounds and iterative refinement for triangular band systems

// Query workspace for triangular band error bounds - real types
query_workspace_refine_triangular_band_real :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Real types need: work = 3*n, iwork = n
	work_size = 3 * n
	iwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	return
}

// Query workspace for triangular band error bounds - complex types
query_workspace_refine_triangular_band_complex :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == complex64 || T == complex128 {
	// Complex types need: work = 2*n, rwork = n
	work_size = 2 * n
	rwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	return
}

// Query result sizes for triangular band error bounds
query_result_sizes_refine_triangular_band :: proc(nrhs: int) -> (ferr_size: int, berr_size: int) {
	return nrhs, nrhs
}

// Triangular band error bounds for real types
m_refine_triangular_band_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix
	b: ^Matrix(T), // Original RHS
	x: ^Matrix(T), // Current solution
	ferr: []T, // Forward error bounds (size nrhs)
	berr: []T, // Backward error bounds (size nrhs)
	work: []T, // Workspace (size 3*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == f32 {
		lapack.stbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	}

	return info, info == 0
}

// Triangular band error bounds for complex types
m_refine_triangular_band_c64_c128 :: proc(
	ab: ^Matrix($T), // Band matrix
	b: ^Matrix(T), // Original RHS
	x: ^Matrix(T), // Current solution
	ferr: []$R, // Forward error bounds (size nrhs)
	berr: []R, // Backward error bounds (size nrhs)
	work: []T, // Complex workspace (size 2*n)
	rwork: []R, // Real workspace (size n)
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == complex64 {
		lapack.ctbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	}

	return info, info == 0
}

// Procedure group for triangular band error bounds
m_refine_triangular_band :: proc {
	m_refine_triangular_band_f32_f64,
	m_refine_triangular_band_c64_c128,
}


// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Swap rows and columns in symmetric matrix
swap_symmetric_rows_columns :: proc(a: ^Matrix($T), i1: int, i2: int, uplo := MatrixRegion.Upper) -> bool {
	return m_swap_symmetric_rows(a, i1, i2, uplo)
}


// ==============================================================================
// Triangular Band System Solution Functions
// ==============================================================================

// Triangular band system solution result structure
TriangularBandSolutionResult :: struct($T: typeid) {
	solution_successful: bool,
	solution_matrix:     Matrix(T), // Solution matrix X
}

// ============================================================================
// TRIANGULAR BAND SYSTEM SOLUTION
// ============================================================================
// Solves triangular band systems of equations

// Solve triangular band system
// Solves A * X = B or A^T * X = B where A is triangular band matrix
m_solve_triangular_band :: proc(
	ab: ^Matrix($T), // Triangular band matrix
	b: ^Matrix(T), // Right-hand side matrix (solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(b.rows >= n, "B matrix too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c := diagonal_type_to_char(diag)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.stbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	}

	return info, info == 0
}


// ==============================================================================
// Triangular Solve with RFP Format Functions
// ==============================================================================

// ============================================================================
// RFP (RECTANGULAR FULL PACKED) TRIANGULAR SOLVE
// ============================================================================
// Solves triangular systems with matrices stored in RFP format
// RFP format stores triangular/symmetric matrices in a compact rectangular array

// Solve triangular system with RFP format matrix
// Solves: op(A) * X = alpha * B  (side = Left)
//     or: X * op(A) = alpha * B  (side = Right)
// where A is stored in RFP format
m_solve_triangular_rfp :: proc(
	a_rfp: []$T, // Triangular matrix in RFP format
	b: ^Matrix(T), // Right-hand side matrix (solution on output)
	alpha: T = 1, // Scalar multiplier
	side := OrthogonalSide.Left,
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	transr := RFPTranspose.NORMAL,
) -> (
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	m := b.rows
	n := b.cols

	// Determine the size of A based on side
	a_size := side == .Left ? m : n
	assert(len(a_rfp) >= a_size * (a_size + 1) / 2, "RFP array too small")
	assert(b.rows >= m && b.cols >= n, "B matrix too small")

	side_c := side == .Left ? u8('L') : u8('R')
	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c := diagonal_type_to_char(diag)
	transr_c := rfp_transpose_to_char(transr)

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldb := Blas_Int(b.ld)
	alpha_copy := alpha

	when T == f32 {
		lapack.stfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == f64 {
		lapack.dtfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	}

	return true
}


// ============================================================================
// RFP (RECTANGULAR FULL PACKED) TRIANGULAR INVERSION
// ============================================================================
// Inverts triangular matrices stored in RFP format

// Invert triangular matrix in RFP format
// Computes the inverse of a triangular matrix A stored in RFP format
m_invert_triangular_rfp :: proc(
	a_rfp: []$T, // Triangular matrix in RFP format (inverted in place)
	n: int, // Order of the matrix
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	diag_c := diagonal_type_to_char(diag)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.stftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	}

	return info, info == 0
}


// ==============================================================================
// Summary and Convenience Overloads
// ==============================================================================

// Triangular band operation overloads
tbtrs :: proc {
	ctbtrs,
	dtbtrs,
	stbtrs,
	ztbtrs,
}

// RFP triangular operation overloads
tfsm :: proc {
	ctfsm,
	dtfsm,
	stfsm,
	ztfsm,
}
tftri :: proc {
	ctftri,
	dtftri,
	stftri,
	ztftri,
}

// ==============================================================================
// RFP Format Conversion Functions
// ==============================================================================

// RFP to packed format conversion result structure
RFPToPackedResult :: struct($T: typeid) {
	conversion_successful: bool,
	packed_matrix:         []T, // Matrix in packed format (AP)
}

// RFP to full format conversion result structure
RFPToFullResult :: struct($T: typeid) {
	conversion_successful: bool,
	full_matrix:           Matrix(T), // Matrix in standard full format
}

// Low-level RFP to packed format conversion functions (ctfttp, dtfttp, stfttp, ztfttp)
ctfttp :: proc(transr: cstring, uplo: cstring, n: ^Blas_Int, ARF: ^complex64, AP: ^complex64, info: ^Blas_Int) {
	ctfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

dtfttp :: proc(transr: cstring, uplo: cstring, n: ^Blas_Int, ARF: ^f64, AP: ^f64, info: ^Blas_Int) {
	dtfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

stfttp :: proc(transr: cstring, uplo: cstring, n: ^Blas_Int, ARF: ^f32, AP: ^f32, info: ^Blas_Int) {
	stfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

ztfttp :: proc(transr: cstring, uplo: cstring, n: ^Blas_Int, ARF: ^complex128, AP: ^complex128, info: ^Blas_Int) {
	ztfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

// High-level RFP to packed format conversion wrapper functions
convert_rfp_to_packed_complex64 :: proc(ARF: []complex64, n: int, uplo: MatrixTriangle = .Upper, transr: bool = false, allocator := context.allocator) -> (result: RFPToPackedResult(complex64), err: LapackError) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]complex64, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ctfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_float64 :: proc(ARF: []f64, n: int, uplo: MatrixTriangle = .Upper, transr: bool = false, allocator := context.allocator) -> (result: RFPToPackedResult(f64), err: LapackError) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]f64, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	dtfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_float32 :: proc(ARF: []f32, n: int, uplo: MatrixTriangle = .Upper, transr: bool = false, allocator := context.allocator) -> (result: RFPToPackedResult(f32), err: LapackError) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]f32, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	stfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_complex128 :: proc(ARF: []complex128, n: int, uplo: MatrixTriangle = .Upper, transr: bool = false, allocator := context.allocator) -> (result: RFPToPackedResult(complex128), err: LapackError) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]complex128, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ztfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

// Generic RFP to packed format conversion function
convert_rfp_to_packed :: proc {
	convert_rfp_to_packed_complex64,
	convert_rfp_to_packed_float64,
	convert_rfp_to_packed_float32,
	convert_rfp_to_packed_complex128,
}

// ============================================================================
// RFP (RECTANGULAR FULL PACKED) FORMAT CONVERSION
// ============================================================================
// Converts between RFP and standard full matrix formats

// Convert RFP format to full matrix format
// Converts a triangular/symmetric matrix from RFP format to standard full storage
m_convert_rfp_to_full :: proc(
	a_rfp: []$T, // Matrix in RFP format
	a: ^Matrix(T), // Output full matrix
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Output matrix too small")

	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.stfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == f64 {
		lapack.dtfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == complex64 {
		lapack.ctfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == complex128 {
		lapack.ztfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	}

	return info, info == 0
}

// Convert full matrix format to RFP format
// Converts a triangular/symmetric matrix from standard full storage to RFP format
m_convert_full_to_rfp :: proc(
	a: ^Matrix($T), // Full matrix
	a_rfp: []T, // Output matrix in RFP format
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Input matrix too small")

	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.strttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == f64 {
		lapack.dtrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == complex64 {
		lapack.ctrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == complex128 {
		lapack.ztrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	}

	return info, info == 0
}


// ==============================================================================
// Generalized Eigenvector Functions
// ==============================================================================

// Eigenvector computation side specification
EigenvectorSide :: enum {
	Right, // Compute right eigenvectors only
	Left, // Compute left eigenvectors only
	Both, // Compute both left and right eigenvectors
}

// Eigenvector selection specification
EigenvectorSelection :: enum {
	All, // Compute all eigenvectors
	Backtransform, // Backtransform using DTGEVC
	Selected, // Compute selected eigenvectors
}

// Generalized eigenvector result structure
GeneralizedEigenvectorResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	left_eigenvectors:      Matrix(T), // Left eigenvectors VL
	right_eigenvectors:     Matrix(T), // Right eigenvectors VR
	num_computed:           int, // Number of eigenvectors computed
	selection_mask:         []bool, // Selection mask for computed eigenvectors
}

// Helper function to convert eigenvector side to string
eigenvector_side_to_cstring :: proc(side: EigenvectorSide) -> cstring {
	switch side {
	case .Right:
		return "R"
	case .Left:
		return "L"
	case .Both:
		return "B"
	}
	return "R"
}

// Helper function to convert eigenvector selection to string
eigenvector_selection_to_cstring :: proc(selection: EigenvectorSelection) -> cstring {
	switch selection {
	case .All:
		return "A"
	case .Backtransform:
		return "B"
	case .Selected:
		return "S"
	}
	return "A"
}

// Low-level generalized eigenvector functions (ctgevc, dtgevc, stgevc, ztgevc)
ctgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^complex64,
	lds: ^Blas_Int,
	P: ^complex64,
	ldp: ^Blas_Int,
	VL: ^complex64,
	ldvl: ^Blas_Int,
	VR: ^complex64,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex64,
	rwork: ^f32,
	info: ^Blas_Int,
) {
	ctgevc_(side, howmny, select, n, S, lds, P, ldp, VL, ldvl, VR, ldvr, mm, m, work, rwork, info, len(side), len(howmny))
}

dtgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^f64,
	lds: ^Blas_Int,
	P: ^f64,
	ldp: ^Blas_Int,
	VL: ^f64,
	ldvl: ^Blas_Int,
	VR: ^f64,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f64,
	info: ^Blas_Int,
) {
	dtgevc_(side, howmny, select, n, S, lds, P, ldp, VL, ldvl, VR, ldvr, mm, m, work, info, len(side), len(howmny))
}

stgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^f32,
	lds: ^Blas_Int,
	P: ^f32,
	ldp: ^Blas_Int,
	VL: ^f32,
	ldvl: ^Blas_Int,
	VR: ^f32,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f32,
	info: ^Blas_Int,
) {
	stgevc_(side, howmny, select, n, S, lds, P, ldp, VL, ldvl, VR, ldvr, mm, m, work, info, len(side), len(howmny))
}

ztgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^complex128,
	lds: ^Blas_Int,
	P: ^complex128,
	ldp: ^Blas_Int,
	VL: ^complex128,
	ldvl: ^Blas_Int,
	VR: ^complex128,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex128,
	rwork: ^f64,
	info: ^Blas_Int,
) {
	ztgevc_(side, howmny, select, n, S, lds, P, ldp, VL, ldvl, VR, ldvr, mm, m, work, rwork, info, len(side), len(howmny))
}

// High-level generalized eigenvector wrapper functions
compute_generalized_eigenvectors_complex64 :: proc(
	S: Matrix(complex64),
	P: Matrix(complex64),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(complex64, f32),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []complex64 = nil
	right_data: []complex64 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]complex64, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]complex64, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]complex64, 2 * int(n), allocator) or_return
	rwork := make([]f32, 2 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	ctgevc(side_str, howmny_str, select_ptr, &n, raw_data(S.data), &lds, raw_data(P.data), &ldp, left_ptr, &ldvl, right_ptr, &ldvr, &mm, &m, raw_data(work), raw_data(rwork), &info)

	// Clean up workspace
	delete(work, allocator)
	delete(rwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(complex64)
	right_matrix: Matrix(complex64)

	if left_data != nil {
		left_matrix = Matrix(complex64) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(complex64) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_float64 :: proc(
	S: Matrix(f64),
	P: Matrix(f64),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(f64, f64),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []f64 = nil
	right_data: []f64 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]f64, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]f64, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]f64, 6 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	dtgevc(side_str, howmny_str, select_ptr, &n, raw_data(S.data), &lds, raw_data(P.data), &ldp, left_ptr, &ldvl, right_ptr, &ldvr, &mm, &m, raw_data(work), &info)

	// Clean up workspace
	delete(work, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(f64)
	right_matrix: Matrix(f64)

	if left_data != nil {
		left_matrix = Matrix(f64) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(f64) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_float32 :: proc(
	S: Matrix(f32),
	P: Matrix(f32),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(f32, f32),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []f32 = nil
	right_data: []f32 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]f32, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]f32, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]f32, 6 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	stgevc(side_str, howmny_str, select_ptr, &n, raw_data(S.data), &lds, raw_data(P.data), &ldp, left_ptr, &ldvl, right_ptr, &ldvr, &mm, &m, raw_data(work), &info)

	// Clean up workspace
	delete(work, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(f32)
	right_matrix: Matrix(f32)

	if left_data != nil {
		left_matrix = Matrix(f32) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(f32) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_complex128 :: proc(
	S: Matrix(complex128),
	P: Matrix(complex128),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(complex128, f64),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []complex128 = nil
	right_data: []complex128 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]complex128, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]complex128, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]complex128, 2 * int(n), allocator) or_return
	rwork := make([]f64, 2 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	ztgevc(side_str, howmny_str, select_ptr, &n, raw_data(S.data), &lds, raw_data(P.data), &ldp, left_ptr, &ldvl, right_ptr, &ldvr, &mm, &m, raw_data(work), raw_data(rwork), &info)

	// Clean up workspace
	delete(work, allocator)
	delete(rwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(complex128)
	right_matrix: Matrix(complex128)

	if left_data != nil {
		left_matrix = Matrix(complex128) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(complex128) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

// Generic generalized eigenvector computation function
compute_generalized_eigenvectors :: proc {
	compute_generalized_eigenvectors_complex64,
	compute_generalized_eigenvectors_float64,
	compute_generalized_eigenvectors_float32,
	compute_generalized_eigenvectors_complex128,
}

// ==============================================================================
// Final Convenience Overloads and Summary
// ==============================================================================

// RFP format conversion overloads
tfttp :: proc {
	ctfttp,
	dtfttp,
	stfttp,
	ztfttp,
}
tfttr :: proc {
	ctfttr,
	dtfttr,
	stfttr,
	ztfttr,
}

// Generalized eigenvector overloads
tgevc :: proc {
	ctgevc,
	dtgevc,
	stgevc,
	ztgevc,
}

// ==============================================================================
// Generalized Schur Form Reordering Functions
// ==============================================================================

// Generalized Schur form reordering result structure
GeneralizedSchurReorderResult :: struct($T: typeid) {
	reordering_successful: bool,
	reordered_A:           Matrix(T), // Reordered matrix A
	reordered_B:           Matrix(T), // Reordered matrix B
	updated_Q:             Matrix(T), // Updated orthogonal matrix Q (if requested)
	updated_Z:             Matrix(T), // Updated orthogonal matrix Z (if requested)
	final_ifst:            int, // Final position after reordering
	final_ilst:            int, // Final position after reordering
}

// Low-level generalized Schur form reordering functions (ctgexc, dtgexc, stgexc, ztgexc)
ctgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	Q: ^complex64,
	ldq: ^Blas_Int,
	Z: ^complex64,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, info)
}

dtgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	Q: ^f64,
	ldq: ^Blas_Int,
	Z: ^f64,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	work: ^f64,
	lwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, work, lwork, info)
}

stgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	Q: ^f32,
	ldq: ^Blas_Int,
	Z: ^f32,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	work: ^f32,
	lwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, work, lwork, info)
}

ztgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	Q: ^complex128,
	ldq: ^Blas_Int,
	Z: ^complex128,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, info)
}

// High-level generalized Schur form reordering wrapper functions
reorder_generalized_schur_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	Q: Matrix(complex64) = {},
	Z: Matrix(complex64) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(complex64),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]complex64, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(complex64) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []complex64 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(complex64)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]complex64, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(complex64) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]complex64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(complex64) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []complex64 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(complex64)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]complex64, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(complex64) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]complex64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(complex64) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	ctgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_copy, &ilst_copy, &info)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_float64 :: proc(
	A: Matrix(f64),
	B: Matrix(f64),
	Q: Matrix(f64) = {},
	Z: Matrix(f64) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(f64),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]f64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(f64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]f64, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(f64) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []f64 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(f64)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]f64, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(f64) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]f64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(f64) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []f64 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(f64)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]f64, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(f64) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]f64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(f64) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	// Query optimal workspace size
	work_query: f64
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	ifst_query := Blas_Int(ifst + 1)
	ilst_query := Blas_Int(ilst + 1)

	dtgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, nil, &ldq, nil, &ldz, &ifst_query, &ilst_query, &work_query, &lwork_query, &info_query)

	if info_query != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(work_query)
	work := make([]f64, int(lwork), allocator) or_return

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	dtgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_copy, &ilst_copy, raw_data(work), &lwork, &info)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_float32 :: proc(
	A: Matrix(f32),
	B: Matrix(f32),
	Q: Matrix(f32) = {},
	Z: Matrix(f32) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(f32),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]f32, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(f32) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]f32, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(f32) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []f32 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(f32)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]f32, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(f32) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]f32, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(f32) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []f32 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(f32)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]f32, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(f32) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]f32, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(f32) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	// Query optimal workspace size
	work_query: f32
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	ifst_query := Blas_Int(ifst + 1)
	ilst_query := Blas_Int(ilst + 1)

	stgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, nil, &ldq, nil, &ldz, &ifst_query, &ilst_query, &work_query, &lwork_query, &info_query)

	if info_query != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(work_query)
	work := make([]f32, int(lwork), allocator) or_return

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	stgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_copy, &ilst_copy, raw_data(work), &lwork, &info)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_complex128 :: proc(
	A: Matrix(complex128),
	B: Matrix(complex128),
	Q: Matrix(complex128) = {},
	Z: Matrix(complex128) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(complex128),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]complex128, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(complex128) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]complex128, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(complex128) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []complex128 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(complex128)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]complex128, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(complex128) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]complex128, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(complex128) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []complex128 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(complex128)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]complex128, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(complex128) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]complex128, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(complex128) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	ztgexc(&wantq, &wantz, &n, raw_data(a_data), &lda, raw_data(b_data), &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_copy, &ilst_copy, &info)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

// Generic generalized Schur form reordering function
reorder_generalized_schur :: proc {
	reorder_generalized_schur_complex64,
	reorder_generalized_schur_float64,
	reorder_generalized_schur_float32,
	reorder_generalized_schur_complex128,
}

// ==============================================================================
// Generalized Schur Form Condition Estimation and SVD Functions
// ==============================================================================

// Condition estimation job specification
ConditionJob :: enum {
	EigenvaluesOnly, // Compute eigenvalues only
	Reciprocal, // Compute reciprocal condition numbers
	EstimateBounds, // Estimate error bounds
	Full, // Full condition estimation
}

// SVD job specification
SVDJob :: enum {
	None, // Do not compute matrix
	Compute, // Compute matrix
}

// Generalized Schur condition result
GeneralizedSchurConditionResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	eigenvalues_alpha:      []T, // Alpha values (eigenvalues for complex)
	eigenvalues_beta:       []T, // Beta values
	selected_count:         int, // Number of selected eigenvalues
	reciprocal_left:        S, // Left reciprocal condition number
	reciprocal_right:       S, // Right reciprocal condition number
	error_bounds:           []S, // Error bound estimates
}

// Generalized SVD result
GeneralizedSVDResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	alpha_values:           []S, // Alpha values from GSVD
	beta_values:            []S, // Beta values from GSVD
	U_matrix:               Matrix(T), // Left orthogonal matrix U
	V_matrix:               Matrix(T), // Right orthogonal matrix V
	Q_matrix:               Matrix(T), // Orthogonal matrix Q
	cycles_performed:       int, // Number of Jacobi cycles
}

// Low-level condition estimation functions (ctgsen, dtgsen, stgsen, ztgsen)
ctgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	alpha: ^complex64,
	beta: ^complex64,
	Q: ^complex64,
	ldq: ^Blas_Int,
	Z: ^complex64,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f32,
	pr: ^f32,
	DIF: ^f32,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsen_(ijob, wantq, wantz, select, n, A, lda, B, ldb, alpha, beta, Q, ldq, Z, ldz, m, pl, pr, DIF, work, lwork, iwork, liwork, info)
}

dtgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	alphar: ^f64,
	alphai: ^f64,
	beta: ^f64,
	Q: ^f64,
	ldq: ^Blas_Int,
	Z: ^f64,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f64,
	pr: ^f64,
	DIF: ^f64,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsen_(ijob, wantq, wantz, select, n, A, lda, B, ldb, alphar, alphai, beta, Q, ldq, Z, ldz, m, pl, pr, DIF, work, lwork, iwork, liwork, info)
}

stgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	alphar: ^f32,
	alphai: ^f32,
	beta: ^f32,
	Q: ^f32,
	ldq: ^Blas_Int,
	Z: ^f32,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f32,
	pr: ^f32,
	DIF: ^f32,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsen_(ijob, wantq, wantz, select, n, A, lda, B, ldb, alphar, alphai, beta, Q, ldq, Z, ldz, m, pl, pr, DIF, work, lwork, iwork, liwork, info)
}

ztgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	alpha: ^complex128,
	beta: ^complex128,
	Q: ^complex128,
	ldq: ^Blas_Int,
	Z: ^complex128,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f64,
	pr: ^f64,
	DIF: ^f64,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsen_(ijob, wantq, wantz, select, n, A, lda, B, ldb, alpha, beta, Q, ldq, Z, ldz, m, pl, pr, DIF, work, lwork, iwork, liwork, info)
}

// Low-level generalized SVD functions (ctgsja, dtgsja, stgsja, ztgsja)
ctgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	tola: ^f32,
	tolb: ^f32,
	alpha: ^f32,
	beta: ^f32,
	U: ^complex64,
	ldu: ^Blas_Int,
	V: ^complex64,
	ldv: ^Blas_Int,
	Q: ^complex64,
	ldq: ^Blas_Int,
	work: ^complex64,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsja_(jobu, jobv, jobq, m, p, n, k, l, A, lda, B, ldb, tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq, work, ncycle, info, len(jobu), len(jobv), len(jobq))
}

dtgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	tola: ^f64,
	tolb: ^f64,
	alpha: ^f64,
	beta: ^f64,
	U: ^f64,
	ldu: ^Blas_Int,
	V: ^f64,
	ldv: ^Blas_Int,
	Q: ^f64,
	ldq: ^Blas_Int,
	work: ^f64,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsja_(jobu, jobv, jobq, m, p, n, k, l, A, lda, B, ldb, tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq, work, ncycle, info, len(jobu), len(jobv), len(jobq))
}

stgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	tola: ^f32,
	tolb: ^f32,
	alpha: ^f32,
	beta: ^f32,
	U: ^f32,
	ldu: ^Blas_Int,
	V: ^f32,
	ldv: ^Blas_Int,
	Q: ^f32,
	ldq: ^Blas_Int,
	work: ^f32,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsja_(jobu, jobv, jobq, m, p, n, k, l, A, lda, B, ldb, tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq, work, ncycle, info, len(jobu), len(jobv), len(jobq))
}

ztgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	tola: ^f64,
	tolb: ^f64,
	alpha: ^f64,
	beta: ^f64,
	U: ^complex128,
	ldu: ^Blas_Int,
	V: ^complex128,
	ldv: ^Blas_Int,
	Q: ^complex128,
	ldq: ^Blas_Int,
	work: ^complex128,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsja_(jobu, jobv, jobq, m, p, n, k, l, A, lda, B, ldb, tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq, work, ncycle, info, len(jobu), len(jobv), len(jobq))
}

// ==============================================================================
// Final Convenience Overloads
// ==============================================================================

// Generalized Schur reordering overloads
tgexc :: proc {
	ctgexc,
	dtgexc,
	stgexc,
	ztgexc,
}

// Generalized Schur condition estimation overloads
tgsen :: proc {
	ctgsen,
	dtgsen,
	stgsen,
	ztgsen,
}

// Generalized SVD overloads
tgsja :: proc {
	ctgsja,
	dtgsja,
	stgsja,
	ztgsja,
}

// ==============================================================================
// Generalized Eigenvalue Sensitivity Analysis Functions
// ==============================================================================

// Sensitivity analysis job specification
SensitivityJob :: enum {
	EigenvaluesOnly, // Compute eigenvalue condition numbers only
	SubspacesOnly, // Compute invariant subspace condition numbers only
	Both, // Compute both eigenvalue and subspace condition numbers
}

// Generalized eigenvalue sensitivity result
GeneralizedSensitivityResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	condition_numbers_S:    []S, // Eigenvalue condition numbers
	condition_numbers_DIF:  []S, // Invariant subspace condition numbers
	num_computed:           int, // Number of condition numbers computed
	selection_mask:         []bool, // Selection mask for computed values
}

// Low-level generalized eigenvalue sensitivity functions (ctgsna, dtgsna, stgsna, ztgsna)
ctgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	VL: ^complex64,
	ldvl: ^Blas_Int,
	VR: ^complex64,
	ldvr: ^Blas_Int,
	S: ^f32,
	DIF: ^f32,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsna_(job, howmny, select, n, A, lda, B, ldb, VL, ldvl, VR, ldvr, S, DIF, mm, m, work, lwork, iwork, info, len(job), len(howmny))
}

dtgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	VL: ^f64,
	ldvl: ^Blas_Int,
	VR: ^f64,
	ldvr: ^Blas_Int,
	S: ^f64,
	DIF: ^f64,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsna_(job, howmny, select, n, A, lda, B, ldb, VL, ldvl, VR, ldvr, S, DIF, mm, m, work, lwork, iwork, info, len(job), len(howmny))
}

stgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	VL: ^f32,
	ldvl: ^Blas_Int,
	VR: ^f32,
	ldvr: ^Blas_Int,
	S: ^f32,
	DIF: ^f32,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsna_(job, howmny, select, n, A, lda, B, ldb, VL, ldvl, VR, ldvr, S, DIF, mm, m, work, lwork, iwork, info, len(job), len(howmny))
}

ztgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	VL: ^complex128,
	ldvl: ^Blas_Int,
	VR: ^complex128,
	ldvr: ^Blas_Int,
	S: ^f64,
	DIF: ^f64,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsna_(job, howmny, select, n, A, lda, B, ldb, VL, ldvl, VR, ldvr, S, DIF, mm, m, work, lwork, iwork, info, len(job), len(howmny))
}

// Helper function to convert sensitivity job to string
sensitivity_job_to_cstring :: proc(job: SensitivityJob) -> cstring {
	switch job {
	case .EigenvaluesOnly:
		return "E"
	case .SubspacesOnly:
		return "V"
	case .Both:
		return "B"
	}
	return "B"
}

// High-level generalized eigenvalue sensitivity wrapper function
compute_generalized_sensitivity_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	VL: Matrix(complex64) = {},
	VR: Matrix(complex64) = {},
	job: SensitivityJob = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedSensitivityResult(complex64, f32),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)

	// Handle left eigenvectors
	vl_ptr: ^complex64 = nil
	if VL.data != nil {
		ldvl = Blas_Int(VL.rows)
		vl_ptr = raw_data(VL.data)
	}

	// Handle right eigenvectors
	vr_ptr: ^complex64 = nil
	if VR.data != nil {
		ldvr = Blas_Int(VR.rows)
		vr_ptr = raw_data(VR.data)
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate output arrays
	s_values: []f32 = nil
	dif_values: []f32 = nil

	max_compute := int(n)
	if select_mask != nil {
		max_compute = len(select_mask)
	}

	if job == .EigenvaluesOnly || job == .Both {
		s_values = make([]f32, max_compute, allocator) or_return
	}

	if job == .SubspacesOnly || job == .Both {
		dif_values = make([]f32, max_compute, allocator) or_return
	}

	// Query workspace size
	work_query: complex64
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	mm := Blas_Int(max_compute)
	m: Blas_Int

	job_str := sensitivity_job_to_cstring(job)
	howmny_str := eigenvector_selection_to_cstring(selection)
	select_ptr := raw_data(select_array) if select_array != nil else nil
	s_ptr := raw_data(s_values) if s_values != nil else nil
	dif_ptr := raw_data(dif_values) if dif_values != nil else nil

	ctgsna(job_str, howmny_str, select_ptr, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, vl_ptr, &ldvl, vr_ptr, &ldvr, s_ptr, dif_ptr, &mm, &m, &work_query, &lwork_query, nil, &info_query)

	if info_query != 0 && info_query != -17 {
		if s_values != nil do delete(s_values, allocator)
		if dif_values != nil do delete(dif_values, allocator)
		if select_array != nil do delete(select_array, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, int(lwork), allocator) or_return
	iwork := make([]Blas_Int, int(n + 2), allocator) or_return

	info: Blas_Int
	ctgsna(job_str, howmny_str, select_ptr, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, vl_ptr, &ldvl, vr_ptr, &ldvr, s_ptr, dif_ptr, &mm, &m, raw_data(work), &lwork, raw_data(iwork), &info)

	delete(work, allocator)
	delete(iwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if s_values != nil do delete(s_values, allocator)
		if dif_values != nil do delete(dif_values, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result mask
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.condition_numbers_S = s_values
	result.condition_numbers_DIF = dif_values
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

// ==============================================================================
// Generalized Sylvester Equation Functions
// ==============================================================================

// Generalized Sylvester equation result
GeneralizedSylvesterResult :: struct($T: typeid, $S: typeid) {
	solution_successful: bool,
	solution_C:          Matrix(T), // Solution matrix C
	solution_F:          Matrix(T), // Solution matrix F
	dif_estimate:        S, // DIF estimate (if requested)
	scale_factor:        S, // Scale factor applied to solution
}

// Low-level generalized Sylvester equation functions (ctgsyl, dtgsyl, stgsyl, ztgsyl)
ctgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	C: ^complex64,
	ldc: ^Blas_Int,
	D: ^complex64,
	ldd: ^Blas_Int,
	E: ^complex64,
	lde: ^Blas_Int,
	F: ^complex64,
	ldf: ^Blas_Int,
	dif: ^f32,
	scale: ^f32,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsyl_(trans, ijob, m, n, A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf, dif, scale, work, lwork, iwork, info, len(trans))
}

dtgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	C: ^f64,
	ldc: ^Blas_Int,
	D: ^f64,
	ldd: ^Blas_Int,
	E: ^f64,
	lde: ^Blas_Int,
	F: ^f64,
	ldf: ^Blas_Int,
	dif: ^f64,
	scale: ^f64,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsyl_(trans, ijob, m, n, A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf, dif, scale, work, lwork, iwork, info, len(trans))
}

stgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	C: ^f32,
	ldc: ^Blas_Int,
	D: ^f32,
	ldd: ^Blas_Int,
	E: ^f32,
	lde: ^Blas_Int,
	F: ^f32,
	ldf: ^Blas_Int,
	dif: ^f32,
	scale: ^f32,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsyl_(trans, ijob, m, n, A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf, dif, scale, work, lwork, iwork, info, len(trans))
}

ztgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	C: ^complex128,
	ldc: ^Blas_Int,
	D: ^complex128,
	ldd: ^Blas_Int,
	E: ^complex128,
	lde: ^Blas_Int,
	F: ^complex128,
	ldf: ^Blas_Int,
	dif: ^f64,
	scale: ^f64,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsyl_(trans, ijob, m, n, A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf, dif, scale, work, lwork, iwork, info, len(trans))
}

// ==============================================================================
// Triangular Packed Condition Number Functions
// ==============================================================================

// Triangular packed condition number result
TriangularPackedConditionResult :: struct($T: typeid) {
	computation_successful: bool,
	reciprocal_condition:   T, // Reciprocal condition number
	condition_number:       T, // 1/rcond
}

// Low-level triangular packed condition number functions (ctpcon, dtpcon, stpcon, ztpcon)
ctpcon :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^complex64, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Blas_Int) {
	ctpcon_(norm, uplo, diag, n, AP, rcond, work, rwork, info, len(norm), len(uplo), len(diag))
}

dtpcon :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^f64, rcond: ^f64, work: ^f64, iwork: ^Blas_Int, info: ^Blas_Int) {
	dtpcon_(norm, uplo, diag, n, AP, rcond, work, iwork, info, len(norm), len(uplo), len(diag))
}

stpcon :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^f32, rcond: ^f32, work: ^f32, iwork: ^Blas_Int, info: ^Blas_Int) {
	stpcon_(norm, uplo, diag, n, AP, rcond, work, iwork, info, len(norm), len(uplo), len(diag))
}

ztpcon :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^complex128, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Blas_Int) {
	ztpcon_(norm, uplo, diag, n, AP, rcond, work, rwork, info, len(norm), len(uplo), len(diag))
}

// High-level triangular packed condition number wrapper function
estimate_triangular_packed_condition_complex64 :: proc(
	AP: []complex64,
	n: int,
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedConditionResult(f32),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected packed size
	packed_size := (n * (n + 1)) / 2
	if len(AP) < packed_size {
		return {}, .InvalidParameter
	}

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator) or_return
	rwork := make([]f32, n, allocator) or_return

	rcond: f32
	info: Blas_Int
	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctpcon(norm_str, uplo_str, diag_str, &n_int, raw_data(AP), &rcond, raw_data(work), raw_data(rwork), &info)

	delete(work, allocator)
	delete(rwork, allocator)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.computation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	return
}

// ==============================================================================
// Final Advanced Function Overloads
// ==============================================================================

// Generalized eigenvalue sensitivity overloads
tgsna :: proc {
	ctgsna,
	dtgsna,
	stgsna,
	ztgsna,
}

// Generalized Sylvester equation overloads
tgsyl :: proc {
	ctgsyl,
	dtgsyl,
	stgsyl,
	ztgsyl,
}

// Triangular packed condition number overloads
tpcon :: proc {
	ctpcon,
	dtpcon,
	stpcon,
	ztpcon,
}
