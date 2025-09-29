package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// TRIDIAGONAL POSITIVE DEFINITE FACTORIZATION (Cholesky for tridiagonal)
// ============================================================================
// These routines compute the factorization of a symmetric positive definite
// tridiagonal matrix A = L * D * L^T or A = U * D * U^H for complex
m_factorize_tridiagonal_pd :: proc {
	m_factorize_tridiagonal_pd_f32_c64,
	m_factorize_tridiagonal_pd_f64_c128,
}
m_solve_factorized_tridiagonal_pd :: proc {
	m_solve_factorized_tridiagonal_pd_f32_c64,
	m_solve_factorized_tridiagonal_pd_f64_c128,
}

// Factorize tridiagonal positive definite matrix (f32/c64)
m_factorize_tridiagonal_pd_f32_c64 :: proc(
	D: []f32, // Diagonal (modified to factored diagonal)
	E: []$T, // Off-diagonal (modified to factored off-diagonal)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	when T == f32 {
		lapack.spttrf_(&n_int, raw_data(D), raw_data(E), &info)
	} else when T == complex64 {
		lapack.cpttrf_(&n_int, raw_data(D), raw_data(E), &info)
	}

	return info, info == 0
}

// Factorize tridiagonal positive definite matrix (f64/c128)
m_factorize_tridiagonal_pd_f64_c128 :: proc(
	D: []f64, // Diagonal (modified to factored diagonal)
	E: []$T, // Off-diagonal (modified to factored off-diagonal)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := len(D)
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)

	when T == f64 {
		lapack.dpttrf_(&n_int, raw_data(D), raw_data(E), &info)
	} else when T == complex128 {
		lapack.zpttrf_(&n_int, raw_data(D), raw_data(E), &info)
	}

	return info, info == 0
}


// Solve using factorized tridiagonal positive definite matrix (f32/c64)
m_solve_factorized_tridiagonal_pd_f32_c64 :: proc(
	D: []f32, // Factored diagonal from factorization
	E: []$T, // Factored off-diagonal from factorization
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	uplo := MatrixRegion.Upper, // Upper or lower (complex only)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(B.rows == n, "RHS dimension mismatch")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == f32 {
		// Real case doesn't use uplo parameter
		lapack.spttrs_(&n_int, &nrhs_int, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		// Complex case uses uplo parameter
		uplo_c := cast(u8)uplo
		lapack.cpttrs_(&uplo_c, &n_int, &nrhs_int, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Solve using factorized tridiagonal positive definite matrix (f64/c128)
m_solve_factorized_tridiagonal_pd_f64_c128 :: proc(
	D: []f64, // Factored diagonal from factorization
	E: []$T, // Factored off-diagonal from factorization
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	uplo := MatrixRegion.Upper, // Upper or lower (complex only)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := len(D)
	nrhs := B.cols
	assert(len(E) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(B.rows == n, "RHS dimension mismatch")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := B.ld

	when T == f64 {
		// Real case doesn't use uplo parameter
		lapack.dpttrs_(&n_int, &nrhs_int, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		// Complex case uses uplo parameter
		uplo_c := cast(u8)uplo
		lapack.zpttrs_(&uplo_c, &n_int, &nrhs_int, raw_data(D), raw_data(E), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}
