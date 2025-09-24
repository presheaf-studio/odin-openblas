package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED ITERATIVE REFINEMENT AND SPLIT CHOLESKY
// ===================================================================================

// Iterative refinement for positive definite banded systems proc group
m_refine_solution_banded_pd :: proc {
	m_refine_solution_banded_pd_f32_c64,
	m_refine_solution_banded_pd_f64_c128,
}

// Split Cholesky factorization for positive definite banded matrices proc group
m_split_cholesky_banded :: proc {
	m_split_cholesky_banded_f32_c64,
	m_split_cholesky_banded_f64_c128,
}

// ===================================================================================
// ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Refinement result structure
RefinementResult :: struct($T: typeid) {
	ferr: []T, // Forward error bounds for each solution
	berr: []T, // Backward error bounds for each solution
	info: Info,
}

// Iterative refinement for positive definite banded systems (f32/complex64)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_f32_c64 :: proc(
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized matrix from PBTRF
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	result: RefinementResult(f32),
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(
		len(AB.data) > 0 && len(AFB.data) > 0 && len(B.data) > 0 && len(X.data) > 0,
		"Matrices cannot be empty",
	)
	assert(AB.rows == AB.cols && AFB.rows == AFB.cols, "AB and AFB must be square")
	assert(B.rows == X.rows && B.cols == X.cols, "B and X must have same dimensions")
	assert(B.rows == AB.rows, "System dimensions must be consistent")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd_val: Blas_Int = kd
	nrhs: Blas_Int = B.cols
	ldab: Blas_Int = AB.ld
	ldafb: Blas_Int = AFB.ld
	ldb: Blas_Int = B.ld
	ldx: Blas_Int = X.ld

	// Allocate error bound arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

	info_val: Info

	when T == f32 {
		// Allocate workspace
		work := make([]f32, 3 * n)
		defer delete(work)
		iwork := make([]Blas_Int, n)
		defer delete(iwork)

		lapack.spbrfs_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info_val,
			len(uplo_c),
		)
	} else when T == complex64 {
		// Allocate workspace
		work := make([]complex64, 2 * n)
		defer delete(work)
		rwork := make([]f32, n)
		defer delete(rwork)

		lapack.cpbrfs_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(uplo_c),
		)
	}

	return RefinementResult(f32){ferr = ferr, berr = berr, info = info_val}, info_val == 0
}

// Iterative refinement for positive definite banded systems (f64/complex128)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_f64_c128 :: proc(
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized matrix from PBTRF
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	result: RefinementResult(f64),
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(
		len(AB.data) > 0 && len(AFB.data) > 0 && len(B.data) > 0 && len(X.data) > 0,
		"Matrices cannot be empty",
	)
	assert(AB.rows == AB.cols && AFB.rows == AFB.cols, "AB and AFB must be square")
	assert(B.rows == X.rows && B.cols == X.cols, "B and X must have same dimensions")
	assert(B.rows == AB.rows, "System dimensions must be consistent")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd_val: Blas_Int = kd
	nrhs: Blas_Int = B.cols
	ldab: Blas_Int = AB.ld
	ldafb: Blas_Int = AFB.ld
	ldb: Blas_Int = B.ld
	ldx: Blas_Int = X.ld

	// Allocate error bound arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

	info_val: Info

	when T == f64 {
		// Allocate workspace
		work := make([]f64, 3 * n)
		defer delete()
		iwork := make([]Blas_Int, n)
		defer delete()

		lapack.dpbrfs_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info_val,
			len(uplo_c),
		)
	} else when T == complex128 {
		// Allocate workspace
		work := make([]complex128, 2 * n)
		defer delete()
		rwork := make([]f64, n)
		defer delete()

		lapack.zpbrfs_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(uplo_c),
		)
	}

	return RefinementResult(f64){ferr = ferr, berr = berr, info = info_val}, info_val == 0

}


// ===================================================================================
// SPLIT CHOLESKY FACTORIZATION IMPLEMENTATION
// ===================================================================================

// Split Cholesky factorization for positive definite banded matrix (f32/complex64)
// Computes split factor S from L^T*L where L = S*S^T (or L^H*L where L = S*S^H for complex)
m_split_cholesky_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd_val: Blas_Int = kd
	ldab: Blas_Int = AB.ld
	info_val: Info

	when T == f32 {
		lapack.spbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))
	} else when T == complex64 {
		lapack.cpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))
	}

	return info_val, info_val == 0
}

// Split Cholesky factorization for positive definite banded matrix (f64/complex128)
// Computes split factor S from L^T*L where L = S*S^T (or L^H*L where L = S*S^H for complex)
m_split_cholesky_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd_val: Blas_Int = kd
	ldab: Blas_Int = AB.ld
	info_val: Info

	when T == f64 {
		lapack.dpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))
	} else when T == complex128 {
		lapack.zpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))
	}

	return info_val, info_val == 0
}


// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if refinement significantly improved the solution
is_refinement_successful :: proc(
	result: RefinementResult($T),
	tolerance: T,
) -> bool where is_float(T) ||
	is_complex(T) {
	if len(result.ferr) == 0 {
		return false
	}

	// Check if all forward errors are below tolerance
	for err in result.ferr {
		if err > tolerance {
			return false
		}
	}

	return true
}

// Get maximum error bounds from refinement
get_max_error_bounds :: proc(
	result: RefinementResult($T),
) -> (
	max_ferr, max_berr: T,
) where is_float(T) ||
	is_complex(T) {
	if len(result.ferr) == 0 || len(result.berr) == 0 {
		return T(0), T(0)
	}

	max_ferr = result.ferr[0]
	max_berr = result.berr[0]

	for i in 1 ..< len(result.ferr) {
		if result.ferr[i] > max_ferr {
			max_ferr = result.ferr[i]
		}
		if result.berr[i] > max_berr {
			max_berr = result.berr[i]
		}
	}

	return max_ferr, max_berr
}

// Delete refinement result
delete_refinement_result :: proc(result: ^RefinementResult($T)) {
	if result.ferr != nil {
		delete(result.ferr)
	}
	if result.berr != nil {
		delete(result.berr)
	}
}
