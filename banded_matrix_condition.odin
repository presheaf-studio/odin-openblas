package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX CONDITION NUMBER ESTIMATION AND EQUILIBRATION
// ===================================================================================

// Estimate condition number of positive definite banded matrix proc group
m_estimate_condition_banded_pd :: proc {
	m_estimate_condition_banded_pd_f32_c64,
	m_estimate_condition_banded_pd_f64_c128,
}

// Compute equilibration scaling for positive definite banded matrix proc group
m_compute_equilibration_banded_pd :: proc {
	m_compute_equilibration_banded_pd_f32_c64,
	m_compute_equilibration_banded_pd_f64_c128,
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Estimate condition number of positive definite banded matrix (f32/complex64)
// Estimates reciprocal condition number using factorization from PBTRF
m_estimate_condition_banded_pd_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix factorization from PBTRF
	kd: Blas_Int, // Number of super/sub-diagonals
	anorm: f32, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")
	assert(anorm >= 0, "anorm must be non-negative")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd := kd
	ldab: Blas_Int = AB.ld
	anorm_val := anorm
	rcond_val: f32
	info_val: Info

	when T == f32 {
		// Allocate workspace
		work := make([]f32, 3 * n)
		defer delete(work)
		iwork := make([]Blas_Int, n)
		defer delete(iwork)

		lapack.spbcon_(
			uplo_c,
			&n,
			&kd_val,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			&rcond_val,
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

		lapack.cpbcon_(
			uplo_c,
			&n,
			&kd_val,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			&rcond_val,
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(uplo_c),
		)
	}

	return rcond_val, info_val, info_val == 0
}

// Estimate condition number of positive definite banded matrix (f64/complex128)
// Estimates reciprocal condition number using factorization from PBTRF
m_estimate_condition_banded_pd_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix factorization from PBTRF
	kd: Blas_Int, // Number of super/sub-diagonals
	anorm: f64, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")
	assert(anorm >= 0, "anorm must be non-negative")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)
	anorm_val := anorm
	rcond_val: f64
	info_val: Info

	when T == f64 {
		// Allocate workspace
		work := make([]f64, 3 * n)
		defer delete(work)
		iwork := make([]Blas_Int, n)
		defer delete(iwork)

		lapack.dpbcon_(
			uplo_c,
			&n,
			&kd_val,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			&rcond_val,
			raw_data(work),
			raw_data(iwork),
			&info_val,
			len(uplo_c),
		)
	} else when T == complex128 {
		// Allocate workspace
		work := make([]complex128, 2 * n)
		defer delete(work)
		rwork := make([]f64, n)
		defer delete(rwork)

		lapack.zpbcon_(
			uplo_c,
			&n,
			&kd_val,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			&rcond_val,
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(uplo_c),
		)
	}

	return rcond_val, info_val, info_val == 0
}


// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if positive definite banded matrix is well-conditioned
is_well_conditioned_banded :: proc(
	AB: ^Matrix($T),
	kd: Blas_Int,
	anorm: T,
	tolerance := 1e-10,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	condition_number: f64,
	well_conditioned: bool,
) {
	when T == f32 || T == complex64 {
		rcond, ok, _ := m_estimate_condition_banded_pd_f32_c64(
			AB,
			kd,
			f32(anorm),
			uplo_upper,
			allocator,
		)
		if !ok || rcond == 0 {
			return 0, false
		}
		cond := 1.0 / f64(rcond)
		return rcond > tolerance, cond
	} else when T == f64 || T == complex128 {
		rcond, ok, _ := m_estimate_condition_banded_pd_f64_c128(
			AB,
			kd,
			f64(anorm),
			uplo_upper,
			allocator,
		)
		if !ok || rcond == 0 {
			return false, 0
		}
		cond := 1.0 / rcond
		return cond, rcond > tolerance
	}
}

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX EQUILIBRATION
// ===================================================================================

// Equilibration result structure
EquilibrationResult :: struct($T: typeid) {
	S:     []T, // Scaling factors
	scond: T, // Ratio of smallest to largest scaling factor
	amax:  T, // Maximum absolute value in matrix
	info:  Blas_Int,
}

// Compute equilibration scaling for positive definite banded matrix (f32/complex64)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_f32_c64 :: proc(
	AB: ^Matrix($T), // Positive definite banded matrix
	kd: Blas_Int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	result: EquilibrationResult(f32),
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd := kd
	ldab: Blas_Int = AB.ld

	// Allocate output arrays
	S := make([]f32, n, allocator)
	scond: f32
	amax: f32
	info_val: Info

	when T == f32 {
		lapack.spbequ_(
			uplo_c,
			&n,
			&kd,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			&scond,
			&amax,
			&info_val,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbequ_(
			uplo_c,
			&n,
			&kd,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			&scond,
			&amax,
			&info_val,
			len(uplo_c),
		)
	}

	return EquilibrationResult(f32){S = S, scond = scond, amax = amax, info = info_val},
		info_val == 0

}

// Compute equilibration scaling for positive definite banded matrix (f64/complex128)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_f64_c128 :: proc(
	AB: ^Matrix($T), // Positive definite banded matrix
	kd: Blas_Int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> EquilibrationResult(f64) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(AB.data) > 0, "Matrix cannot be empty")
	assert(AB.rows == AB.cols, "Matrix must be square")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd := kd
	ldab: Blas_Int = AB.ld

	// Allocate output arrays
	S := make([]f64, n, allocator)
	scond: f64
	amax: f64
	info_val: Info

	when T == f64 {
		lapack.dpbequ_(
			uplo_c,
			&n,
			&kd,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			&scond,
			&amax,
			&info_val,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbequ_(
			uplo_c,
			&n,
			&kd,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			&scond,
			&amax,
			&info_val,
			len(uplo_c),
		)
	}

	return EquilibrationResult(f64){S = S, scond = scond, amax = amax, info = info_val},
		info_val == 0
}

// ===================================================================================
// EQUILIBRATION CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if equilibration is needed based on scaling factor condition
needs_equilibration :: proc(
	result: EquilibrationResult($T),
	threshold := 0.1, // Default threshold for scond
) -> bool {
	return result.ok && result.scond < T(threshold)
}

// Apply equilibration scaling to matrix and vector
apply_equilibration_scaling :: proc(
	AB: ^Matrix($T), // Matrix to scale (modified in-place)
	b: []T, // Right-hand side vector (modified in-place)
	S: []T, // Scaling factors from equilibration
	kd: Blas_Int, // Bandwidth
	uplo_upper := true,
) {
	n := AB.cols

	// Scale matrix: A_scaled = S * A * S
	for j in 0 ..< n {
		// Determine row range based on bandwidth
		row_start := max(0, j - kd)
		row_end := min(n, j + kd + 1)

		for i in row_start ..< row_end {
			old_val := matrix_get(AB, i, j)
			when T == complex64 {
				new_val := old_val * complex64(S[i] * S[j])
			} else when T == complex128 {
				new_val := old_val * complex128(S[i] * S[j])
			} else {
				new_val := old_val * T(S[i] * S[j])
			}
			matrix_set(AB, i, j, new_val)
		}
	}

	// Scale right-hand side: b_scaled = S * b
	for i in 0 ..< len(b) {
		when T == complex64 {
			b[i] *= complex64(S[i])
		} else when T == complex128 {
			b[i] *= complex128(S[i])
		} else {
			b[i] *= T(S[i])
		}
	}
}

// Undo equilibration scaling on solution vector
undo_equilibration_scaling :: proc(
	x: []$T, // Solution vector (modified in-place)
	S: []T, // Scaling factors from equilibration
) {
	// Scale solution: x_original = S * x_scaled
	for i in 0 ..< len(x) {
		when T == complex64 {
			x[i] *= complex64(S[i])
		} else when T == complex128 {
			x[i] *= complex128(S[i])
		} else {
			x[i] *= T(S[i])
		}
	}
}

// Complete equilibration workflow for solving systems
equilibrate_and_prepare_system :: proc(
	AB: ^Matrix($T), // Matrix (will be modified if equilibration needed)
	b: []T, // RHS vector (will be modified if equilibration needed)
	kd: Blas_Int, // Bandwidth
	uplo_upper := true,
	equilibration_threshold := 0.1,
	allocator := context.allocator,
) -> (
	scaling: EquilibrationResult(T),
	equilibration_applied: bool,
) where is_float(T) || is_complex(T) {
	// Compute equilibration factors
	when T == f32 || T == complex64 {
		scaling := m_compute_equilibration_banded_pd_f32_c64(AB, kd, uplo_upper, allocator)
	} else when T == f64 || T == complex128 {
		scaling := m_compute_equilibration_banded_pd_f64_c128(AB, kd, uplo_upper, allocator)
	}

	if !scaling.ok {return scaling, false}

	// Check if equilibration is needed
	if needs_equilibration(scaling, equilibration_threshold) {
		// Apply scaling
		apply_equilibration_scaling(AB, b, scaling.S, kd, uplo_upper)
		return scaling, true
	}

	return scaling, false
}

// Compute improved condition number after equilibration
estimate_equilibrated_condition :: proc(
	original_rcond: $T,
	equilibration_result: EquilibrationResult(T),
) -> T {
	if !equilibration_result.ok || equilibration_result.scond == 0 {
		return original_rcond
	}

	// Estimate improvement in condition number
	// The equilibrated matrix typically has better conditioning
	improvement_factor := T(1) / equilibration_result.scond
	return min(T(1), original_rcond * improvement_factor)
}

// Delete equilibration result
delete_equilibration_result :: proc(result: ^EquilibrationResult($T)) {
	if result.S != nil {
		delete(result.S)
	}
}
