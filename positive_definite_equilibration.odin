package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE MATRIX EQUILIBRATION
// ===================================================================================

// Standard equilibration proc group
m_equilibrate_positive_definite :: proc {
	m_equilibrate_positive_definite_f32_c64,
	m_equilibrate_positive_definite_f64_c128,
}

// Equilibration with balancing proc group
m_equilibrate_positive_definite_balanced :: proc {
	m_equilibrate_positive_definite_balanced_f32_c64,
	m_equilibrate_positive_definite_balanced_f64_c128,
}

// ===================================================================================
// STANDARD EQUILIBRATION IMPLEMENTATION
// ===================================================================================

// Compute equilibration scaling factors for positive definite matrix (f32/complex64)
// Computes diagonal scaling S such that S*A*S has unit diagonal
m_equilibrate_positive_definite_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f32, // Pre-allocated scaling factors (size n)
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.ld

	when T == f32 {
		lapack.spoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex64 {
		lapack.cpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Compute equilibration scaling factors for positive definite matrix (f64/complex128)
// Computes diagonal scaling S such that S*A*S has unit diagonal
m_equilibrate_positive_definite_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f64, // Pre-allocated scaling factors (size n)
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.ld

	when T == f64 {
		lapack.dpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex128 {
		lapack.zpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// BALANCED EQUILIBRATION IMPLEMENTATION
// ===================================================================================

// Compute balanced equilibration scaling factors for positive definite matrix (f32/complex64)
// Uses improved algorithm for better numerical stability
m_equilibrate_positive_definite_balanced_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f32, // Pre-allocated scaling factors (size n)
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.ld

	when T == f32 {
		lapack.spoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex64 {
		lapack.cpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Compute balanced equilibration scaling factors for positive definite matrix (f64/complex128)
// Uses improved algorithm for better numerical stability
m_equilibrate_positive_definite_balanced_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f64, // Pre-allocated scaling factors (size n)
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.ld

	when T == f64 {
		lapack.dpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex128 {
		lapack.zpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}
