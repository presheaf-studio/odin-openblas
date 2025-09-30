package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:math"
import "core:slice"

// Specifies the level of accuracy for Jacobi SVD methods


// ===================================================================================
// SINGULAR VALUE DECOMPOSITION
// Decompose matrices as A = U*Sigma*V^T
// ===================================================================================


cs_decomp :: proc {
	cs_decomp_f32_f64,
	cs_decomp_c64_c128,
}


// BIDIAGONAL MATRIX OPERATIONS
// ===================================================================================

// Helper function to prepare CS decomposition parameters
cs_decomp_prepare :: proc(
	U1_shape: [2]Blas_Int,
	U2_shape: [2]Blas_Int,
	V1T_shape: [2]Blas_Int,
	V2T_shape: [2]Blas_Int,
	compute_u1: bool,
	compute_u2: bool,
	compute_v1t: bool,
	compute_v2t: bool,
	trans: bool,
) -> (
	m: Blas_Int,
	p: Blas_Int,
	q: Blas_Int,
	r: Blas_Int,
	ldu1: Blas_Int,
	ldu2: Blas_Int,
	ldv1t: Blas_Int,
	ldv2t: Blas_Int,
	jobu1_c: cstring,
	jobu2_c: cstring,
	jobv1t_c: cstring,
	jobv2t_c: cstring,
	trans_c: cstring,
) {
	// Extract dimensions
	if trans {
		m = U1_shape[1] + U2_shape[1]
		p = U1_shape[1]
		q = U2_shape[1]
	} else {
		m = U1_shape[0] + U2_shape[0]
		p = U1_shape[0]
		q = U2_shape[0]
	}

	r = min(min(p, q), min(m - q, m - p))

	// Set leading dimensions
	ldu1 = U1_shape[1] if U1_shape[1] > 0 else 1
	ldu2 = U2_shape[1] if U2_shape[1] > 0 else 1
	ldv1t = V1T_shape[1] if V1T_shape[1] > 0 else 1
	ldv2t = V2T_shape[1] if V2T_shape[1] > 0 else 1

	// Set job parameters
	jobu1_c = compute_u1 ? cstring("Y") : cstring("N")
	jobu2_c = compute_u2 ? cstring("Y") : cstring("N")
	jobv1t_c = compute_v1t ? cstring("Y") : cstring("N")
	jobv2t_c = compute_v2t ? cstring("Y") : cstring("N")
	trans_c = trans ? cstring("T") : cstring("N")

	return
}

// Helper to allocate bidiagonal arrays for real types
make_bidiag_real_arrays :: proc($T: typeid, r: Blas_Int) -> (b11d: []T, b11e: []T, b12d: []T, b12e: []T, b21d: []T, b21e: []T, b22d: []T, b22e: []T) {
	b11d = make([]T, r)
	b11e = make([]T, r - 1)
	b12d = make([]T, r)
	b12e = make([]T, r - 1)
	b21d = make([]T, r)
	b21e = make([]T, r - 1)
	b22d = make([]T, r)
	b22e = make([]T, r - 1)
	return
}

// Helper to delete bidiagonal arrays
delete_bidiag_real_arrays :: proc(b11d: $T, b11e: T, b12d: T, b12e: T, b21d: T, b21e: T, b22d: T, b22e: T) {
	delete(b11d)
	delete(b11e)
	delete(b12d)
	delete(b12e)
	delete(b21d)
	delete(b21e)
	delete(b22d)
	delete(b22e)
}

// Query result sizes for CS decomposition
query_result_sizes_cs_decomp :: proc(
	p: int, // Rows of U1
	q: int, // Rows of U2
) -> (
	theta_size: int,
	phi_size: int,
	r: int, // min(p, q)
) {
	r = min(p, q)
	theta_size = r
	phi_size = max(0, r - 1)
	return
}

// Query workspace size for CS decomposition
query_workspace_cs_decomp :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	work_size: int,
	rwork_size: int,
	info: Info,
) where is_float(T) ||
	is_complex(T) {
	// Prepare parameters
	U1_shape := [2]Blas_Int{U1.rows, U1.cols}
	U2_shape := [2]Blas_Int{U2.rows, U2.cols}
	V1T_shape := [2]Blas_Int{V1T.rows, V1T.cols}
	V2T_shape := [2]Blas_Int{V2T.rows, V2T.cols}

	m, p, q, r, ldu1, ldu2, ldv1t, ldv2t, jobu1_c, jobu2_c, jobv1t_c, jobv2t_c, trans_c := cs_decomp_prepare(U1_shape, U2_shape, V1T_shape, V2T_shape, compute_u1, compute_u2, compute_v1t, compute_v2t, trans)

	// Create dummy arrays - use f64 for all, transmute for f32
	dummy_theta := [1]f64{}
	dummy_phi := [1]f64{}
	dummy_b11d := [1]f64{}
	dummy_b11e := [1]f64{}
	dummy_b12d := [1]f64{}
	dummy_b12e := [1]f64{}
	dummy_b21d := [1]f64{}
	dummy_b21e := [1]f64{}
	dummy_b22d := [1]f64{}
	dummy_b22e := [1]f64{}
	dummy_rwork := [1]f64{}

	lwork: Blas_Int = -1

	when T == f32 {
		work_query: f32
		lapack.sbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			cast(^f32)&dummy_theta[0],
			cast(^f32)&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			cast(^f32)&dummy_b11d[0],
			cast(^f32)&dummy_b11e[0],
			cast(^f32)&dummy_b12d[0],
			cast(^f32)&dummy_b12e[0],
			cast(^f32)&dummy_b21d[0],
			cast(^f32)&dummy_b21e[0],
			cast(^f32)&dummy_b22d[0],
			cast(^f32)&dummy_b22e[0],
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == f64 {
		work_query: f64
		lapack.dbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			&dummy_theta[0],
			&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			&dummy_b11d[0],
			&dummy_b11e[0],
			&dummy_b12d[0],
			&dummy_b12e[0],
			&dummy_b21d[0],
			&dummy_b21e[0],
			&dummy_b22d[0],
			&dummy_b22e[0],
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == complex64 {
		work_query: complex64
		lapack.cbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			cast(^f32)&dummy_theta[0],
			cast(^f32)&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			cast(^f32)&dummy_b11d[0],
			cast(^f32)&dummy_b11e[0],
			cast(^f32)&dummy_b12d[0],
			cast(^f32)&dummy_b12e[0],
			cast(^f32)&dummy_b21d[0],
			cast(^f32)&dummy_b21e[0],
			cast(^f32)&dummy_b22d[0],
			cast(^f32)&dummy_b22e[0],
			cast(^f32)&dummy_rwork[0],
			(trans ? 8 * r : 7 * r),
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(trans ? 8 * r : 7 * r)

	} else when T == complex128 {
		work_query: complex128
		lapack.zbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			&dummy_theta[0],
			&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			&dummy_b11d[0],
			&dummy_b11e[0],
			&dummy_b12d[0],
			&dummy_b12e[0],
			&dummy_b21d[0],
			&dummy_b21e[0],
			&dummy_b22d[0],
			&dummy_b22e[0],
			&dummy_rwork[0],
			(trans ? 8 * r : 7 * r),
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(trans ? 8 * r : 7 * r)
	}

	return work_size, rwork_size, info
}

// CS Decomposition of unitary/orthogonal matrix partitioned into 2x2 blocks
// Computes the CS decomposition of an orthogonal/unitary matrix in bidiagonal-block form
// Combined f32 and f64 CS decomposition
// Note: Bidiagonal arrays (b11d, b11e, etc.) are workspace arrays - the CS decomposition
// transforms the input matrices in-place
cs_decomp_f32_f64 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	theta: []T, // Cosines (pre-allocated, size r)
	phi: []T, // Sines (pre-allocated, size r-1)
	work: []T, // Workspace (pre-allocated)
	// Bidiagonal workspace arrays (pre-allocated)
	b11d: []T, // Size r
	b11e: []T, // Size r-1
	b12d: []T, // Size r
	b12e: []T, // Size r-1
	b21d: []T, // Size r
	b21e: []T, // Size r-1
	b22d: []T, // Size r
	b22e: []T, // Size r-1
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	// Prepare parameters
	U1_shape := [2]Blas_Int{U1.rows, U1.cols}
	U2_shape := [2]Blas_Int{U2.rows, U2.cols}
	V1T_shape := [2]Blas_Int{V1T.rows, V1T.cols}
	V2T_shape := [2]Blas_Int{V2T.rows, V2T.cols}
	m, p, q, r, ldu1, ldu2, ldv1t, ldv2t, jobu1_c, jobu2_c, jobv1t_c, jobv2t_c, trans_c := cs_decomp_prepare(U1_shape, U2_shape, V1T_shape, V2T_shape, compute_u1, compute_u2, compute_v1t, compute_v2t, trans)

	// Verify array sizes
	assert(len(theta) >= int(r), "theta array too small (need at least r)")
	assert(len(phi) >= int(max(0, r - 1)), "phi array too small (need at least r-1)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_cs_decomp to get size)")

	// Verify bidiagonal workspace arrays
	assert(len(b11d) >= int(r), "b11d array too small (need at least r)")
	assert(len(b11e) >= int(max(0, r - 1)), "b11e array too small (need at least r-1)")
	assert(len(b12d) >= int(r), "b12d array too small (need at least r)")
	assert(len(b12e) >= int(max(0, r - 1)), "b12e array too small (need at least r-1)")
	assert(len(b21d) >= int(r), "b21d array too small (need at least r)")
	assert(len(b21e) >= int(max(0, r - 1)), "b21e array too small (need at least r-1)")
	assert(len(b22d) >= int(r), "b22d array too small (need at least r)")
	assert(len(b22e) >= int(max(0, r - 1)), "b22e array too small (need at least r-1)")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(work),
			&lwork,
			&info,
		)

	} else when T == f64 {
		lapack.dbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}

// Combined complex64 and complex128 CS decomposition
// Note: For complex types, theta and phi are always f32 for complex64 and f64 for complex128
cs_decomp_c64_c128 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	theta: []$F, // Cosines (pre-allocated, size r) - f32 for complex64, f64 for complex128
	phi: []F, // Sines (pre-allocated, size r-1) - f32 for complex64, f64 for complex128
	work: []T, // Complex workspace (pre-allocated)
	rwork: []F, // Real workspace (pre-allocated)
	// Bidiagonal workspace arrays (pre-allocated) - real type matches theta/phi
	b11d: []F, // Size r
	b11e: []F, // Size r-1
	b12d: []F, // Size r
	b12e: []F, // Size r-1
	b21d: []F, // Size r
	b21e: []F, // Size r-1
	b22d: []F, // Size r
	b22e: []F, // Size r-1
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && F == f32) || (T == complex128 && F == f64) {
	// Prepare parameters
	U1_shape := [2]Blas_Int{U1.rows, U1.cols}
	U2_shape := [2]Blas_Int{U2.rows, U2.cols}
	V1T_shape := [2]Blas_Int{V1T.rows, V1T.cols}
	V2T_shape := [2]Blas_Int{V2T.rows, V2T.cols}
	m, p, q, r, ldu1, ldu2, ldv1t, ldv2t, jobu1_c, jobu2_c, jobv1t_c, jobv2t_c, trans_c := cs_decomp_prepare(U1_shape, U2_shape, V1T_shape, V2T_shape, compute_u1, compute_u2, compute_v1t, compute_v2t, trans)

	// Verify array sizes
	assert(len(theta) >= int(r), "theta array too small (need at least r)")
	assert(len(phi) >= int(max(0, r - 1)), "phi array too small (need at least r-1)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_cs_decomp to get size)")

	// Verify real workspace
	lrwork := Blas_Int((8 * r) if trans else (7 * r))
	assert(len(rwork) >= int(lrwork), "rwork array too small")

	// Verify bidiagonal workspace arrays
	assert(len(b11d) >= int(r), "b11d array too small (need at least r)")
	assert(len(b11e) >= int(max(0, r - 1)), "b11e array too small (need at least r-1)")
	assert(len(b12d) >= int(r), "b12d array too small (need at least r)")
	assert(len(b12e) >= int(max(0, r - 1)), "b12e array too small (need at least r-1)")
	assert(len(b21d) >= int(r), "b21d array too small (need at least r)")
	assert(len(b21e) >= int(max(0, r - 1)), "b21e array too small (need at least r-1)")
	assert(len(b22d) >= int(r), "b22d array too small (need at least r)")
	assert(len(b22e) >= int(max(0, r - 1)), "b22e array too small (need at least r-1)")

	lwork := Blas_Int(len(work))

	when T == complex64 {
		// For complex64, theta and phi are f32 arrays
		lapack.cbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == complex128 {
		// For complex128, theta and phi are f64 arrays

		lapack.zbbcsd_(
			&jobu1_c,
			&jobu2_c,
			&jobv1t_c,
			&jobv2t_c,
			&trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}
