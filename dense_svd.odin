package openblas

import lapack "./f77"
// import "base:builtin"
// import "base:intrinsics"
// import "core:math"
// import "core:slice"

// ===================================================================================
// GENERAL SVD (SINGULAR VALUE DECOMPOSITION) FOR DENSE MATRICES
// ===================================================================================

// ===================================================================================
// SVD JOB CONTROL ENUMS
// ===================================================================================

SVD_Job :: enum u8 {
	All       = 'A', // all columns/rows are returned in the destination array
	Some      = 'S', // the first min(m,n) columns/rows are returned in the destination array
	Overwrite = 'O', // the first min(m,n) columns/rows are overwritten on the array A
	None      = 'N', // no columns/rows are computed
}

// Job options for QR-based SVD U matrix (left singular vectors)
SVD_QR_JobU :: enum u8 {
	All      = 'A', // All M left singular vectors are returned in U
	Some     = 'S', // min(M,N) left singular vectors are returned in U (same as 'U')
	Min      = 'U', // min(M,N) left singular vectors are returned in U (same as 'S')
	Rank     = 'R', // Only NUMRANK left singular vectors based on numerical rank
	Factored = 'F', // Left singular vectors returned in factored form as Q*U
	None     = 'N', // Left singular vectors are not computed
}

// Job options for QR-based SVD V matrix (right singular vectors)
SVD_QR_JobV :: enum u8 {
	All  = 'A', // All N right singular vectors are returned in V (same as 'V')
	Full = 'V', // All N right singular vectors are returned in V (same as 'A')
	Rank = 'R', // Only NUMRANK right singular vectors (only if JobU='R' or 'N')
	None = 'N', // Right singular vectors are not computed
}

// Specifies the level of accuracy in the computed SVD (for QR-based methods)
SVD_Accuracy :: enum u8 {
	Aggressive = 'A', // Aggressive truncation: ||delta A||_F <= f(m,n)*EPS*||A||_F
	Medium     = 'M', // Medium truncation: only when diagonal drops in QR factorization
	High       = 'H', // High accuracy: no rank determination based on QR factorization
	Extended   = 'E', // Same as High + condition number estimation in rwork[0]
}

SVD_Jacobi_Accuracy :: enum u8 {
	High       = 'H', // High accuracy requested
	Controlled = 'C', // Controlled (lower) accuracy, faster computation
}

// Job options for Jacobi SVD U/V computation
SVD_Jacobi_Job :: enum u8 {
	Compute = 'U', // Compute singular vectors (U for left, V for right)
	Vectors = 'V', // Alternative name for compute (used for V)
	None    = 'N', // Do not compute singular vectors
}

// Job options for Jacobi SVD range restriction
SVD_Jacobi_Range :: enum u8 {
	Restrict = 'R', // Restrict range of singular values
	None     = 'N', // No range restriction
}

// Job options for Jacobi SVD transpose hint
SVD_Jacobi_Transpose :: enum u8 {
	Transpose = 'T', // Matrix is transposed
	None      = 'N', // Matrix is not transposed
}

// Job options for Jacobi SVD perturbation
SVD_Jacobi_Perturb :: enum u8 {
	Perturb = 'P', // Perturb small singular values
	None    = 'N', // No perturbation
}

// Job options for Jacobi variant matrix format (gesvj)
SVD_Jacobi_Variant_Matrix :: enum u8 {
	General         = 'G', // General matrix
	UpperTriangular = 'U', // Upper triangular matrix
}

// Job options for Jacobi variant U/V computation (gesvj)
SVD_Jacobi_Variant_Job :: enum u8 {
	Compute = 'U', // Compute singular vectors (U for left)
	Vectors = 'V', // Alternative name (V for right)
	None    = 'N', // Do not compute singular vectors
}

SVD_Range_Option :: enum u8 {
	All   = 'A', // all singular values will be found.
	Value = 'V', //  all singular values in the half-open interval (VL,VU] will be found.
	Index = 'I', // the IL-th through IU-th singular values will be found.
}

// Job options for selective SVD (gesvdx)
SVD_Select_Job :: enum u8 {
	Vectors = 'V', // Compute singular vectors
	None    = 'N', // Do not compute singular vectors
}

// ===================================================================================
// SVD PROCEDURE GROUPS
// ===================================================================================

dns_svd_simple :: proc {
	dns_svd_simple_real,
	dns_svd_simple_complex,
}

dns_svd_qr :: proc {
	dns_svd_qr_real,
	dns_svd_qr_complex,
}

dns_svd_dc :: proc {
	dns_svd_dc_real,
	dns_svd_dc_complex,
}

dns_svd_select :: proc {
	dns_svd_select_real,
	dns_svd_select_complex,
}

// Compute SVD using Jacobi method (highest accuracy)
// Especially good for small matrices and when high accuracy is needed
dns_svd_jacobi :: proc {
	dns_svd_jacobi_real,
	dns_svd_jacobi_complex,
}

dns_svd_jacobi_variant :: proc {
	dns_svd_jacobi_variant_real,
	dns_svd_jacobi_variant_complex,
}

// ===================================================================================
// STANDARD SVD
// ===================================================================================

// Query result sizes for SVD
query_result_sizes_dns_svd_simple :: proc(A: ^Matrix($T), jobu: SVD_Job = .Some, jobvt: SVD_Job = .Some) -> (S_size: int, U_rows: int, U_cols: int, VT_rows: int, VT_cols: int) where is_float(T) || is_complex(T) {
	assert(!(jobu == .Overwrite && jobvt == .Overwrite), "Cannot store U and VT in A")
	m := int(A.rows)
	n := int(A.cols)
	min_mn := min(m, n)

	S_size = min_mn

	if jobu != .None {
		U_rows = m
		U_cols = jobu == .All ? m : min_mn
	}

	if jobvt != .None {
		VT_rows = jobvt == .All ? n : min_mn
		VT_cols = n
	}

	return S_size, U_rows, U_cols, VT_rows, VT_cols
}

// Query workspace size for SVD (both real and complex)
// For real types, rwork_size will be 0
query_workspace_dns_svd_simple :: proc(A: ^Matrix($T), jobu: SVD_Job = .Some, jobvt: SVD_Job = .Some) -> (work_size: int, rwork_size: int, info: Info) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	jobu_c := cast(u8)jobu
	jobvt_c := cast(u8)jobvt

	lwork: Blas_Int = QUERY_WORKSPACE
	ldu: Blas_Int = 1
	ldvt: Blas_Int = 1

	dummy_s := [1]T{}

	when T == f32 {
		work_query: f32
		lapack.sgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldvt, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0 // Not used for real types
	} else when T == f64 {
		work_query: f64
		lapack.dgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldvt, &work_query, &lwork, &info)
		work_size = int(work_query)
		rwork_size = 0 // Not used for real types
	} else when T == complex64 {
		work_query: complex64
		dummy_rwork := [1]f32{}

		lapack.cgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldvt, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = 5 * int(min_mn) // Complex types need real workspace
	} else when T == complex128 {
		work_query: complex128
		dummy_rwork := [1]f64{}

		lapack.zgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldvt, &work_query, &lwork, &dummy_rwork[0], &info)
		work_size = int(real(work_query))
		rwork_size = 5 * int(min_mn) // Complex types need real workspace
	}

	return work_size, rwork_size, info
}

// Compute SVD using standard algorithm
// A = U * Sigma * V^T
dns_svd_simple_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated, optional)
	jobu: SVD_Job = .Some,
	jobvt: SVD_Job = .Some,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	jobu_c := cast(char)jobu
	jobvt_c := cast(char)jobvt

	ldu: Blas_Int = 1
	ptr_u: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		u_cols := jobu == .All ? int(m) : int(min_mn)
		assert(U.rows == Blas_Int(m) && U.cols >= Blas_Int(u_cols), "U matrix dimensions incorrect")
		ptr_u = raw_data(U.data)
	}

	ldvt: Blas_Int = 1
	ptr_vt: ^Cmplx = nil
	if jobvt != .None && VT != nil {
		ldvt = VT.ld
		vt_rows := jobvt == .All ? int(n) : int(min_mn)
		assert(VT.rows >= Blas_Int(vt_rows) && VT.cols == n, "VT matrix dimensions incorrect")
		ptr_vt = raw_data(VT.data)
	}

	assert(len(work) > 0, "work array must be provided (use query_workspace_svd to get size)")
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

dns_svd_simple_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated) - f32 for complex64, f64 for complex128
	U: ^Matrix(Cmplx), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(Cmplx), // Right singular vectors transposed (pre-allocated, optional)
	work: []Cmplx, // Workspace (pre-allocated, optional)
	rwork: []Real, // Real workspace for complex (pre-allocated)
	jobu: SVD_Job = .Some,
	jobvt: SVD_Job = .Some,
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	jobu_c := cast(char)jobu
	jobvt_c := cast(char)jobvt

	ldu: Blas_Int = 1
	ptr_u: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		u_cols := jobu == .All ? int(m) : int(min_mn)
		assert(U.rows == m && U.cols >= Blas_Int(u_cols), "U matrix dimensions incorrect")
		ptr_u = raw_data(U.data)
	}

	ldvt: Blas_Int = 1
	ptr_vt: ^Cmplx = nil
	if jobvt != .None && VT != nil {
		ldvt = VT.ld
		vt_rows := jobvt == .All ? int(n) : int(min_mn)
		assert(VT.rows >= Blas_Int(vt_rows) && VT.cols == n, "VT matrix dimensions incorrect")
		ptr_vt = raw_data(VT.data)
	}

	assert(len(work) > 0, "work array must be provided (use query_workspace_svd to get size)")
	lwork := Blas_Int(len(work))

	assert(len(rwork) >= 5 * int(min_mn), "rwork array too small (need at least 5*min(m,n))")

	when T == complex64 {
		lapack.cgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zgesvd_(&jobu_c, &jobvt_c, &m, &n, raw_data(A.data), &lda, raw_data(S), ptr_u, &ldu, ptr_vt, &ldvt, raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// QR-BASED SVD WITH PIVOTING
// ===================================================================================

// Query result sizes for QR-based SVD
query_result_sizes_dns_svd_qr :: proc(A: ^Matrix($T), jobu: SVD_Job = .All, jobv: SVD_Job = .All) -> (S_size: int, U_rows: int, U_cols: int, V_rows: int, V_cols: int, iwork_size: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	min_mn := min(m, n)
	S_size = min_mn

	if jobu != .None {
		U_rows = m
		U_cols = m
	}

	if jobv != .None {
		V_rows = n
		V_cols = n
	}

	// Integer workspace size
	iwork_size = 3 * min_mn

	return S_size, U_rows, U_cols, V_rows, V_cols, iwork_size
}

// Query workspace sizes for QR-based SVD
// For real types, rwork also used (unlike standard SVD)
query_workspace_dns_svd_qr :: proc(A: ^Matrix($T), jobu: SVD_QR_JobU = .All, jobv: SVD_QR_JobV = .All, accuracy: SVD_Accuracy = .High, pivot: bool = true, rank_reveal: bool = true) -> (work_size: int, rwork_size: int, iwork_size: int, info: Info) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	joba_c := cast(u8)accuracy
	jobp_c := pivot ? u8('P') : u8('N')
	jobr_c := rank_reveal ? u8('R') : u8('N')
	jobu_c := cast(u8)jobu // Convert enum to char for LAPACK
	jobv_c := cast(u8)jobv // Convert enum to char for LAPACK

	iwork_size = 3 * int(min_mn)

	liwork := QUERY_WORKSPACE
	lwork := QUERY_WORKSPACE
	lrwork := QUERY_WORKSPACE
	iwork_query: Blas_Int
	numrank: Blas_Int

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1

	when T == f32 {
		work_query: f32
		rwork_query: f32
		dummy_s := [1]f32{}

		lapack.sgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldv, &numrank, &iwork_query, &liwork, &work_query, &lwork, &rwork_query, &lrwork, &info)

		work_size = int(work_query)
		rwork_size = int(rwork_query)

	} else when T == f64 {
		work_query: f64
		rwork_query: f64
		dummy_s := [1]f64{}

		lapack.dgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldv, &numrank, &iwork_query, &liwork, &work_query, &lwork, &rwork_query, &lrwork, &info)

		work_size = int(work_query)
		rwork_size = int(rwork_query)

	} else when T == complex64 {
		cwork_query: complex64
		rwork_query: f32
		dummy_s := [1]f32{}

		lapack.cgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldv, &numrank, &iwork_query, &liwork, &cwork_query, &lwork, &rwork_query, &lrwork, &info)

		work_size = int(real(cwork_query))
		rwork_size = int(rwork_query)

	} else when T == complex128 {
		zwork_query: complex128
		rwork_query: f64
		dummy_s := [1]f64{}

		lapack.zgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, &dummy_s[0], nil, &ldu, nil, &ldv, &numrank, &iwork_query, &liwork, &zwork_query, &lwork, &rwork_query, &lrwork, &info)

		work_size = int(real(zwork_query))
		rwork_size = int(rwork_query)
	}

	// Use the queried iwork size
	if iwork_query > 0 {
		iwork_size = int(iwork_query)
	}

	return work_size, rwork_size, iwork_size, info
}

// SVD using QR factorization with column pivoting
// High accuracy and rank-revealing, especially for rank-deficient matrices
// Note: Caller should resize S, U.cols, V.cols based on numrank if rank_reveal is true
dns_svd_qr_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T), // Right singular vectors NOT transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated, optional)
	rwork: []T, // Real workspace (pre-allocated, optional - used for real types)
	jobu: SVD_QR_JobU = .All,
	jobv: SVD_QR_JobV = .All,
	accuracy: SVD_Accuracy = .High, // Accuracy level for QR factorization
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
) -> (
	numerical_rank: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	joba_c := cast(u8)accuracy // H=high, M=medium accuracy
	jobp_c := pivot ? u8('P') : u8('N') // P=pivot, N=no pivot
	jobr_c := rank_reveal ? u8('R') : u8('N') // R=rank revealing
	jobu_c := cast(u8)jobu // Convert enum to char for LAPACK
	jobv_c := cast(u8)jobv // Convert enum to char for LAPACK

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect (should be m x m)")
		u_ptr = raw_data(U.data)
	}

	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect (should be n x n)")
		v_ptr = raw_data(V.data)
	}

	assert(len(iwork) > 0, "iwork array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(rwork) > 0, "rwork array must be provided (use query_workspace_svd_qr to get size)")

	liwork := Blas_Int(len(iwork))
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when T == f32 {
		lapack.sgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, &numerical_rank, raw_data(iwork), &liwork, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	} else when T == f64 {
		lapack.dgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, &numerical_rank, raw_data(iwork), &liwork, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	}

	return numerical_rank, info, info == 0
}

// Combined complex64 and complex128 SVD with QR (pre-allocated arrays)
// Note: Caller should resize S, U.cols, V.cols based on numrank if rank_reveal is true
dns_svd_qr_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated) - f32 for complex64, f64 for complex128
	U: ^Matrix(Cmplx), // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(Cmplx), // Right singular vectors NOT transposed (pre-allocated, optional)
	iwork: []Blas_Int = nil, // Integer workspace (pre-allocated, optional)
	work: []Cmplx = nil, // Workspace (pre-allocated, optional)
	rwork: []Real = nil, // Real workspace (pre-allocated, required for complex)
	jobu: SVD_QR_JobU = .All,
	jobv: SVD_QR_JobV = .All,
	accuracy: SVD_Accuracy = .High, // Accuracy level for QR factorization
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
) -> (
	numrank: Blas_Int,
	info: Info,
	ok: bool, // Numerical rank
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	joba_c := cast(u8)accuracy // Convert enum to char for LAPACK
	jobp_c := pivot ? u8('P') : u8('N') // P=pivot, N=no pivot
	jobr_c := rank_reveal ? u8('R') : u8('N') // R=rank revealing
	jobu_c := cast(u8)jobu // Convert enum to char for LAPACK
	jobv_c := cast(u8)jobv // Convert enum to char for LAPACK

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect (should be m x m)")
		u_ptr = raw_data(U.data)
	}

	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect (should be n x n)")
		v_ptr = raw_data(V.data)
	}

	assert(len(iwork) > 0, "iwork array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(rwork) > 0, "rwork array must be provided (use query_workspace_svd_qr to get size)")

	liwork := Blas_Int(len(iwork))
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when T == complex64 {
		lapack.cgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, &numrank, raw_data(iwork), &liwork, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	} else when T == complex128 {
		lapack.zgesvdq_(joba_c, jobp_c, jobr_c, jobu_c, jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, &numrank, raw_data(iwork), &liwork, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	}

	return numrank, info, info == 0
}

// ===================================================================================
// DIVIDE-AND-CONQUER SVD
// ===================================================================================

// Query result sizes for divide-and-conquer SVD
// Same as regular SVD result sizes
query_result_sizes_dns_svd_dc :: proc(A: ^Matrix($T), jobz: SVD_Job = .Some) -> (S_size: int, U_rows: int, U_cols: int, VT_rows: int, VT_cols: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	min_mn := min(m, n)

	S_size = min_mn

	switch jobz {
	case .None:
	// No vectors computed
	case .Overwrite:
		if m >= n {
			U_rows = m
			U_cols = n
		} else {
			VT_rows = m
			VT_cols = n
		}
	case .Some:
		U_rows = m
		U_cols = min_mn
		VT_rows = min_mn
		VT_cols = n
	case .All:
		U_rows = m
		U_cols = m
		VT_rows = n
		VT_cols = n
	}

	return
}

// Query workspace size for divide-and-conquer SVD
query_workspace_dns_svd_dc :: proc(A: ^Matrix($T), jobz: SVD_Job = .Some) -> (work_size: int, rwork_size: int, iwork_size: int, info: Info) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	jobz_c := cast(u8)jobz

	iwork_size = int(8 * min_mn)

	ldu: Blas_Int = 1
	ldvt: Blas_Int = 1
	switch jobz {
	case .None:
	// No vectors
	case .Overwrite:
		if m >= n {
			ldu = m
		} else {
			ldvt = n
		}
	case .Some:
		ldu = m
		ldvt = min_mn
	case .All:
		ldu = m
		ldvt = n
	}

	lwork: Blas_Int = QUERY_WORKSPACE
	dummy_iwork := [1]Blas_Int{}
	work_query: T
	dummy_s := [1]T{}
	rwork_size = 0

	when T == f32 {
		lapack.sgesdd_(
			&jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)

	} else when T == f64 {
		lapack.dgesdd_(
			&jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
	} else when T == complex64 {
		dummy_rwork := [1]f32{}
		lapack.cgesdd_(
			&jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * min_mn + 1))
		if jobz == .All {
			rwork_size = max(rwork_size, int(5 * min_mn * min_mn + 5 * min_mn))
		}

	} else when T == complex128 {
		dummy_rwork := [1]f64{}
		lapack.zgesdd_(
			&jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * min_mn + 1))
		if jobz == .All {
			rwork_size = max(rwork_size, int(5 * min_mn * min_mn + 5 * min_mn))
		}
	}

	return work_size, rwork_size, iwork_size, info
}

// Compute SVD using divide-and-conquer algorithm
// Faster than standard SVD for large matrices
// Combined f32 real and complex64 divide-and-conquer SVD
dns_svd_dc_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	rwork: []T = nil, // Real workspace (pre-allocated, optional)
	jobz: SVD_Job = .Some,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	jobz_c := cast(u8)jobz

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	ldvt: Blas_Int = 1
	vt_ptr: ^T = nil

	switch jobz {
	case .None:
	// No vectors computed
	case .Overwrite:
		if m >= n && U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= n, "U matrix dimensions incorrect for overwrite mode")
			u_ptr = raw_data(U.data)
		} else if m < n && VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= m && VT.cols == n, "VT matrix dimensions incorrect for overwrite mode")
			vt_ptr = raw_data(VT.data)
		}
	case .Some:
		if U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= min_mn, "U matrix dimensions incorrect")
			u_ptr = raw_data(U.data)
		}
		if VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= min_mn && VT.cols == n, "VT matrix dimensions incorrect")
			vt_ptr = raw_data(VT.data)
		}
	case .All:
		if U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= m, "U matrix dimensions incorrect")
			u_ptr = raw_data(U.data)
		}
		if VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= n && VT.cols == n, "VT matrix dimensions incorrect")
			vt_ptr = raw_data(VT.data)
		}
	}

	// Verify workspace sizes
	assert(len(iwork) >= int(8 * min_mn), "iwork array too small (need at least 8*min(m,n))")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_dc to get size)")
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgesdd_(&jobz_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgesdd_(&jobz_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(iwork), &info)
	}

	return info, info == 0
}

// Combined f64 real and complex128 divide-and-conquer SVD
dns_svd_dc_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated) - f32 for complex64, f64 for complex128
	U: ^Matrix(Cmplx), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(Cmplx), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real = nil, // Real workspace for complex (pre-allocated, required)
	jobz: SVD_Job = .Some,
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")

	jobz_c := cast(u8)jobz

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	ldvt: Blas_Int = 1
	vt_ptr: ^T = nil

	switch jobz {
	case .None:
	// No vectors computed
	case .Overwrite:
		if m >= n && U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= n, "U matrix dimensions incorrect for overwrite mode")
			u_ptr = raw_data(U.data)
		} else if m < n && VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= m && VT.cols == n, "VT matrix dimensions incorrect for overwrite mode")
			vt_ptr = raw_data(VT.data)
		}
	case .Some:
		if U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= min_mn, "U matrix dimensions incorrect")
			u_ptr = raw_data(U.data)
		}
		if VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= min_mn && VT.cols == n, "VT matrix dimensions incorrect")
			vt_ptr = raw_data(VT.data)
		}
	case .All:
		if U != nil {
			ldu = U.ld
			assert(U.rows == m && U.cols >= m, "U matrix dimensions incorrect")
			u_ptr = raw_data(U.data)
		}
		if VT != nil {
			ldvt = VT.ld
			assert(VT.rows >= n && VT.cols == n, "VT matrix dimensions incorrect")
			vt_ptr = raw_data(VT.data)
		}
	}

	assert(len(iwork) >= int(8 * min_mn), "iwork array too small (need at least 8*min(m,n))")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_dc to get size)")
	lwork := Blas_Int(len(work))

	when T == complex64 {
		rwork_size := min_mn * max(5 * min_mn + 7, 2 * min_mn + 1)
		if jobz == .All {
			rwork_size = max(rwork_size, 5 * min_mn * min_mn + 5 * min_mn)
		}
		assert(len(rwork) >= int(rwork_size), "rwork array too small for complex divide-and-conquer SVD")

		lapack.cgesdd_(&jobz_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	} else when T == complex128 {
		rwork_size := min_mn * max(5 * min_mn + 7, 2 * min_mn + 1)
		if jobz == .All {
			rwork_size = max(rwork_size, 5 * min_mn * min_mn + 5 * min_mn)
		}
		assert(len(rwork) >= int(rwork_size), "rwork array too small for complex divide-and-conquer SVD")

		lapack.zgesdd_(&jobz_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// SELECTIVE SVD
// ===================================================================================
// Query result sizes for selective SVD
// Returns sizes based on the selection criteria
// max_ns: Maximum number of singular values that could be found
query_result_sizes_dns_svd_select :: proc(
	A: ^Matrix($T),
	range_mode: SVD_Range_Option = .All,
	il: Blas_Int = 1, // Lower index (for range=.Index)
	iu: Blas_Int = -1, // Upper index (for range=.Index, -1 = min(m,n))
	jobu: SVD_Select_Job = .Vectors,
	jobvt: SVD_Select_Job = .Vectors,
) -> (
	max_ns: int,
	S_size: int,
	U_rows: int,
	U_cols: int,
	VT_rows: int,
	VT_cols: int,
) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	min_mn := Blas_Int(min(m, n))

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	switch range_mode {
	case .All:
		max_ns = int(min_mn)
	case .Index:
		max_ns = int(iu_val - il + 1)
	case .Value:
		max_ns = int(min_mn) // Conservative estimate for value range
	}

	S_size = max_ns

	if jobu == .Vectors {
		U_rows = m
		U_cols = max_ns
	}

	if jobvt == .Vectors {
		VT_rows = max_ns
		VT_cols = n
	}

	return
}

// Query workspace size for selective SVD
query_workspace_dns_svd_select :: proc(
	A: ^Matrix($T),
	range_mode: SVD_Range_Option = .All,
	vl: $Real, // Lower bound (for range=.Value) - no default value possible for generic types
	vu: Real, // Upper bound (for range=.Value) - no default value possible for generic types
	il: Blas_Int = 1, // Lower index (for range=.Index)
	iu: Blas_Int = -1, // Upper index (for range=.Index, -1 = min(m,n))
	jobu: SVD_Select_Job = .Vectors,
	jobvt: SVD_Select_Job = .Vectors,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info,
) where (T == f32 && Real == f32) || (T == f64 && Real == f64) || (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	range_c := cast(u8)range_mode
	jobu_c := cast(u8)jobu
	jobvt_c := cast(u8)jobvt

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size for rwork calculation
	max_ns: Blas_Int
	switch range_mode {
	case .All:
		max_ns = min_mn
	case .Index:
		max_ns = iu_val - il + 1
	case .Value:
		max_ns = min_mn // Conservative estimate for value range
	}

	iwork_size = int(12 * min_mn)

	ldu: Blas_Int = 1
	if jobu == .Vectors {
		ldu = m
	}

	ldvt: Blas_Int = 1
	if jobvt == .Vectors {
		ldvt = max_ns
	}

	lwork: Blas_Int = QUERY_WORKSPACE
	ns: Blas_Int
	dummy_iwork := [1]Blas_Int{}

	when T == f32 {
		work_query: f32
		dummy_s := [1]f32{}
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)

		lapack.sgesvdx_(
			&jobu_c,
			&jobvt_c,
			&range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f32,
			&vu_f32,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == f64 {
		work_query: f64
		dummy_s := [1]f64{}
		vl_f64 := f64(vl)
		vu_f64 := f64(vu)

		lapack.dgesvdx_(
			&jobu_c,
			&jobvt_c,
			&range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f64,
			&vu_f64,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == complex64 {
		cwork_query: complex64
		dummy_s := [1]f32{}
		dummy_rwork := [1]f32{}
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)

		lapack.cgesvdx_(
			&jobu_c,
			&jobvt_c,
			&range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f32,
			&vu_f32,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&cwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(cwork_query))
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1)))

	} else when T == complex128 {
		zwork_query: complex128
		dummy_s := [1]f64{}
		dummy_rwork := [1]f64{}
		vl_f64 := f64(vl)
		vu_f64 := f64(vu)

		lapack.zgesvdx_(
			&jobu_c,
			&jobvt_c,
			&range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f64,
			&vu_f64,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&zwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(zwork_query))
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1)))
	}

	return work_size, rwork_size, iwork_size, info
}
// Compute selected singular values and vectors
// Can compute subset by index range or value range
// Combined f32/f64 real selective SVD
dns_svd_select_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	range_mode: SVD_Range_Option = .All,
	vl: T, // Lower bound (for range=.Value)
	vu: T, // Upper bound (for range=.Value)
	il: Blas_Int = 1, // Lower index (for range=.Index)
	iu: Blas_Int = -1, // Upper index (for range=.Index, -1 = min(m,n))
	jobu: SVD_Select_Job = .Vectors,
	jobvt: SVD_Select_Job = .Vectors,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool, // Number of singular values found
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	range_c := cast(u8)range_mode
	jobu_c := cast(u8)jobu
	jobvt_c := cast(u8)jobvt

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	max_ns: Blas_Int
	switch range_mode {
	case .All:
		max_ns = min_mn
	case .Index:
		max_ns = iu_val - il + 1
	case .Value:
		max_ns = min_mn // Conservative estimate for value range
	}

	assert(len(S) >= int(max_ns), "S array too small for potential output")

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu == .Vectors && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols >= max_ns, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}

	ldvt: Blas_Int = 1
	vt_ptr: ^T = nil
	if jobvt == .Vectors && VT != nil {
		ldvt = VT.ld
		assert(VT.rows >= max_ns && VT.cols == n, "VT matrix dimensions incorrect")
		vt_ptr = raw_data(VT.data)
	}

	assert(len(iwork) >= int(12 * min_mn), "iwork array too small (need at least 12*min(m,n))")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_select to get size)")
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgesvdx_(&jobu_c, &jobvt_c, &range_c, &m, &n, raw_data(A.data), &lda, &vl, &vu, &il, &iu_val, &ns, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgesvdx_(&jobu_c, &jobvt_c, &range_c, &m, &n, raw_data(A.data), &lda, &vl, &vu, &il, &iu_val, &ns, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(iwork), &info)
	}

	// Note: Caller should resize S, U.cols, VT.rows based on ns if needed
	return ns, info, info == 0
}

// Combined complex64/complex128 selective SVD
dns_svd_select_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated)
	U: ^Matrix(Cmplx), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(Cmplx), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace for complex (pre-allocated)
	range_mode: SVD_Range_Option = .All,
	vl: Real, // Lower bound (for range=.Value)
	vu: Real, // Upper bound (for range=.Value)
	il: Blas_Int = 1, // Lower index (for range=.Index)
	iu: Blas_Int = -1, // Upper index (for range=.Index, -1 = min(m,n))
	jobu: SVD_Select_Job = .Vectors,
	jobvt: SVD_Select_Job = .Vectors,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	range_c := cast(u8)range_mode
	jobu_c := cast(u8)jobu
	jobvt_c := cast(u8)jobvt

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	max_ns: Blas_Int
	switch range_mode {
	case .All:
		max_ns = min_mn
	case .Index:
		max_ns = iu_val - il + 1
	case .Value:
		max_ns = min_mn // Conservative estimate for value range
	}

	assert(len(S) >= int(max_ns), "S array too small for potential output")

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu == .Vectors && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols >= max_ns, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}

	ldvt: Blas_Int = 1
	vt_ptr: ^T = nil
	if jobvt == .Vectors && VT != nil {
		ldvt = VT.ld
		assert(VT.rows >= max_ns && VT.cols == n, "VT matrix dimensions incorrect")
		vt_ptr = raw_data(VT.data)
	}

	assert(len(iwork) >= int(12 * min_mn), "iwork array too small (need at least 12*min(m,n))")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_select to get size)")
	lwork := Blas_Int(len(work))

	max_rwork := min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1))
	assert(len(rwork) >= int(max_rwork), "rwork array too small for complex selective SVD")

	when T == complex64 {
		lapack.cgesvdx_(&jobu_c, &jobvt_c, &range_c, &m, &n, raw_data(A.data), &lda, &vl, &vu, &il, &iu_val, &ns, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	} else when T == complex128 {
		lapack.zgesvdx_(&jobu_c, &jobvt_c, &range_c, &m, &n, raw_data(A.data), &lda, &vl, &vu, &il, &iu_val, &ns, raw_data(S), u_ptr, &ldu, vt_ptr, &ldvt, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	}

	// Note: Caller should resize S, U.cols, VT.rows based on ns if needed
	return ns, info, info == 0
}


// ===================================================================================
// JACOBI SVD (HIGH ACCURACY) - WORKSPACE QUERIES
// ===================================================================================

// Query workspace size for Jacobi SVD (high accuracy)
query_workspace_dns_svd_jacobi :: proc(
	A: ^Matrix($T),
	jobu: SVD_Jacobi_Job = .Compute,
	jobv: SVD_Jacobi_Job = .Vectors,
	accuracy: SVD_Jacobi_Accuracy = .High,
	jobr: SVD_Jacobi_Range = .None,
	jobt: SVD_Jacobi_Transpose = .None,
	jobp: SVD_Jacobi_Perturb = .None,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info, // For complex types
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	joba_c := cast(u8)accuracy
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobr_c := cast(u8)jobr
	jobt_c := cast(u8)jobt
	jobp_c := cast(u8)jobp

	iwork_size = int(m + 3 * n)

	dummy_s := [1]f64{}
	dummy_iwork := [1]Blas_Int{}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	lwork: Blas_Int = QUERY_WORKSPACE
	work_query: T
	rwork_size = 0

	when T == f32 {
		lapack.sgejsv_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&jobr_c,
			&jobt_c,
			&jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dgejsv_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&jobr_c,
			&jobt_c,
			&jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(work_query)
	} else when T == complex64 {
		rwork_query: f32
		lrwork: Blas_Int = -1

		lapack.cgejsv_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&jobr_c,
			&jobt_c,
			&jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)

	} else when T == complex128 {
		rwork_query: f64
		lrwork: Blas_Int = -1

		lapack.zgejsv_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&jobr_c,
			&jobt_c,
			&jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	}

	return work_size, rwork_size, iwork_size, info
}

// Query result sizes for Jacobi SVD
query_result_sizes_dns_svd_jacobi :: proc(A: ^Matrix($T), jobu: SVD_Jacobi_Job = .Compute, jobv: SVD_Jacobi_Job = .Vectors) -> (s_size: int, u_rows: int, u_cols: int, v_rows: int, v_cols: int, work_stat_size: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	min_mn := min(m, n)

	s_size = min_mn
	work_stat_size = 7 // Statistics array size

	if jobu != .None {
		u_rows = m
		u_cols = m
	}

	if jobv != .None {
		v_rows = n
		v_cols = n
	}

	return s_size, u_rows, u_cols, v_rows, v_cols, work_stat_size
}

// Combined f32/f64 real Jacobi SVD (pre-allocated arrays)
dns_svd_jacobi_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors NOT transposed (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work_stat: []T, // Statistics array (pre-allocated, size 7)
	jobu: SVD_Jacobi_Job = .Compute,
	jobv: SVD_Jacobi_Job = .Vectors,
	accuracy: SVD_Jacobi_Accuracy = .High,
	jobr: SVD_Jacobi_Range = .None,
	jobt: SVD_Jacobi_Transpose = .None,
	jobp: SVD_Jacobi_Perturb = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(m + 3 * n), "iwork array too small")
	assert(len(work_stat) >= 7, "work_stat array too small (need at least 7)")

	joba_c := cast(u8)accuracy
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobr_c := cast(u8)jobr
	jobt_c := cast(u8)jobt
	jobp_c := cast(u8)jobp

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}
	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect")
		v_ptr = raw_data(V.data)
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgejsv_(&joba_c, &jobu_c, &jobv_c, &jobr_c, &jobt_c, &jobp_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, raw_data(work), &lwork, raw_data(iwork), &info)

		copy(work_stat, work[:7])

	} else when T == f64 {
		lapack.dgejsv_(&joba_c, &jobu_c, &jobv_c, &jobr_c, &jobt_c, &jobp_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, raw_data(work), &lwork, raw_data(iwork), &info)

		copy(work_stat, work[:7])
	}

	return info, info == 0
}

// Combined complex64/complex128 Jacobi SVD (pre-allocated arrays)
dns_svd_jacobi_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated)
	U: ^Matrix(Cmplx) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(Cmplx) = nil, // Right singular vectors NOT transposed (pre-allocated, optional)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace for complex (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work_stat: []Real, // Statistics array (pre-allocated, size 7)
	jobu: SVD_Jacobi_Job = .Compute,
	jobv: SVD_Jacobi_Job = .Vectors,
	accuracy: SVD_Jacobi_Accuracy = .High,
	jobr: SVD_Jacobi_Range = .None,
	jobt: SVD_Jacobi_Transpose = .None,
	jobp: SVD_Jacobi_Perturb = .None,
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(m + 3 * n), "iwork array too small")
	assert(len(work_stat) >= 7, "work_stat array too small (need at least 7)")

	joba_c := cast(u8)accuracy
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobr_c := cast(u8)jobr
	jobt_c := cast(u8)jobt
	jobp_c := cast(u8)jobp

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}
	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect")
		v_ptr = raw_data(V.data)
	}

	lwork := Blas_Int(len(work))
	assert(len(rwork) > 0, "rwork array must be provided for complex types")
	lrwork := Blas_Int(len(rwork))

	when T == complex64 {
		lapack.cgejsv_(&joba_c, &jobu_c, &jobv_c, &jobr_c, &jobt_c, &jobp_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)

		// Extract statistics from rwork array
		copy(work_stat, rwork[:7])
	} else when T == complex128 {
		lapack.zgejsv_(&joba_c, &jobu_c, &jobv_c, &jobr_c, &jobt_c, &jobp_c, &m, &n, raw_data(A.data), &lda, raw_data(S), u_ptr, &ldu, v_ptr, &ldv, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)

		// Extract statistics from rwork array
		copy(work_stat, rwork[:7])
	}

	return info, info == 0
}


// ===================================================================================
// JACOBI SVD VARIANT (GESVJ) - WORKSPACE QUERIES
// ===================================================================================

// Query workspace size for Jacobi SVD variant (gesvj)
// rwork_size: For complex types only
query_workspace_dns_svd_jacobi_variant :: proc(A: ^Matrix($T), jobu: SVD_Jacobi_Variant_Job = .Compute, jobv: SVD_Jacobi_Variant_Job = .Vectors, joba: SVD_Jacobi_Variant_Matrix = .General) -> (work_size: int, rwork_size: int, info: Info) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Job parameters
	joba_c := cast(u8)joba
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv

	// Determine V dimensions
	mv := n
	if jobu != .None && m < n {
		mv = m
	}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	lwork: Blas_Int = -1

	// Dummy arrays for queries
	dummy_s := [1]T{}
	work_query: T
	rwork_size = 0

	when T == f32 {
		lapack.sgesvj_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dgesvj_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&info,
		)
		work_size = int(work_query)
	} else when T == complex64 {
		rwork_size = max(6, int(m + n)) // Real workspace for complex variant
		lapack.cgesvj_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			nil, // rwork dummy
			&info,
		)
		work_size = int(real(work_query))
	} else when T == complex128 {
		rwork_size = max(6, int(m + n)) // Real workspace for complex variant
		lapack.zgesvj_(
			&joba_c,
			&jobu_c,
			&jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			nil, // rwork dummy
			&info,
		)
		work_size = int(real(work_query))
	}

	return work_size, rwork_size, info
}

// Query result sizes for Jacobi SVD variant
query_result_sizes_dns_svd_jacobi_variant :: proc(A: ^Matrix($T), jobu: SVD_Jacobi_Variant_Job = .Compute, jobv: SVD_Jacobi_Variant_Job = .Vectors) -> (s_size: int, u_rows: int, u_cols: int, v_rows: int, v_cols: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)

	s_size = n

	if jobu != .None {
		u_rows = m
		u_cols = n
	}

	if jobv != .None {
		v_rows = n
		if jobu != .None && m < n {
			v_rows = m
		}
		v_cols = n
	}

	return s_size, u_rows, u_cols, v_rows, v_cols
}

// Compute SVD using Jacobi method variant (gesvj)
// Computes the SVD directly with V instead of V^T
// Good for matrices with well-conditioned columns
// Combined f32/f64 real Jacobi variant SVD (pre-allocated arrays)
dns_svd_jacobi_variant_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []T, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	jobu: SVD_Jacobi_Variant_Job = .Compute,
	jobv: SVD_Jacobi_Variant_Job = .Vectors,
	joba: SVD_Jacobi_Variant_Matrix = .General,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(S) >= int(n), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	joba_c := cast(u8)joba
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv

	mv := n // Number of rows for V
	if jobu != .None && m < n {
		mv = m // If U is computed and m < n, V has m rows
	}

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == n, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}
	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == mv && V.cols == n, "V matrix dimensions incorrect")
		v_ptr = raw_data(V.data)
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgesvj_(&joba_c, &jobu_c, &jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), &mv, v_ptr, &ldv, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgesvj_(&joba_c, &jobu_c, &jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), &mv, v_ptr, &ldv, raw_data(work), &lwork, &info)
	}

	if jobu != .None && U != nil {
		copy(U.data, A.data[:m * n])
	}

	return info, info == 0
}

// Combined complex64/complex128 Jacobi variant SVD (pre-allocated arrays)
dns_svd_jacobi_variant_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten)
	S: []$Real, // Singular values (pre-allocated)
	U: ^Matrix(Cmplx) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(Cmplx) = nil, // Right singular vectors (pre-allocated, optional)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace for complex (pre-allocated)
	jobu: SVD_Jacobi_Variant_Job = .Compute,
	jobv: SVD_Jacobi_Variant_Job = .Vectors,
	joba: SVD_Jacobi_Variant_Matrix = .General,
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(S) >= int(n), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	joba_c := cast(u8)joba
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv

	mv := n
	if jobu != .None && m < n {
		mv = m
	}

	ldu: Blas_Int = 1
	u_ptr: ^Cmplx = nil
	if jobu != .None && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == n, "U matrix dimensions incorrect")
		u_ptr = raw_data(U.data)
	}
	ldv: Blas_Int = 1
	v_ptr: ^Cmplx = nil
	if jobv != .None && V != nil {
		ldv = V.ld
		assert(V.rows == mv && V.cols == n, "V matrix dimensions incorrect")
		v_ptr = raw_data(V.data)
	}

	lwork := Blas_Int(len(work))
	assert(len(rwork) > 0, "rwork array must be provided for complex types")
	lrwork := Blas_Int(len(rwork))

	when T == complex64 {
		lapack.cgesvj_(&joba_c, &jobu_c, &jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), &mv, v_ptr, &ldv, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	} else when T == complex128 {
		lapack.zgesvj_(&joba_c, &jobu_c, &jobv_c, &m, &n, raw_data(A.data), &lda, raw_data(S), &mv, v_ptr, &ldv, raw_data(work), &lwork, raw_data(rwork), &lrwork, &info)
	}

	if jobu != .None && U != nil {
		copy(U.data, A.data[:m * n])
	}

	return info, info == 0
}

// ===================================================================================
// GENERALIZED SINGULAR VALUE DECOMPOSITION (GSVD)
// ===================================================================================
// Merged from svd_generalized.odin - All functions work with general dense matrices
//
// GSVD computes the generalized SVD of two matrices A (m×n) and B (p×n):
//   A = U * D1 * [0 R] * Q^T
//   B = V * D2 * [0 R] * Q^T
// where D1 and D2 are diagonal matrices with D1^2 + D2^2 = I
//
// Available algorithms:
// - Standard GSVD: sggsvd/dggsvd/cggsvd/zggsvd
// - Blocked GSVD: sggsvd3/dggsvd3/cggsvd3/zggsvd3 (improved performance)
// - Preprocessing: sggsvp/dggsvp/cggsvp/zggsvp (reduce to standard form)
// - Blocked preprocessing: sggsvp3/dggsvp3/cggsvp3/zggsvp3
// ===================================================================================

// Job options for Generalized SVD matrix computation
GSVD_Job :: enum u8 {
	Compute = 'U', // Compute matrix (U for left, V for middle, Q for right)
	Full    = 'V', // Alternative name for compute (used for V)
	Right   = 'Q', // Alternative name for compute (used for Q)
	None    = 'N', // Do not compute matrix
}

// Query workspace size for blocked generalized SVD
query_workspace_dns_svd_generalized_blocked :: proc(A: ^Matrix($T), B: ^Matrix(T), jobu := GSVD_Job.Compute, jobv := GSVD_Job.Full, jobq := GSVD_Job.Right) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	m_int := A.rows
	n_int := A.cols
	p_int := B.rows

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	// Integer workspace is always n
	iwork_size = int(n_int)

	// Real workspace depends on type
	when is_float(T) {
		rwork_size = 0 // No real workspace for real types
	} else {
		rwork_size = 2 * int(n_int) // Complex types need real workspace
	}

	// Query for optimal workspace
	lwork := QUERY_WORKSPACE
	work_query: T
	k, l: Blas_Int
	info: Info

	when T == f32 {
		lapack.sggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, &info)
	} else when T == f64 {
		lapack.dggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, &info)
	} else when T == complex64 {
		lapack.cggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, nil, &info)
	} else when T == complex128 {
		lapack.zggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, nil, &info)
	}

	// Convert work query result
	when is_float(T) {
		work_size = int(work_query)
	} else {
		work_size = int(real(work_query))
	}

	return
}

// Blocked/improved version of generalized SVD
dns_svd_generalized_blocked :: proc {
	dns_svd_generalized_blocked_real,
	dns_svd_generalized_blocked_complex,
}

dns_svd_generalized_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []T, // Pre-allocated output singular values (size n)
	beta: []T, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) > 0, "workspace required")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), &lwork, raw_data(iwork), &info)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD (complex64/complex128)
dns_svd_generalized_blocked_complex :: proc(
	A: ^Matrix($Cmplx),
	B: ^Matrix(Cmplx),
	alpha: []$Real, // Pre-allocated output singular values (size n)
	beta: []Real, // Pre-allocated output singular values (size n)
	U: ^Matrix(Cmplx) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(Cmplx) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(Cmplx) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) > 0, "workspace required")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.cggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	} else when T == complex128 {
		lapack.zggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), &info)
	}

	return k, l, info, info == 0
}


// ===================================================================================
// GENERALIZED SINGULAR VALUE DECOMPOSITION
// ===================================================================================
// Query workspace size for standard generalized SVD
query_workspace_dns_svd_generalized :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	p := int(B.rows)

	// Integer workspace is always n
	iwork_size = n

	// Workspace estimates for non-blocked algorithm
	when is_float(T) {
		work_size = max(3 * n, m, p) + n
		rwork_size = 0 // No real workspace for real types
	} else {
		work_size = max(3 * n, m, p) + n
		rwork_size = 2 * n // Complex types need real workspace
	}

	return
}

// Compute generalized SVD: U^H*A*Q = D1*[0 R], V^H*B*Q = D2*[0 R]
dns_svd_generalized :: proc {
	dns_svd_generalized_real,
	dns_svd_generalized_complex,
}

dns_svd_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []T, // Pre-allocated output singular values (size n)
	beta: []T, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	p := B.rows

	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)) + int(n), "workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == f32 {
		lapack.sggsvd_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dggsvd_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), raw_data(iwork), &info)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD (complex64/complex128)
dns_svd_generalized_complex :: proc(
	A: ^Matrix($Cmplx),
	B: ^Matrix(Cmplx),
	alpha: []$Real, // Pre-allocated output singular values (size n)
	beta: []Real, // Pre-allocated output singular values (size n)
	U: ^Matrix(Cmplx) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(Cmplx) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(Cmplx) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	p := B.rows

	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)) + int(n), "workspace too small")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == complex64 {
		lapack.cggsvd_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), raw_data(rwork), raw_data(iwork), &info)
	} else when T == complex128 {
		lapack.zggsvd_(&jobu_c, &jobv_c, &jobq_c, &m, &n, &p, &k, &l, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(alpha), raw_data(beta), u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(work), raw_data(rwork), raw_data(iwork), &info)
	}

	return k, l, info, info == 0
}


// ===================================================================================
// GENERALIZED SVD PREPROCESSING
// ===================================================================================

// Query workspace size for generalized SVD preprocessing
query_workspace_dns_svd_generalized_preprocess :: proc(A: ^Matrix($T), B: ^Matrix(T)) -> (work_size: int, rwork_size: int, iwork_size: int, tau_size: int) where is_float(T) || is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	p := int(B.rows)

	tau_size = n
	iwork_size = n

	when is_float(T) {
		work_size = max(3 * n, m, p)
		rwork_size = 0 // No real workspace for real types
	} else {
		work_size = max(3 * n, m, p)
		rwork_size = 2 * n // Complex types need real workspace
	}

	return
}

// Preprocessing for generalized SVD - reduce to standard form
dns_svd_generalized_preprocess :: proc {
	dns_svd_generalized_preprocess_real,
	dns_svd_generalized_preprocess_complex,
}

dns_svd_generalized_preprocess_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T,
	tolb: T,
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []T, // Pre-allocated tau array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)), "workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == f32 {
		lapack.sggsvp_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), raw_data(work), &info)
	} else when T == f64 {
		lapack.dggsvp_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), raw_data(work), &info)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD preprocessing
dns_svd_generalized_preprocess_complex :: proc(
	A: ^Matrix($Cmplx),
	B: ^Matrix(Cmplx),
	tola: $Real,
	tolb: Real,
	U: ^Matrix(Cmplx) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(Cmplx) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(Cmplx) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []Cmplx, // Pre-allocated tau array (size n)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)), "workspace too small")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == complex64 {
		lapack.cggsvp_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(rwork), raw_data(tau), raw_data(work), &info)
	} else when T == complex128 {
		lapack.zggsvp_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(rwork), raw_data(tau), raw_data(work), &info)
	}

	return k, l, info, info == 0
}


// Query workspace size for blocked generalized SVD preprocessing
query_workspace_dns_svd_generalized_preprocess_blocked :: proc(A: ^Matrix($T), B: ^Matrix(T), jobu := GSVD_Job.Compute, jobv := GSVD_Job.Full, jobq := GSVD_Job.Right) -> (work_size: int, rwork_size: int, iwork_size: int, tau_size: int) where is_float(T) || is_complex(T) {
	m_int := A.rows
	p_int := B.rows
	n_int := A.cols

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tau_size = int(n_int)
	iwork_size = int(n_int)

	when is_float(T) {
		rwork_size = 0 // No real workspace for real types
	} else {
		rwork_size = 2 * int(n_int)
	}

	lwork := QUERY_WORKSPACE
	work_query: T
	k, l: Blas_Int
	info: Info
	tola_default := T(1e-8) when is_float(T) else 1e-8
	tolb_default := T(1e-8) when is_float(T) else 1e-8

	when T == f32 {
		lapack.sggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, &work_query, &lwork, &info)
	} else when T == f64 {
		lapack.dggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, &work_query, &lwork, &info)
	} else when T == complex64 {
		lapack.cggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, nil, &work_query, &lwork, &info)
	} else when T == complex128 {
		lapack.zggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, nil, &work_query, &lwork, &info)
	}

	// Convert work query result
	when is_float(T) {
		work_size = int(work_query)
	} else {
		work_size = int(real(work_query))
	}

	return
}

// Improved/blocked version of generalized SVD preprocessing
dns_svd_generalized_preprocess_blocked :: proc {
	dns_svd_generalized_preprocess_blocked_real,
	dns_svd_generalized_preprocess_blocked_complex,
}

dns_svd_generalized_preprocess_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T,
	tolb: T,
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []T, // Pre-allocated tau array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^Cmplx = nil
	v_ptr: ^Cmplx = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))
	if lwork == 0 {
		lwork = QUERY_WORKSPACE
		work_query: T

		when T == f32 {
			lapack.sggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), &work_query, &lwork, &info)
		} else when T == f64 {
			lapack.dggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), &work_query, &lwork, &info)
		}
		// Return with required workspace size in k
		k = Blas_Int(work_query)
		return k, 0, -1, false
	}

	// Perform preprocessing
	when T == f32 {
		lapack.sggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return k, l, info, info == 0
}

dns_svd_generalized_preprocess_blocked_complex :: proc(
	A: ^Matrix($Cmplx),
	B: ^Matrix(Cmplx),
	tola: $Real,
	tolb: Real,
	U: ^Matrix(Cmplx) = nil,
	V: ^Matrix(Cmplx) = nil,
	Q: ^Matrix(Cmplx) = nil,
	tau: []Cmplx,
	work: []Cmplx,
	rwork: []Real,
	iwork: []Blas_Int,
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) ||
	(Cmplx == complex128 && Real == f64) {
	m := A.rows
	p := A.cols
	n := B.cols

	lda := A.ld
	ldb := B.ld

	// Set job parameters
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? (T == complex64 ? R(1.2e-7) : R(2.2e-16)) : tola
	tolb_actual := tolb == 0 ? (T == complex64 ? R(1.2e-7) : R(2.2e-16)) : tolb

	// Matrix leading dimensions
	ldu := U != nil ? U.ld : Blas_Int(1)
	ldv := V != nil ? V.ld : Blas_Int(1)
	ldq := Q != nil ? Q.ld : Blas_Int(1)

	// Matrix data pointers
	u_ptr := U != nil ? raw_data(U.data) : nil
	v_ptr := V != nil ? raw_data(V.data) : nil
	q_ptr := Q != nil ? raw_data(Q.data) : nil

	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.cggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(rwork), raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m, &p, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, &tola_actual, &tolb_actual, &k, &l, u_ptr, &ldu, v_ptr, &ldv, q_ptr, &ldq, raw_data(iwork), raw_data(rwork), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return k, l, info, info == 0
}
