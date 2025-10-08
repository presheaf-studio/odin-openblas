package openblas

import lapack "./f77"

// Job options for equilibration
Equilibration_Job :: enum u8 {
	None   = 'N', // No equilibration
	Row    = 'R', // Row scaling only
	Column = 'C', // Column scaling only
	Both   = 'B', // Both row and column scaling
}

// Job options for condition number estimate
Condition_Norm :: enum u8 {
	One      = '1', // 1-norm condition number
	Infinity = 'I', // Infinity-norm condition number
}

// Job options for factorization in expert drivers
Factorization_Job :: enum u8 {
	Compute            = 'N', // Factor the matrix A
	Factored           = 'F', // A is already factored
	Equilibrate_Factor = 'E', // Equilibrate then factor
}

// ===================================================================================
// BASIC LINEAR SOLVERS (GESV family)
// ===================================================================================

dns_solve_expert :: proc {
	dns_solve_expert_real,
	dns_solve_expert_complex,
}

// Basic linear solve: AX = B using LU factorization
// A is overwritten with L and U factors
// B is overwritten with solution X
dns_solve :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with LU factors)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated, size min(m,n))
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")
	assert(len(ipiv) >= int(n), "ipiv array too small")

	when T == f32 {
		lapack.sgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT LINEAR SOLVERS (GESVX family)
// ===================================================================================

// Query workspace size for expert linear solver
query_workspace_dns_solve_expert :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when is_float(T) {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when is_complex(T) {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Expert linear solver with equilibration, condition estimation, and iterative refinement
dns_solve_expert_real :: proc(
	A: ^Matrix($T), // Input matrix (preserved if AF provided)
	AF: ^Matrix(T), // Factored matrix (pre-allocated, optional)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated if factored)
	B: ^Matrix(T), // RHS matrix (preserved)
	X: ^Matrix(T), // Solution matrix (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	fact: Factorization_Job = .Compute,
	trans: TransposeMode = .None,
	equed: Equilibration_Job = .None, // Input/output equilibration type
	R: []T = nil, // Row scaling factors (pre-allocated if equed)
	C: []T = nil, // Column scaling factors (pre-allocated if equed)
) -> (
	rcond: T,
	ferr: []T,
	berr: []T,
	info: Info,
	ok: bool, // Reciprocal condition number estimate// Forward error bounds (pre-allocated)// Backward error bounds (pre-allocated)
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF != nil ? AF.ld : A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "ferr array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	fact_c := cast(char)fact
	trans_c := cast(char)trans
	equed_c := cast(char)equed

	af_ptr := AF != nil ? raw_data(AF.data) : raw_data(A.data)
	ipiv_ptr := len(ipiv) > 0 ? raw_data(ipiv) : nil
	r_ptr := len(R) > 0 ? raw_data(R) : nil
	c_ptr := len(C) > 0 ? raw_data(C) : nil

	when T == f32 {
		lapack.sgesvx_(&fact_c, &trans_c, &n, &nrhs, raw_data(A.data), &lda, af_ptr, &ldaf, ipiv_ptr, &equed_c, r_ptr, c_ptr, raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgesvx_(&fact_c, &trans_c, &n, &nrhs, raw_data(A.data), &lda, af_ptr, &ldaf, ipiv_ptr, &equed_c, r_ptr, c_ptr, raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return rcond, ferr, berr, info, info == 0
}

dns_solve_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (preserved if AF provided)
	AF: ^Matrix(Cmplx), // Factored matrix (pre-allocated, optional)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated if factored)
	B: ^Matrix(Cmplx), // RHS matrix (preserved)
	X: ^Matrix(Cmplx), // Solution matrix (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []$Real, // Real workspace (pre-allocated)
	fact: Factorization_Job = .Compute,
	trans: TransposeMode = .None,
	equed: Equilibration_Job = .None, // Input/output equilibration type
	R: []Real = nil, // Row scaling factors (pre-allocated if equed)
	C: []Real = nil, // Column scaling factors (pre-allocated if equed)
) -> (
	rcond: Real,
	ferr: []Real,
	berr: []Real,
	info: Info,
	ok: bool, // Reciprocal condition number estimate// Forward error bounds (pre-allocated)// Backward error bounds (pre-allocated)
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF != nil ? AF.ld : A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "ferr array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	fact_c := cast(char)fact
	trans_c := cast(char)trans
	equed_c := cast(char)equed

	af_ptr := AF != nil ? raw_data(AF.data) : raw_data(A.data)
	ipiv_ptr := len(ipiv) > 0 ? raw_data(ipiv) : nil
	r_ptr := len(R) > 0 ? raw_data(R) : nil
	c_ptr := len(C) > 0 ? raw_data(C) : nil

	when Cmplx == complex64 {
		lapack.cgesvx_(&fact_c, &trans_c, &n, &nrhs, raw_data(A.data), &lda, af_ptr, &ldaf, ipiv_ptr, &equed_c, r_ptr, c_ptr, raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgesvx_(&fact_c, &trans_c, &n, &nrhs, raw_data(A.data), &lda, af_ptr, &ldaf, ipiv_ptr, &equed_c, r_ptr, c_ptr, raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return rcond, ferr, berr, info, info == 0
}

// ===================================================================================
// LU FACTORIZATION (GETRF/GETRS family)
// ===================================================================================
// LU factorization: A = P*L*U
dns_lu_factorize :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with L and U)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated, size min(m,n))
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(ipiv) >= int(min(m, n)), "ipiv array too small")

	when T == f32 {
		lapack.sgetrf_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgetrf_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.cgetrf_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zgetrf_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	}

	return info, info == 0
}

// Solve linear system using LU factors: A*X = B or A^T*X = B
dns_lu_solve :: proc(
	A: ^Matrix($T), // LU factors from dns_lu_factorize
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")
	assert(len(ipiv) >= int(n), "ipiv array too small")

	trans_c := cast(char)trans

	when T == f32 {
		lapack.sgetrs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dgetrs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cgetrs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zgetrs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// Query workspace size for matrix inversion
query_workspace_dns_lu_invert :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.rows
	lda := A.ld

	lwork: Blas_Int = QUERY_WORKSPACE
	work_query: T
	info: Info

	when T == f32 {
		lapack.sgetri_(&n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dgetri_(&n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == complex64 {
		lapack.cgetri_(&n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when T == complex128 {
		lapack.zgetri_(&n, nil, &lda, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Matrix inversion using LU factors
dns_lu_invert :: proc(
	A: ^Matrix($T), // LU factors (overwritten with inverse)
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	work: []T, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	lda := A.ld
	lwork := Blas_Int(len(work))

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.sgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION (GECON)
// ===================================================================================

dns_condition :: proc {
	dns_condition_real,
	dns_condition_complex,
}

// Query workspace size for condition number estimation
query_workspace_dns_condition :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when is_float(T) {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when is_complex(T) {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Estimate condition number of LU-factored matrix
dns_condition_real :: proc(
	A: ^Matrix($T), // LU factors from dns_lu_factorize
	anorm: T, // 1-norm or infinity-norm of original matrix
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	norm: Condition_Norm = .One,
) -> (
	rcond: T,
	info: Info,
	ok: bool, // Reciprocal condition number
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	norm_c := cast(u8)norm

	when T == f32 {
		lapack.sgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

dns_condition_complex :: proc(
	A: ^Matrix($Cmplx), // LU factors from dns_lu_factorize
	anorm: $Real, // 1-norm or infinity-norm of original matrix
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	norm: Condition_Norm = .One,
) -> (
	rcond: Real,
	info: Info,
	ok: bool, // Reciprocal condition number
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	norm_c := cast(u8)norm

	when Cmplx == complex64 {
		lapack.cgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (GERFS family)
// ===================================================================================
// Note: Equilibration functions (dns_equilibrate, dns_equilibrate_improved)
// are in auxiliary.odin

dns_solve_refine :: proc {
	dns_solve_refine_real,
	dns_solve_refine_complex,
}

// Query workspace size for iterative refinement
query_workspace_dns_solve_refine :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when is_float(T) {
		work_size = int(3 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when is_complex(T) {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Iterative refinement for linear systems
dns_solve_refine_real :: proc(
	A: ^Matrix($T), // Original matrix A
	AF: ^Matrix(T), // LU factorization from dns_lu_factorize
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	B: ^Matrix(T), // Right-hand side (preserved)
	X: ^Matrix(T), // Solution (input/output)
	ferr: []T, // Forward error bounds (pre-allocated, size nrhs)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols && AF.rows == n, "AF must match A dimensions")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "ferr array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(work) >= int(3 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")
	assert(len(ipiv) >= int(n), "ipiv array too small")

	trans_c := cast(u8)trans

	when T == f32 {
		lapack.sgerfs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgerfs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

dns_solve_refine_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix A
	AF: ^Matrix(Cmplx), // LU factorization from dns_lu_factorize
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	B: ^Matrix(Cmplx), // Right-hand side (preserved)
	X: ^Matrix(Cmplx), // Solution (input/output)
	ferr: []$Real, // Forward error bounds (pre-allocated, size nrhs)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	trans: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols && AF.rows == n, "AF must match A dimensions")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "ferr array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")
	assert(len(ipiv) >= int(n), "ipiv array too small")

	trans_c := cast(u8)trans

	when Cmplx == complex64 {
		lapack.cgerfs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgerfs_(&trans_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT (DSGESV/ZCGESV family)
// ===================================================================================

dns_solve_mixed_precision :: proc {
	dns_solve_mixed_precision_f64,
	dns_solve_mixed_precision_c128,
}

// Query workspace size for mixed precision solver
query_workspace_dns_solve_mixed_precision :: proc($T: typeid, n: int, nrhs: int) -> (work_size: int, swork_size: int, rwork_size: int) where T == f64 || T == complex128 {
	when T == f64 {
		work_size = int(n * nrhs)
		swork_size = int(n * (n + nrhs))
		rwork_size = 0
	} else when T == complex128 {
		work_size = int(n * nrhs)
		swork_size = int(n * (n + nrhs))
		rwork_size = int(n)
	}
	return work_size, swork_size, rwork_size
}

// Mixed precision solver: uses f32 factorization for f64 system
dns_solve_mixed_precision_f64 :: proc(
	A: ^Matrix(f64), // Input matrix (overwritten with LU factors)
	B: ^Matrix(f64), // RHS matrix (overwritten with solution)
	X: ^Matrix(f64), // Solution matrix (pre-allocated)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []f64, // Workspace (pre-allocated)
	swork: []f32, // Single precision workspace (pre-allocated)
	iter: ^Blas_Int, // Iteration count (output)
) -> (
	info: Info,
	ok: bool,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(n * nrhs), "work array too small")
	assert(len(swork) >= int(n * (n + nrhs)), "swork array too small")

	lapack.dsgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(work), raw_data(swork), iter, &info)

	return info, info == 0
}

// Mixed precision solver: uses complex64 factorization for complex128 system
dns_solve_mixed_precision_c128 :: proc(
	A: ^Matrix(complex128), // Input matrix (overwritten with LU factors)
	B: ^Matrix(complex128), // RHS matrix (overwritten with solution)
	X: ^Matrix(complex128), // Solution matrix (pre-allocated)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []complex128, // Workspace (pre-allocated)
	swork: []complex64, // Single precision workspace (pre-allocated)
	rwork: []f64, // Real workspace (pre-allocated)
	iter: ^Blas_Int, // Iteration count (output)
) -> (
	info: Info,
	ok: bool,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ipiv) >= int(n), "ipiv array too small")
	assert(len(work) >= int(n * nrhs), "work array too small")
	assert(len(swork) >= int(n * (n + nrhs)), "swork array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	lapack.zcgesv_(&n, &nrhs, raw_data(A.data), &lda, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(work), raw_data(swork), raw_data(rwork), iter, &info)

	return info, info == 0
}

// ===================================================================================
// RECURSIVE LU FACTORIZATION (GETRF2 family)
// ===================================================================================

dns_lu_factorize_recursive :: dns_lu_factorize_recursive_generic

// Recursive LU factorization (more efficient for tall matrices)
dns_lu_factorize_recursive_generic :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with L and U)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated, size min(m,n))
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(ipiv) >= int(min(m, n)), "ipiv array too small")

	when T == f32 {
		lapack.sgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.cgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXTENDED EXPERT LINEAR SOLVERS WITH COMPREHENSIVE ERROR BOUNDS (GESVXX/GERFSX family)
// ===================================================================================

dns_solve_refine_extended :: proc {
	dns_solve_refine_extended_real,
	dns_solve_refine_extended_complex,
}

dns_solve_expert_extended :: proc {
	dns_solve_expert_extended_real,
	dns_solve_expert_extended_complex,
}

// Query workspace size for extended iterative refinement
query_workspace_dns_solve_refine_extended :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when is_complex(T) {
		work_size = int(2 * n)
		rwork_size = int(3 * n)
		iwork_size = 0
	}
	return work_size, rwork_size, iwork_size
}

// Extended iterative refinement with comprehensive error bounds (GERFSX)
dns_solve_refine_extended_real :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // LU factorization from dns_lu_factorize
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	R: []T, // Row scaling factors (pre-allocated, size n)
	C: []T, // Column scaling factors (pre-allocated, size n)
	B: ^Matrix(T), // Right-hand side (preserved)
	X: ^Matrix(T), // Solution (input/output)
	trans: TransposeMode = .None,
	equed: Equilibration_Job = .None,
	n_err_bnds: int = 3, // Number of error bounds to compute
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []T, // Normwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	err_bnds_comp: []T, // Componentwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols && AF.rows == n, "AF must match A dimensions")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(R) >= int(n), "R array too small")
	assert(len(C) >= int(n), "C array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "err_bnds_norm array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "err_bnds_comp array too small")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	trans_c := cast(u8)trans
	equed_c := cast(u8)equed
	n_err_bnds_int := Blas_Int(n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: T = 0

	when T == f32 {
		lapack.sgerfsx_(
			&trans_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dgerfsx_(
			&trans_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

dns_solve_refine_extended_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // LU factorization from dns_lu_factorize
	ipiv: []Blas_Int, // Pivot indices from dns_lu_factorize
	R: []$Real, // Row scaling factors (pre-allocated, size n)
	C: []Real, // Column scaling factors (pre-allocated, size n)
	B: ^Matrix(Cmplx), // Right-hand side (preserved)
	X: ^Matrix(Cmplx), // Solution (input/output)
	trans: TransposeMode = .None,
	equed: Equilibration_Job = .None,
	n_err_bnds: int = 3, // Number of error bounds to compute
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []Real, // Normwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	err_bnds_comp: []Real, // Componentwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols && AF.rows == n, "AF must match A dimensions")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(R) >= int(n), "R array too small")
	assert(len(C) >= int(n), "C array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "err_bnds_norm array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "err_bnds_comp array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(3 * n), "rwork array too small")

	trans_c := cast(u8)trans
	equed_c := cast(u8)equed
	n_err_bnds_int := Blas_Int(n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: Real = 0

	when Cmplx == complex64 {
		lapack.cgerfsx_(
			&trans_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zgerfsx_(
			&trans_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// Query workspace size for expert extended solver
query_workspace_dns_solve_expert_extended :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when is_complex(T) {
		work_size = int(2 * n)
		rwork_size = int(3 * n)
		iwork_size = 0
	}
	return work_size, rwork_size, iwork_size
}

// Expert solver with extended error bounds (GESVXX)
dns_solve_expert_extended_real :: proc(
	A: ^Matrix($T), // Input matrix (preserved if AF provided)
	AF: ^Matrix(T), // Factored matrix (pre-allocated, optional)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated if factored)
	R: []T, // Row scaling factors (pre-allocated, size n)
	C: []T, // Column scaling factors (pre-allocated, size n)
	B: ^Matrix(T), // RHS matrix (preserved)
	X: ^Matrix(T), // Solution matrix (pre-allocated)
	fact: Factorization_Job = .Compute,
	trans: TransposeMode = .None,
	equed: ^Equilibration_Job, // Input/output equilibration type
	n_err_bnds: int = 3, // Number of error bounds to compute
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []T, // Normwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	err_bnds_comp: []T, // Componentwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
) -> (
	rcond: T,
	rpvgrw: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF != nil ? AF.ld : A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(R) >= int(n), "R array too small")
	assert(len(C) >= int(n), "C array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "err_bnds_norm array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "err_bnds_comp array too small")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans
	equed_c := cast(u8)(equed^)
	n_err_bnds_int := Blas_Int(n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: T = 0

	af_ptr := AF != nil ? raw_data(AF.data) : raw_data(A.data)

	when T == f32 {
		lapack.sgesvxx_(
			&fact_c,
			&trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			af_ptr,
			&ldaf,
			raw_data(ipiv),
			&equed_c,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dgesvxx_(
			&fact_c,
			&trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			af_ptr,
			&ldaf,
			raw_data(ipiv),
			&equed_c,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	equed^ = cast(Equilibration_Job)(equed_c)
	return rcond, rpvgrw, info, info == 0
}

dns_solve_expert_extended_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (preserved if AF provided)
	AF: ^Matrix(Cmplx), // Factored matrix (pre-allocated, optional)
	ipiv: []Blas_Int, // Pivot indices (pre-allocated if factored)
	R: []$Real, // Row scaling factors (pre-allocated, size n)
	C: []Real, // Column scaling factors (pre-allocated, size n)
	B: ^Matrix(Cmplx), // RHS matrix (preserved)
	X: ^Matrix(Cmplx), // Solution matrix (pre-allocated)
	fact: Factorization_Job = .Compute,
	trans: TransposeMode = .None,
	equed: ^Equilibration_Job, // Input/output equilibration type
	n_err_bnds: int = 3, // Number of error bounds to compute
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []Real, // Normwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	err_bnds_comp: []Real, // Componentwise error bounds (pre-allocated, size nrhs * n_err_bnds)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
) -> (
	rcond: Real,
	rpvgrw: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF != nil ? AF.ld : A.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n && X.rows == n, "B and X must have same number of rows as A")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(R) >= int(n), "R array too small")
	assert(len(C) >= int(n), "C array too small")
	assert(len(berr) >= int(nrhs), "berr array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "err_bnds_norm array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "err_bnds_comp array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(3 * n), "rwork array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans
	equed_c := cast(u8)(equed^)
	n_err_bnds_int := Blas_Int(n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: Real = 0

	af_ptr := AF != nil ? raw_data(AF.data) : raw_data(A.data)

	when Cmplx == complex64 {
		lapack.cgesvxx_(
			&fact_c,
			&trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			af_ptr,
			&ldaf,
			raw_data(ipiv),
			&equed_c,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zgesvxx_(
			&fact_c,
			&trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			af_ptr,
			&ldaf,
			raw_data(ipiv),
			&equed_c,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	equed^ = cast(Equilibration_Job)(equed_c)
	return rcond, rpvgrw, info, info == 0
}
