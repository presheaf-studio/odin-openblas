package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE CHOLESKY FACTORIZATION (POTRF family)
// ===================================================================================

cholesky_factorize :: proc {
	cholesky_factorize_real,
	cholesky_factorize_complex,
}

cholesky_solve :: proc {
	cholesky_solve_real,
	cholesky_solve_complex,
}

cholesky_invert :: proc {
	cholesky_invert_real,
	cholesky_invert_complex,
}

// Cholesky factorization: A = L*L^T (lower) or A = U^T*U (upper)
cholesky_factorize_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with Cholesky factor)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == f64 {
		lapack.dpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

cholesky_factorize_complex :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with Cholesky factor)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")

	uplo_c := cast(u8)uplo

	when T == complex64 {
		lapack.cpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex128 {
		lapack.zpotrf_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CHOLESKY SOLVE (POTRS family)
// ===================================================================================

// Solve linear system using Cholesky factors: A*X = B
cholesky_solve_real :: proc(
	A: ^Matrix($T), // Cholesky factors from cholesky_factorize
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spotrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpotrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

cholesky_solve_complex :: proc(
	A: ^Matrix($T), // Cholesky factors from cholesky_factorize
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")

	uplo_c := cast(u8)uplo

	when T == complex64 {
		lapack.cpotrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpotrs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CHOLESKY MATRIX INVERSION (POTRI family)
// ===================================================================================

// Compute inverse of positive definite matrix using Cholesky factors
cholesky_invert_real :: proc(
	A: ^Matrix($T), // Cholesky factors (overwritten with inverse)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spotri_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == f64 {
		lapack.dpotri_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

cholesky_invert_complex :: proc(
	A: ^Matrix($T), // Cholesky factors (overwritten with inverse)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")

	uplo_c := cast(u8)uplo

	when T == complex64 {
		lapack.cpotri_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	} else when T == complex128 {
		lapack.zpotri_(&uplo_c, &n, raw_data(A.data), &lda, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION (POCON family)
// ===================================================================================

cholesky_condition_number :: proc {
	cholesky_condition_number_real,
	cholesky_condition_number_complex,
}

// Query workspace size for condition number estimation
query_workspace_cholesky_condition_number :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when T == f32 || T == f64 {
		work_size = int(3 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Estimate condition number of Cholesky-factored matrix
cholesky_condition_number_real :: proc(
	A: ^Matrix($T), // Cholesky factors from cholesky_factorize
	anorm: T, // 1-norm of original matrix
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: T,
	info: Info,
	ok: bool, // Reciprocal condition number
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(work) >= int(3 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

cholesky_condition_number_complex :: proc(
	A: ^Matrix($Cmplx), // Cholesky factors from cholesky_factorize
	anorm: $Real, // 1-norm of original matrix
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	info: Info,
	ok: bool, // Reciprocal condition number
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpocon_(&uplo_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// EQUILIBRATION (POEQU family)
// ===================================================================================

cholesky_equilibrate :: proc {
	cholesky_equilibrate_real,
	cholesky_equilibrate_complex,
}

// Compute row and column scalings to equilibrate symmetric positive definite matrix
cholesky_equilibrate_real :: proc(
	A: ^Matrix($T), // Input matrix (not modified)
	S: []T, // Scaling factors (pre-allocated, size n)
) -> (
	scond: T,
	amax: T,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(S) >= int(n), "S array too small")

	when T == f32 {
		lapack.spoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == f64 {
		lapack.dpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

cholesky_equilibrate_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (not modified)
	S: []$Real, // Scaling factors (pre-allocated, size n)
) -> (
	scond: Real,
	amax: Real,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(S) >= int(n), "S array too small")

	when Cmplx == complex64 {
		lapack.cpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when Cmplx == complex128 {
		lapack.zpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE SOLVER (POSV family)
// ===================================================================================

cholesky_positive_definite_solve :: proc {
	cholesky_positive_definite_solve_real,
	cholesky_positive_definite_solve_complex,
}

// Solve positive definite linear system AX = B using Cholesky factorization
// This is a simple driver that combines factorization and solve steps
cholesky_positive_definite_solve_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with Cholesky factor)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.sposv_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dposv_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

cholesky_positive_definite_solve_complex :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with Cholesky factor)
	B: ^Matrix(T), // RHS matrix (overwritten with solution)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B matrix must have same number of rows as A")

	uplo_c := cast(u8)uplo

	when T == complex64 {
		lapack.cposv_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zposv_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE EXPERT SOLVER (POSVX family)
// ===================================================================================

cholesky_positive_definite_solve_expert :: proc {
	cholesky_positive_definite_solve_expert_real,
	cholesky_positive_definite_solve_expert_complex,
}

query_workspace_cholesky_positive_definite_solve_expert :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when T == f32 || T == f64 {
		work_size = int(3 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Expert driver for solving positive definite systems with equilibration and error bounds
cholesky_positive_definite_solve_expert_real :: proc(
	A: ^Matrix($T), // Input matrix (may be equilibrated)
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (pre-allocated)
	AF: ^Matrix(T), // Cholesky factorization (pre-allocated if fact='F')
	S: []T, // Scaling factors (pre-allocated, size n)
	ferr: []T, // Forward error bounds (pre-allocated, size nrhs)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	fact: Factorization_Type = .New,
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: T,
	equed: Equilibration_Type,
	info: Info,
	ok: bool,// Equilibration status
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(3 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo
	equed_c: u8 = 0

	when T == f32 {
		lapack.sposvx_(&fact_c, &uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, &equed_c, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dposvx_(&fact_c, &uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, &equed_c, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	equed = cast(Equilibration_Type)equed_c
	return rcond, equed, info, info == 0
}

cholesky_positive_definite_solve_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (may be equilibrated)
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution matrix (pre-allocated)
	AF: ^Matrix(Cmplx), // Cholesky factorization (pre-allocated if fact='F')
	S: []$Real, // Scaling factors (pre-allocated, size n)
	ferr: []Real, // Forward error bounds (pre-allocated, size nrhs)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	fact: Factorization_Type = .New,
	uplo: MatrixRegion = .Upper,
) -> (
	rcond: Real,
	equed: Equilibration_Type,
	info: Info,
	ok: bool,// Equilibration status
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo
	equed_c: u8 = 0

	when Cmplx == complex64 {
		lapack.cposvx_(&fact_c, &uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, &equed_c, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zposvx_(&fact_c, &uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, &equed_c, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	equed = cast(Equilibration_Type)equed_c
	return rcond, equed, info, info == 0
}

// ===================================================================================
// POSITIVE DEFINITE EXTRA-EXPERT SOLVER (POSVXX family)
// ===================================================================================

cholesky_positive_definite_solve_extra_expert :: proc {
	cholesky_positive_definite_solve_extra_expert_real,
	cholesky_positive_definite_solve_extra_expert_complex,
}

query_workspace_cholesky_positive_definite_solve_extra_expert :: proc(A: ^Matrix($T), n_err_bnds: int = 3) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when T == f32 || T == f64 {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		rwork_size = int(2 * n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Extra-precise iterative refinement solver with multiple error bounds
cholesky_positive_definite_solve_extra_expert_real :: proc(
	A: ^Matrix($T), // Input matrix
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (pre-allocated)
	AF: ^Matrix(T), // Cholesky factorization (pre-allocated if fact='F')
	S: []T, // Scaling factors (pre-allocated, size n)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []T, // Normwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	err_bnds_comp: []T, // Componentwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	params: []T, // Algorithm parameters (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	fact: Factorization_Type = .New,
	uplo: MatrixRegion = .Upper,
	n_err_bnds: int = 3,
) -> (
	rcond: T,
	rpvgrw: T,
	equed: Equilibration_Type,
	info: Info,// Reciprocal pivot growth factor
	ok: bool,
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo
	equed_c: u8 = 0
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when T == f32 {
		lapack.sposvxx_(
			&fact_c,
			&uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dposvxx_(
			&fact_c,
			&uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	equed = cast(Equilibration_Type)equed_c
	return rcond, rpvgrw, equed, info, info == 0
}

cholesky_positive_definite_solve_extra_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution matrix (pre-allocated)
	AF: ^Matrix(Cmplx), // Cholesky factorization (pre-allocated if fact='F')
	S: []$Real, // Scaling factors (pre-allocated, size n)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []Real, // Normwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	err_bnds_comp: []Real, // Componentwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	params: []Real, // Algorithm parameters (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	fact: Factorization_Type = .New,
	uplo: MatrixRegion = .Upper,
	n_err_bnds: int = 3,
) -> (
	rcond: Real,
	rpvgrw: Real,
	equed: Equilibration_Type,
	info: Info,// Reciprocal pivot growth factor
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(2 * n), "rwork array too small")

	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo
	equed_c: u8 = 0
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when Cmplx == complex64 {
		lapack.cposvxx_(
			&fact_c,
			&uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zposvxx_(
			&fact_c,
			&uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	equed = cast(Equilibration_Type)equed_c
	return rcond, rpvgrw, equed, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (PORFS family)
// ===================================================================================

cholesky_iterative_refinement :: proc {
	cholesky_iterative_refinement_real,
	cholesky_iterative_refinement_complex,
}

query_workspace_cholesky_iterative_refinement :: proc(A: ^Matrix($T)) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when T == f32 || T == f64 {
		work_size = int(3 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		rwork_size = int(n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Iterative refinement for positive definite linear systems
cholesky_iterative_refinement_real :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Cholesky factors
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution (improved on output)
	ferr: []T, // Forward error bounds (pre-allocated, size nrhs)
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
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

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(3 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.sporfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dporfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

cholesky_iterative_refinement_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Cholesky factors
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution (improved on output)
	ferr: []$Real, // Forward error bounds (pre-allocated, size nrhs)
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(n), "rwork array too small")

	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.cporfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zporfs_(&uplo_c, &n, &nrhs, raw_data(A.data), &lda, raw_data(AF.data), &ldaf, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT ITERATIVE REFINEMENT (PORFSX family)
// ===================================================================================

cholesky_iterative_refinement_expert :: proc {
	cholesky_iterative_refinement_expert_real,
	cholesky_iterative_refinement_expert_complex,
}

query_workspace_cholesky_iterative_refinement_expert :: proc(A: ^Matrix($T), n_err_bnds: int = 3) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	n := A.rows

	when T == f32 || T == f64 {
		work_size = int(4 * n)
		iwork_size = int(n)
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = int(2 * n)
		rwork_size = int(2 * n)
		iwork_size = 0
	}

	return work_size, rwork_size, iwork_size
}

// Expert iterative refinement with multiple error bounds
cholesky_iterative_refinement_expert_real :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Cholesky factors
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution (improved on output)
	S: []T, // Scaling factors
	berr: []T, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []T, // Normwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	err_bnds_comp: []T, // Componentwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	params: []T, // Algorithm parameters (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	equed: Equilibration_Type = .None,
	n_err_bnds: int = 3,
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

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(4 * n), "work array too small")
	assert(len(iwork) >= int(n), "iwork array too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)equed
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when T == f32 {
		lapack.sporfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dporfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

cholesky_iterative_refinement_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Original matrix
	AF: ^Matrix(Cmplx), // Cholesky factors
	B: ^Matrix(Cmplx), // RHS matrix
	X: ^Matrix(Cmplx), // Solution (improved on output)
	S: []$Real, // Scaling factors
	berr: []Real, // Backward error bounds (pre-allocated, size nrhs)
	err_bnds_norm: []Real, // Normwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	err_bnds_comp: []Real, // Componentwise error bounds [nrhs x n_err_bnds] (pre-allocated)
	params: []Real, // Algorithm parameters (pre-allocated)
	work: []Cmplx, // Workspace (pre-allocated)
	rwork: []Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	equed: Equilibration_Type = .None,
	n_err_bnds: int = 3,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == n && AF.cols == n, "AF dimensions incorrect")
	assert(B.rows == n, "B must have same number of rows as A")
	assert(X.rows == n && X.cols == nrhs, "X dimensions incorrect")
	assert(len(work) >= int(2 * n), "work array too small")
	assert(len(rwork) >= int(2 * n), "rwork array too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)equed
	n_err_bnds_c := Blas_Int(n_err_bnds)
	nparams := Blas_Int(len(params))

	when Cmplx == complex64 {
		lapack.cporfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zporfsx_(
			&uplo_c,
			&equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_c,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// IMPROVED EQUILIBRATION (POEQUB family)
// ===================================================================================

cholesky_equilibrate_improved :: proc {
	cholesky_equilibrate_improved_real,
	cholesky_equilibrate_improved_complex,
}

// Improved equilibration for positive definite matrices
cholesky_equilibrate_improved_real :: proc(
	A: ^Matrix($T), // Input matrix (not modified)
	S: []T, // Scaling factors (pre-allocated, size n)
) -> (
	scond: T,
	amax: T,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(S) >= int(n), "S array too small")

	when T == f32 {
		lapack.spoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == f64 {
		lapack.dpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

cholesky_equilibrate_improved_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (not modified)
	S: []$Real, // Scaling factors (pre-allocated, size n)
) -> (
	scond: Real,
	amax: Real,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(S) >= int(n), "S array too small")

	when Cmplx == complex64 {
		lapack.cpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when Cmplx == complex128 {
		lapack.zpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// CHOLESKY WITH COMPLETE PIVOTING (PSTRF family)
// ===================================================================================

cholesky_factorize_pivoted :: proc {
	cholesky_factorize_pivoted_real,
	cholesky_factorize_pivoted_complex,
}

query_workspace_cholesky_factorize_pivoted :: proc(A: ^Matrix($T)) -> (work_size: int) where is_float(T) || is_complex(T) {
	n := A.rows
	return int(2 * n)
}

// Cholesky factorization with complete pivoting for semi-definite matrices
cholesky_factorize_pivoted_real :: proc(
	A: ^Matrix($T), // Input matrix (overwritten with Cholesky factor)
	piv: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []T, // Workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	tol: T = -1, // Tolerance for rank determination (negative = use default)
) -> (
	rank: int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(piv) >= int(n), "piv array too small")
	assert(len(work) >= int(2 * n), "work array too small")

	uplo_c := cast(u8)uplo
	rank_c: Blas_Int

	when T == f32 {
		lapack.spstrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(piv), &rank_c, &tol, raw_data(work), &info)
	} else when T == f64 {
		lapack.dpstrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(piv), &rank_c, &tol, raw_data(work), &info)
	}

	return int(rank_c), info, info == 0
}

cholesky_factorize_pivoted_complex :: proc(
	A: ^Matrix($Cmplx), // Input matrix (overwritten with Cholesky factor)
	piv: []Blas_Int, // Pivot indices (pre-allocated, size n)
	work: []$Real, // Real workspace (pre-allocated)
	uplo: MatrixRegion = .Upper,
	tol: Real = -1, // Tolerance for rank determination (negative = use default)
) -> (
	rank: int,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	lda := A.ld

	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(piv) >= int(n), "piv array too small")
	assert(len(work) >= int(2 * n), "work array too small")

	uplo_c := cast(u8)uplo
	rank_c: Blas_Int

	when Cmplx == complex64 {
		lapack.cpstrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(piv), &rank_c, &tol, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zpstrf_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(piv), &rank_c, &tol, raw_data(work), &info)
	}

	return int(rank_c), info, info == 0
}
