package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED LINEAR SYSTEM SOLVERS
// ===================================================================================

// Simple solver for positive definite banded systems (polymorphic)

// Expert solver for positive definite banded systems proc group
m_solve_banded_pd_expert :: proc {
	m_solve_banded_pd_expert_f32_c64,
	m_solve_banded_pd_expert_f64_c128,
}

// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Solve positive definite banded system (polymorphic)
// Solves A*X = B where A is positive definite banded
m_solve_banded_pd :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(T), // Right-hand side (input/output - solution on output)
	kd: Blas_Int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	// Validate inputs
	assert(len(AB.data) > 0 && len(B.data) > 0, "Matrices cannot be empty")
	assert(AB.rows == AB.cols, "AB must be square")
	assert(B.rows == AB.rows, "System dimensions must be consistent")
	assert(kd >= 0 && kd < AB.rows, "Invalid bandwidth kd")

	uplo_c: cstring = "U" if uplo_upper else "L"
	n: Blas_Int = AB.cols
	kd := kd
	nrhs: Blas_Int = B.cols
	ldab: Blas_Int = AB.ld
	ldb: Blas_Int = B.ld
	info_val: Info

	when T == f32 {
		lapack.spbsv_(
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info_val,
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dpbsv_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info_val,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbsv_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info_val,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbsv_(
			uplo_c,
			&n,
			&kd_val,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info_val,
			len(uplo_c),
		)
	}

	return info_val, info_val == 0
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Expert solver result structure
ExpertSolverResult :: struct($T: typeid) {
	X:     Matrix(T), // Solution matrix
	rcond: T, // Reciprocal condition number
	ferr:  []T, // Forward error bounds
	berr:  []T, // Backward error bounds
	equed: EquilibrationState, // Equilibration state
	S:     []T, // Scaling factors (if equilibrated)
	ok:    bool,
	info:  Blas_Int,
}

// Expert solve for positive definite banded system (f32/complex64)
// Solves with equilibration, condition estimation, and error bounds
m_solve_banded_pd_expert_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(T) = nil, // Pre-factored matrix (optional)
	S_in: []f32 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f32) where T == f32 || T == complex64 {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(T)
	if AFB == nil {
		AFB_local = make_matrix(T, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(T, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f32, n, allocator)
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)
	rcond: f32
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace and call appropriate routine
	info_val: Info

	when T == f32 {
		work := make([]f32, 3 * n, context.temp_allocator)
		iwork := make([]Blas_Int, n, context.temp_allocator)

		lapack.spbsvx_(
			fact_c,
			uplo_c,
			&n_val,
			&kd_val,
			&nrhs_val,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info_val,
			len(fact_c),
			len(uplo_c),
			1,
		)
	} else when T == complex64 {
		work := make([]complex64, 2 * n, context.temp_allocator)
		rwork := make([]f32, n, context.temp_allocator)

		lapack.cpbsvx_(
			fact_c,
			uplo_c,
			&n_val,
			&kd_val,
			&nrhs_val,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(fact_c),
			len(uplo_c),
			1, // Last is for equed
		)
	}

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f32) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		ok = info_val == 0,
		info = info_val,
	}
}

// Expert solve for positive definite banded system (f64/complex128)
m_solve_banded_pd_expert_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(T) = nil, // Pre-factored matrix (optional)
	S_in: []f64 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f64) where T == f64 || T == complex128 {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(T)
	if AFB == nil {
		AFB_local = make_matrix(T, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(T, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f64, n, allocator)
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)
	rcond: f64
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace and call appropriate routine
	info_val: Info

	when T == f64 {
		work := make([]f64, 3 * n, context.temp_allocator)
		iwork := make([]Blas_Int, n, context.temp_allocator)

		lapack.dpbsvx_(
			fact_c,
			uplo_c,
			&n_val,
			&kd_val,
			&nrhs_val,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info_val,
			len(fact_c),
			len(uplo_c),
			1,
		)
	} else when T == complex128 {
		work := make([]complex128, 2 * n, context.temp_allocator)
		rwork := make([]f64, n, context.temp_allocator)

		lapack.zpbsvx_(
			fact_c,
			uplo_c,
			&n_val,
			&kd_val,
			&nrhs_val,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			&equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info_val,
			len(fact_c),
			len(uplo_c),
			1,
		)
	}

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f64) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		ok = info_val == 0,
		info = info_val,
	}
}


// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Simple solve with solution extraction
solve_banded_system :: proc(
	A: Matrix($T), // Input matrix (will be copied)
	b: []T, // Right-hand side vector
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	x: []T,
	ok: bool,
) {
	// Create working copies
	AB := make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)
	copy_matrix(&A, &AB)

	B := make_matrix(T, len(b), 1, .General, allocator)
	for i in 0 ..< len(b) {
		matrix_set(&B, i, 0, b[i])
	}

	// Solve system
	ok, _ = m_solve_banded_pd(&AB, &B, Blas_Int(kd), uplo_upper, allocator)

	// Extract solution
	if ok {
		x = make([]T, len(b), allocator)
		for i in 0 ..< len(b) {
			x[i] = matrix_get(&B, i, 0)
		}
	}

	delete_matrix(&AB)
	delete_matrix(&B)
	return x, ok
}

solve_banded_expert :: proc(
	A: Matrix($T), // Input matrix
	b: []T, // Right-hand side
	kd: int, // Bandwidth
	equilibrate := true,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(T),
) {
	// Create working copies
	AB := make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)
	copy_matrix(&A, &AB)

	B := make_matrix(T, len(b), 1, .General, allocator)
	for i in 0 ..< len(b) {
		matrix_set(&B, i, 0, b[i])
	}

	fact_option := FactorizationOption.Equilibrate if equilibrate else FactorizationOption.Factor

	when T == f32 || T == complex64 {
		result = m_solve_banded_pd_expert_f32_c64(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else when T == f64 || T == complex128 {
		result = m_solve_banded_pd_expert_f64_c128(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else {
		#panic("Unsupported type for expert solve")
	}

	delete_matrix(&AB)
	delete_matrix(&B)
	return result
}

is_solution_accurate :: proc(
	result: ExpertSolverResult($T),
	tolerance: T,
) -> bool where is_float(T) ||
	is_complex(T) {
	if !result.ok {
		return false
	}

	// Check condition number
	if result.rcond < tolerance {
		return false // Matrix is ill-conditioned
	}

	// Check error bounds
	for err in result.ferr {
		if err > tolerance {
			return false
		}
	}

	return true
}

delete_expert_result :: proc(result: ^ExpertSolverResult($T)) {
	delete_matrix(&result.X)
	if result.ferr != nil do delete(result.ferr)
	if result.berr != nil do delete(result.berr)
	if result.S != nil do delete(result.S)
}
