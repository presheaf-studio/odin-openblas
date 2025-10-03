package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// TRIANGULAR MATRIX OPERATIONS AND CONDITION NUMBER ESTIMATION
// ===================================================================================
//
// This file provides condition number estimation and other operations for
// full-storage triangular matrices:
// - Condition number estimation (TRCON)
// - Matrix norm computations
// - Well-conditioning checks
//
// All functions use the non-allocating API pattern with pre-allocated arrays.

// ===================================================================================
// CONDITION NUMBER ESTIMATION (TRCON)
// ===================================================================================

// Estimate condition number of triangular matrix
estimate_condition_triangular :: proc {
	estimate_condition_triangular_real,
	estimate_condition_triangular_complex,
}

// Real triangular condition number estimation (f32/f64)
estimate_condition_triangular_real :: proc(
	A: []$T, // Triangular matrix [n×n]
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	n: int, // Matrix dimension
	lda: int, // Leading dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	// Validate inputs
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)

	when T == f32 {
		lapack.strcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtrcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// Complex triangular condition number estimation (complex64/complex128)
estimate_condition_triangular_complex :: proc(
	A: []$Cmplx, // Triangular matrix [n×n]
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []$Real, // Pre-allocated real workspace (size n)
	n: int, // Matrix dimension
	lda: int, // Leading dimension
	norm: MatrixNorm = .OneNorm, // Norm type
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	// Validate inputs
	assert(validate_triangular(n, lda, len(A)), "Invalid triangular matrix dimensions")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(uplo)
	diag_c := u8(diag)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)

	when Cmplx == complex64 {
		lapack.ctrcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztrcon_(&norm_c, &uplo_c, &diag_c, &n_blas, raw_data(A), &lda_blas, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	ok = (info == 0)
	return rcond, info, ok
}

// ===================================================================================
// MATRIX NORM COMPUTATIONS
// ===================================================================================

// Compute 1-norm of triangular matrix
one_norm_triangular :: proc {
	one_norm_triangular_real,
	one_norm_triangular_complex,
}

one_norm_triangular_real :: proc(A: []$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> T where is_float(T) {
	max_col_sum := T(0)

	switch uplo {
	case .Upper:
		for j in 0 ..< n {
			col_sum := T(0)
			for i in 0 ..= j {
				if diag == .Unit && i == j {
					col_sum += T(1)
				} else {
					col_sum += abs(A[i + j * lda])
				}
			}
			max_col_sum = max(max_col_sum, col_sum)
		}
	case .Lower:
		for j in 0 ..< n {
			col_sum := T(0)
			for i in j ..< n {
				if diag == .Unit && i == j {
					col_sum += T(1)
				} else {
					col_sum += abs(A[i + j * lda])
				}
			}
			max_col_sum = max(max_col_sum, col_sum)
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_col_sum
}

one_norm_triangular_complex :: proc(A: []$Cmplx, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> $Real where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	max_col_sum := Real(0)

	switch uplo {
	case .Upper:
		for j in 0 ..< n {
			col_sum := Real(0)
			for i in 0 ..= j {
				if diag == .Unit && i == j {
					col_sum += Real(1)
				} else {
					col_sum += abs(A[i + j * lda])
				}
			}
			max_col_sum = max(max_col_sum, col_sum)
		}
	case .Lower:
		for j in 0 ..< n {
			col_sum := Real(0)
			for i in j ..< n {
				if diag == .Unit && i == j {
					col_sum += Real(1)
				} else {
					col_sum += abs(A[i + j * lda])
				}
			}
			max_col_sum = max(max_col_sum, col_sum)
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_col_sum
}

// Compute infinity-norm of triangular matrix
infinity_norm_triangular :: proc {
	infinity_norm_triangular_real,
	infinity_norm_triangular_complex,
}

infinity_norm_triangular_real :: proc(A: []$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> T where is_float(T) {
	max_row_sum := T(0)

	switch uplo {
	case .Upper:
		for i in 0 ..< n {
			row_sum := T(0)
			for j in i ..< n {
				if diag == .Unit && i == j {
					row_sum += T(1)
				} else {
					row_sum += abs(A[i + j * lda])
				}
			}
			max_row_sum = max(max_row_sum, row_sum)
		}
	case .Lower:
		for i in 0 ..< n {
			row_sum := T(0)
			for j in 0 ..= i {
				if diag == .Unit && i == j {
					row_sum += T(1)
				} else {
					row_sum += abs(A[i + j * lda])
				}
			}
			max_row_sum = max(max_row_sum, row_sum)
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_row_sum
}

infinity_norm_triangular_complex :: proc(A: []$Cmplx, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> $Real where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	max_row_sum := Real(0)

	switch uplo {
	case .Upper:
		for i in 0 ..< n {
			row_sum := Real(0)
			for j in i ..< n {
				if diag == .Unit && i == j {
					row_sum += Real(1)
				} else {
					row_sum += abs(A[i + j * lda])
				}
			}
			max_row_sum = max(max_row_sum, row_sum)
		}
	case .Lower:
		for i in 0 ..< n {
			row_sum := Real(0)
			for j in 0 ..= i {
				if diag == .Unit && i == j {
					row_sum += Real(1)
				} else {
					row_sum += abs(A[i + j * lda])
				}
			}
			max_row_sum = max(max_row_sum, row_sum)
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_row_sum
}

// Compute max-norm (largest absolute value) of triangular matrix
max_norm_triangular :: proc {
	max_norm_triangular_real,
	max_norm_triangular_complex,
}

max_norm_triangular_real :: proc(A: []$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> T where is_float(T) {
	max_val := T(0)

	switch uplo {
	case .Upper:
		for j in 0 ..< n {
			for i in 0 ..= j {
				if diag == .Unit && i == j {
					max_val = max(max_val, T(1))
				} else {
					max_val = max(max_val, abs(A[i + j * lda]))
				}
			}
		}
	case .Lower:
		for j in 0 ..< n {
			for i in j ..< n {
				if diag == .Unit && i == j {
					max_val = max(max_val, T(1))
				} else {
					max_val = max(max_val, abs(A[i + j * lda]))
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_val
}

max_norm_triangular_complex :: proc(A: []$Cmplx, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> $Real where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	max_val := Real(0)

	switch uplo {
	case .Upper:
		for j in 0 ..< n {
			for i in 0 ..= j {
				if diag == .Unit && i == j {
					max_val = max(max_val, Real(1))
				} else {
					max_val = max(max_val, abs(A[i + j * lda]))
				}
			}
		}
	case .Lower:
		for j in 0 ..< n {
			for i in j ..< n {
				if diag == .Unit && i == j {
					max_val = max(max_val, Real(1))
				} else {
					max_val = max(max_val, abs(A[i + j * lda]))
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return max_val
}

// Compute specified norm of triangular matrix
norm_triangular :: proc {
	norm_triangular_real,
	norm_triangular_complex,
}

norm_triangular_real :: proc(A: []$T, n: int, lda: int, norm_type: MatrixNorm, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> T where is_float(T) {
	switch norm_type {
	case .OneNorm:
		return one_norm_triangular_real(A, n, lda, uplo, diag)
	case .InfinityNorm:
		return infinity_norm_triangular_real(A, n, lda, uplo, diag)
	case .MaxNorm:
		return max_norm_triangular_real(A, n, lda, uplo, diag)
	case .FrobeniusNorm:
		// Use existing frobenius_norm_triangular from triangular.odin
		tri := Triangular(T) {
			data = A,
			n    = n,
			lda  = lda,
			uplo = uplo,
			diag = diag,
		}
		return frobenius_norm_triangular_real(&tri)
	}
}

norm_triangular_complex :: proc(A: []$Cmplx, n: int, lda: int, norm_type: MatrixNorm, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> $Real where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	switch norm_type {
	case .OneNorm:
		return one_norm_triangular_complex(A, n, lda, uplo, diag)
	case .InfinityNorm:
		return infinity_norm_triangular_complex(A, n, lda, uplo, diag)
	case .MaxNorm:
		return max_norm_triangular_complex(A, n, lda, uplo, diag)
	case .FrobeniusNorm:
		// Use existing frobenius_norm_triangular from triangular.odin
		tri := Triangular(Cmplx) {
			data = A,
			n    = n,
			lda  = lda,
			uplo = uplo,
			diag = diag,
		}
		return frobenius_norm_triangular_complex(&tri)
	}
}

// ===================================================================================
// WELL-CONDITIONING CHECKS
// ===================================================================================

// Note: is_well_conditioned_triangular is defined in packed_triangular.odin

// Estimate condition number and return both reciprocal and actual condition number
estimate_condition_with_actual :: proc {
	estimate_condition_with_actual_real,
	estimate_condition_with_actual_complex,
}

estimate_condition_with_actual_real :: proc(A: []$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit, norm_type: MatrixNorm = .OneNorm, allocator := context.allocator) -> (rcond: T, cond: T, ok: bool) where is_float(T) {
	work, work_err := make([]T, 3 * n, allocator)
	if work_err != nil do return 0, 0, false
	defer delete(work, allocator)
	iwork, iwork_err := make([]Blas_Int, n, allocator)
	if iwork_err != nil do return 0, 0, false
	defer delete(iwork, allocator)

	rcond_val, info, success := estimate_condition_triangular_real(A, work, iwork, n, lda, norm_type, uplo, diag)
	if !success || rcond_val <= 0 {
		return 0, max(T), false
	}
	return rcond_val, 1.0 / rcond_val, true
}

estimate_condition_with_actual_complex :: proc(A: []$Cmplx, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit, norm_type: MatrixNorm = .OneNorm, allocator := context.allocator) -> (rcond: $Real, cond: Real, ok: bool) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	work, work_err := make([]Cmplx, 2 * n, allocator)
	if work_err != nil do return 0, 0, false
	defer delete(work, allocator)
	rwork, rwork_err := make([]Real, n, allocator)
	if rwork_err != nil do return 0, 0, false
	defer delete(rwork, allocator)

	rcond_val, info, success := estimate_condition_triangular_complex(A, work, rwork, n, lda, norm_type, uplo, diag)
	if !success || rcond_val <= 0 {
		return 0, max(Real), false
	}
	return rcond_val, 1.0 / rcond_val, true
}

// ===================================================================================
// SYLVESTER EQUATION SOLVER (TRSYL)
// ===================================================================================

// Solve the triangular Sylvester equation: op(A)*X ± X*op(B) = scale*C
// where op(M) can be M, M^T, or M^H
solve_sylvester :: proc {
	solve_sylvester_real,
	solve_sylvester_complex,
}

// Real Sylvester equation solver (f32/f64)
solve_sylvester_real :: proc(
	A: []$T, // Triangular matrix A [m×m]
	B: []T, // Triangular matrix B [n×n]
	C: []T, // Right-hand side matrix C [m×n] (overwritten with solution X)
	m, n: int, // Matrix dimensions
	lda, ldb, ldc: int, // Leading dimensions
	trana: TransposeMode = .None, // Transpose operation for A
	tranb: TransposeMode = .None, // Transpose operation for B
	isgn: int = 1, // Sign in equation: +1 for plus, -1 for minus
) -> (
	scale: T,
	info: Info,// Scaling factor applied to C
	ok: bool,
) where is_float(T) {
	// Validate inputs
	assert(len(A) >= m * lda, "A array too small")
	assert(len(B) >= n * ldb, "B array too small")
	assert(len(C) >= m * ldc, "C array too small")
	assert(lda >= max(1, m), "Leading dimension lda too small")
	assert(ldb >= max(1, n), "Leading dimension ldb too small")
	assert(ldc >= max(1, m), "Leading dimension ldc too small")
	assert(isgn == 1 || isgn == -1, "isgn must be 1 or -1")

	trana_c := u8(trana)
	tranb_c := u8(tranb)
	m_blas := Blas_Int(m)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)
	ldb_blas := Blas_Int(ldb)
	ldc_blas := Blas_Int(ldc)
	isgn_blas := Blas_Int(isgn)

	when T == f32 {
		lapack.strsyl_(&trana_c, &tranb_c, &isgn_blas, &m_blas, &n_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(C), &ldc_blas, &scale, &info)
	} else when T == f64 {
		lapack.dtrsyl_(&trana_c, &tranb_c, &isgn_blas, &m_blas, &n_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(C), &ldc_blas, &scale, &info)
	}

	ok = (info == 0)
	return scale, info, ok
}

// Complex Sylvester equation solver (complex64/complex128)
solve_sylvester_complex :: proc(
	A: []$Cmplx, // Triangular matrix A [m×m]
	B: []Cmplx, // Triangular matrix B [n×n]
	C: []Cmplx, // Right-hand side matrix C [m×n] (overwritten with solution X)
	m, n: int, // Matrix dimensions
	lda, ldb, ldc: int, // Leading dimensions
	trana: TransposeMode = .None, // Transpose operation for A
	tranb: TransposeMode = .None, // Transpose operation for B
	isgn: int = 1, // Sign in equation: +1 for plus, -1 for minus
) -> (
	scale: $Real,
	info: Info,// Scaling factor applied to C (real type)
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	// Validate inputs
	assert(len(A) >= m * lda, "A array too small")
	assert(len(B) >= n * ldb, "B array too small")
	assert(len(C) >= m * ldc, "C array too small")
	assert(lda >= max(1, m), "Leading dimension lda too small")
	assert(ldb >= max(1, n), "Leading dimension ldb too small")
	assert(ldc >= max(1, m), "Leading dimension ldc too small")
	assert(isgn == 1 || isgn == -1, "isgn must be 1 or -1")

	trana_c := u8(trana)
	tranb_c := u8(tranb)
	m_blas := Blas_Int(m)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)
	ldb_blas := Blas_Int(ldb)
	ldc_blas := Blas_Int(ldc)
	isgn_blas := Blas_Int(isgn)

	when Cmplx == complex64 {
		lapack.ctrsyl_(&trana_c, &tranb_c, &isgn_blas, &m_blas, &n_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(C), &ldc_blas, &scale, &info)
	} else when Cmplx == complex128 {
		lapack.ztrsyl_(&trana_c, &tranb_c, &isgn_blas, &m_blas, &n_blas, raw_data(A), &lda_blas, raw_data(B), &ldb_blas, raw_data(C), &ldc_blas, &scale, &info)
	}

	ok = (info == 0)
	return scale, info, ok
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace size for condition number estimation
query_workspace_condition_triangular :: proc($T: typeid) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		return 3, 1, 0 // 3*n work, n iwork, 0 rwork (per n)
	} else when is_complex(T) {
		return 2, 0, 1 // 2*n work, 0 iwork, n rwork (per n)
	}
}
