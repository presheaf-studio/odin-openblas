package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"

// ===================================================================================
// RANDOM ORTHOGONAL TRANSFORMATION (DLAROR/SLAROR/CLAROR/ZLAROR)
// ===================================================================================
query_workspace_dns_apply_random_orthogonal :: proc(A: ^Matrix($T), work: []T, side: OrthogonalSide = .Left, init: OrthogonalInit = .None) -> (work_size: int) {
	m := A.rows
	n := A.cols
	work_size: Blas_Int
	switch side {
	case .Left:
		work_size = m
	case .Right:
		work_size = n
	case .Both:
		work_size = max(m, n)
	}
	return work_size
}
// Apply random orthogonal transformation for all types
dns_apply_random_orthogonal :: proc(A: ^Matrix($T), work: []T, side: OrthogonalSide = .Left, init: OrthogonalInit = .None, seed: ^[4]i32) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	// Validate input
	assert(A != nil && A.data != nil, "Matrix A cannot be nil")
	assert(seed != nil, "Seed cannot be nil")
	assert(seed[3] % 2 == 1, "seed[3] must be odd")

	m := A.rows
	n := A.cols
	lda := A.ld

	side_c := cast(u8)side
	init_c := cast(u8)init

	when T == f32 {
		lapack.slaror_(&side_c, &init_c, &m, &n, raw_data(A.data), &lda, &seed[0], raw_data(work), &info)
	} else when T == f64 {
		lapack.dlaror_(&side_c, &init_c, &m, &n, raw_data(A.data), &lda, &seed[0], raw_data(work), &info)
	} else when T == complex64 {
		lapack.claror_(&side_c, &init_c, &m, &n, raw_data(A.data), &lda, &seed[0], raw_data(work), &info)
	} else when T == complex128 {
		lapack.zlaror_(&side_c, &init_c, &m, &n, raw_data(A.data), &lda, &seed[0], raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// GIVENS ROTATION APPLICATION (DLAROT/SLAROT/CLAROT/ZLAROT)
// ===================================================================================

// Apply Givens rotation to matrix row/column for all types
dns_apply_givens_rotation :: proc(
	A: ^Matrix($T),
	row_mode: bool, // true for row operation, false for column operation
	left_rotation: bool, // Apply rotation on the left
	right_rotation: bool, // Apply rotation on the right
	num_elements: int, // Number of elements to rotate
	c: $RotType, // Cosine of rotation angle
	s: RotType, // Sine of rotation angle
	start_row: int = 0, // Starting row index
	start_col: int = 0, // Starting column index
	xleft: ^RotType = nil, // Element rotated into A from the left
	xright: ^RotType = nil, // Element rotated into A from the right
) where (T == f32 && RotType == f32) || (T == f64 && RotType == f64) || (T == complex64 && RotType == f32) || (T == complex128 && RotType == f64) {
	lrows := row_mode ? Blas_Int(1) : Blas_Int(0)
	lleft := left_rotation ? Blas_Int(1) : Blas_Int(0)
	lright := right_rotation ? Blas_Int(1) : Blas_Int(0)
	nl_val := Blas_Int(num_elements)
	c_val := c
	s_val := s
	lda := A.ld

	// Calculate starting position
	start_ptr := &A.data[start_col * A.ld + start_row]

	// Optional rotation elements
	xleft_val := xleft != nil ? xleft^ : RotType(0)
	xright_val := xright != nil ? xright^ : RotType(0)

	when T == f32 {
		lapack.slarot_(&lrows, &lleft, &lright, &nl_val, &c_val, &s_val, start_ptr, &lda, &xleft_val, &xright_val)
	} else when T == f64 {
		lapack.dlarot_(&lrows, &lleft, &lright, &nl_val, &c_val, &s_val, start_ptr, &lda, &xleft_val, &xright_val)
	} else when T == complex64 {
		lapack.clarot_(&lrows, &lleft, &lright, &nl_val, &c_val, &s_val, start_ptr, &lda, &xleft_val, &xright_val)
	} else when T == complex128 {
		lapack.zlarot_(&lrows, &lleft, &lright, &nl_val, &c_val, &s_val, start_ptr, &lda, &xleft_val, &xright_val)
	}

	// Update output parameters
	if xleft != nil {
		xleft^ = xleft_val
	}
	if xright != nil {
		xright^ = xright_val
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Generate random orthogonal matrix
dns_generate_random_orthogonal :: proc(A: ^Matrix($T), seed: [4]i32 = {1, 2, 3, 4}) -> (info: Info) where is_float(T) || is_complex(T) {
	assert(A.rows == A.cols, "Random orthogonal matrix must be square")

	initialize_matrix(A, T(0), T(1), .Full)

	// Apply random orthogonal transformation
	seed_copy := seed
	seed_copy[3] = seed_copy[3] | 1 // Ensure seed[3] is odd

	return dns_apply_random_orthogonal(A, .Left, .Identity, &seed_copy)
}

// Apply single Givens rotation to two rows/columns
apply_simple_givens_rotation :: proc(
	A: ^Matrix($T),
	i, j: int, // Row/column indices to rotate
	c, s: $RotType, // Cosine and sine of rotation angle
	apply_to_rows: bool = true, // true for row rotation, false for column rotation
) where (T == f32 && RotType == f32) || (T == f64 && RotType == f64) || (T == complex64 && RotType == f32) || (T == complex128 && RotType == f64) {
	if apply_to_rows {
		// Rotate rows i and j
		for col in 0 ..< A.cols {
			xi := A.data[col * A.ld + i]
			xj := A.data[col * A.ld + j]

			when is_float(T) {
				A.data[col * A.ld + i] = T(c) * xi + T(s) * xj
				A.data[col * A.ld + j] = -T(s) * xi + T(c) * xj
			} else when is_complex(T) {
				A.data[col * A.ld + i] = T(complex(c, 0)) * xi + T(complex(s, 0)) * xj
				A.data[col * A.ld + j] = -T(complex(s, 0)) * xi + T(complex(c, 0)) * xj
			}
		}
	} else {
		// Rotate columns i and j
		for row in 0 ..< A.rows {
			xi := A.data[i * A.ld + row]
			xj := A.data[j * A.ld + row]

			when is_float(T) {
				A.data[i * A.ld + row] = T(c) * xi + T(s) * xj
				A.data[j * A.ld + row] = -T(s) * xi + T(c) * xj
			} else when is_complex(T) {
				A.data[i * A.ld + row] = T(complex(c, 0)) * xi + T(complex(s, 0)) * xj
				A.data[j * A.ld + row] = -T(complex(s, 0)) * xi + T(complex(c, 0)) * xj
			}
		}
	}
}

// ===================================================================================
// WORKSPACE QUERIES FOR ORTHOGONAL MATRIX OPERATIONS
// ===================================================================================

// Query workspace size for generating Q from QR factorization (ORGQR/UNGQR)
query_workspace_generate_qr :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Query workspace size for generating Q from LQ factorization (ORGLQ/UNGLQ)
query_workspace_generate_lq :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Query workspace size for multiplying by Q from QR factorization (ORMQR/UNMQR)
query_workspace_multiply_qr :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Query workspace size for multiplying by Q from LQ factorization (ORMLQ/UNMLQ)
query_workspace_multiply_lq :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// ===================================================================================
// QR ORTHOGONAL GENERATION (ORGQR/UNGQR)
// ===================================================================================

// Generate orthogonal/unitary matrix Q from QR factorization
// Works for both real (ORGQR) and complex (UNGQR) types
//
// This generates the explicit m×n matrix Q with orthonormal columns from the
// elementary reflectors returned by QR factorization (GEQRF). The first k columns
// of Q are the orthogonal/unitary matrix corresponding to the factorization.
//
// Parameters:
//   A: On entry, contains elementary reflectors from GEQRF.
//      On exit, overwritten with the m×n matrix Q.
//   tau: Scalar factors of elementary reflectors from GEQRF (length ≥ k)
//   k: Number of elementary reflectors (≤ min(m,n))
//   work: Workspace array (use query_workspace_generate_qr to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
dns_orthogonal_from_qr :: proc(A: ^Matrix($T), tau: []T, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorgqr_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorgqr_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cungqr_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zungqr_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// LQ ORTHOGONAL GENERATION (ORGLQ/UNGLQ)
// ===================================================================================

// Generate orthogonal/unitary matrix Q from LQ factorization
// Works for both real (ORGLQ) and complex (UNGLQ) types
//
// This generates the explicit m×n matrix Q with orthonormal rows from the
// elementary reflectors returned by LQ factorization (GELQF). The first k rows
// of Q are the orthogonal/unitary matrix corresponding to the factorization.
//
// Parameters:
//   A: On entry, contains elementary reflectors from GELQF.
//      On exit, overwritten with the m×n matrix Q.
//   tau: Scalar factors of elementary reflectors from GELQF (length ≥ k)
//   k: Number of elementary reflectors (≤ min(m,n))
//   work: Workspace array (use query_workspace_generate_lq to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
dns_orthogonal_from_lq :: proc(A: ^Matrix($T), tau: []T, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorglq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorglq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunglq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunglq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// PARAMETER ENUMS FOR MULTIPLICATION
// ===================================================================================

// Side of multiplication (left or right)
MultiplicationSide :: enum u8 {
	Left  = 'L', // Apply Q from the left (Q * C)
	Right = 'R', // Apply Q from the right (C * Q)
}

// Transpose operation
TransposeOperation :: enum u8 {
	NoTranspose = 'N', // Apply Q
	Transpose   = 'T', // Apply Q^T (real) or Q^H (complex conjugate transpose)
}

// ===================================================================================
// QR ORTHOGONAL MULTIPLICATION (ORMQR/UNMQR)
// ===================================================================================

// Multiply by orthogonal/unitary matrix Q from QR factorization
// Works for both real (ORMQR) and complex (UNMQR) types
//
// Performs one of the operations: C := Q*C, C := Q^T*C, C := C*Q, or C := C*Q^T
// where Q is the orthogonal/unitary matrix from QR factorization (GEQRF).
//
// Parameters:
//   A: Matrix containing elementary reflectors from GEQRF
//   tau: Scalar factors from GEQRF (length ≥ k)
//   C: On entry, the m×n matrix C. On exit, overwritten by the product.
//   side: Whether to apply Q from left or right
//   transpose: Whether to apply Q or Q^T (Q^H for complex)
//   k: Number of elementary reflectors (≤ min(m,n) of A)
//   work: Workspace array (use query_workspace_multiply_qr to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
dns_orthogonal_apply_qr :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormqr_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormqr_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmqr_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmqr_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// LQ ORTHOGONAL MULTIPLICATION (ORMLQ/UNMLQ)
// ===================================================================================

// Multiply by orthogonal/unitary matrix Q from LQ factorization
// Works for both real (ORMLQ) and complex (UNMLQ) types
//
// Performs one of the operations: C := Q*C, C := Q^T*C, C := C*Q, or C := C*Q^T
// where Q is the orthogonal/unitary matrix from LQ factorization (GELQF).
//
// Parameters:
//   A: Matrix containing elementary reflectors from GELQF
//   tau: Scalar factors from GELQF (length ≥ k)
//   C: On entry, the m×n matrix C. On exit, overwritten by the product.
//   side: Whether to apply Q from left or right
//   transpose: Whether to apply Q or Q^T (Q^H for complex)
//   k: Number of elementary reflectors (≤ min(m,n) of A)
//   work: Workspace array (use query_workspace_multiply_lq to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
dns_orthogonal_apply_lq :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormlq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormlq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmlq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmlq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// ADDITIONAL WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace size for generating Q from QR factorization (ORGQR/UNGQR)
query_workspace_generate_q_from_qr :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungqr_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Query workspace size for generating Q from LQ factorization (ORGLQ/UNGLQ)
query_workspace_generate_q_from_lq :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunglq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Query workspace size for multiplying by Q from QR factorization (ORMQR/UNMQR)
query_workspace_multiply_q_from_qr :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmqr_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	}
	return int(real(work_query))
}

// Query workspace size for multiplying by Q from LQ factorization (ORMLQ/UNMLQ)
query_workspace_multiply_q_from_lq :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose

	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmlq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	}
	return int(real(work_query))
}

// ===================================================================================
// QL ORTHOGONAL MULTIPLICATION (ORMQL/UNMQL)
// ===================================================================================

// Query workspace size for multiplying by Q from QL factorization (ORMQL/UNMQL)
query_workspace_multiply_q_from_ql :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormql_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormql_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmql_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmql_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Multiply by orthogonal/unitary matrix Q from QL factorization
// Works for both real (ORMQL) and complex (UNMQL) types
//
// Performs one of the operations: C := Q*C, C := Q^T*C, C := C*Q, or C := C*Q^T
// where Q is the orthogonal/unitary matrix from QL factorization (GEQLF).
//
// Parameters:
//   A: Matrix containing elementary reflectors from GEQLF
//   tau: Scalar factors from GEQLF (length ≥ k)
//   C: On entry, the m×n matrix C. On exit, overwritten by the product.
//   side: Whether to apply Q from left or right
//   transpose: Whether to apply Q or Q^T (Q^H for complex)
//   k: Number of elementary reflectors (≤ min(m,n) of A)
//   work: Workspace array (use query_workspace_multiply_q_from_ql to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_from_ql :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormql_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormql_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmql_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmql_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RQ ORTHOGONAL MULTIPLICATION (ORMRQ/UNMRQ)
// ===================================================================================

// Query workspace size for multiplying by Q from RQ factorization (ORMRQ/UNMRQ)
query_workspace_multiply_q_from_rq :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormrq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormrq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmrq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmrq_(&side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	return int(real(work_query))
}

// Multiply by orthogonal/unitary matrix Q from RQ factorization
// Works for both real (ORMRQ) and complex (UNMRQ) types
//
// Performs one of the operations: C := Q*C, C := Q^T*C, C := C*Q, or C := C*Q^T
// where Q is the orthogonal/unitary matrix from RQ factorization (GERQF).
//
// Parameters:
//   A: Matrix containing elementary reflectors from GERQF
//   tau: Scalar factors from GERQF (length ≥ k)
//   C: On entry, the m×n matrix C. On exit, overwritten by the product.
//   side: Whether to apply Q from left or right
//   transpose: Whether to apply Q or Q^T (Q^H for complex)
//   k: Number of elementary reflectors (≤ min(m,n) of A)
//   work: Workspace array (use query_workspace_multiply_q_from_rq to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_from_rq :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormrq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormrq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmrq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmrq_(&side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// QL ORTHOGONAL GENERATION (ORGQL/UNGQL)
// ===================================================================================

// Query workspace size for generating Q from QL factorization (ORGQL/UNGQL)
query_workspace_generate_q_from_ql :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgql_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgql_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungql_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungql_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	}

	return int(real(work_query))
}

// Generate orthogonal/unitary matrix Q from QL factorization
// Works for both real (ORGQL) and complex (UNGQL) types
//
// This generates the explicit m×n matrix Q with orthonormal columns from the
// elementary reflectors returned by QL factorization (GEQLF). The last k columns
// of Q are the orthogonal/unitary matrix corresponding to the factorization.
//
// Parameters:
//   A: On entry, contains elementary reflectors from GEQLF.
//      On exit, overwritten with the m×n matrix Q.
//   tau: Scalar factors of elementary reflectors from GEQLF (length ≥ k)
//   k: Number of elementary reflectors (≤ min(m,n))
//   work: Workspace array (use query_workspace_generate_q_from_ql to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_q_from_ql :: proc(A: ^Matrix($T), tau: []T, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorgql_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorgql_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cungql_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zungql_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RQ ORTHOGONAL GENERATION (ORGRQ/UNGRQ)
// ===================================================================================

// Query workspace size for generating Q from RQ factorization (ORGRQ/UNGRQ)
query_workspace_generate_q_from_rq :: proc(A: ^Matrix($T), k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgrq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgrq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungrq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungrq_(&m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Generate orthogonal/unitary matrix Q from RQ factorization
// Works for both real (ORGRQ) and complex (UNGRQ) types
//
// This generates the explicit m×n matrix Q with orthonormal rows from the
// elementary reflectors returned by RQ factorization (GERQF). The last k rows
// of Q are the orthogonal/unitary matrix corresponding to the factorization.
//
// Parameters:
//   A: On entry, contains elementary reflectors from GERQF.
//      On exit, overwritten with the m×n matrix Q.
//   tau: Scalar factors of elementary reflectors from GERQF (length ≥ k)
//   k: Number of elementary reflectors (≤ min(m,n))
//   work: Workspace array (use query_workspace_generate_q_from_rq to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_q_from_rq :: proc(A: ^Matrix($T), tau: []T, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorgrq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorgrq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cungrq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zungrq_(&m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL ORTHOGONAL GENERATION (ORGBR/UNGBR)
// ===================================================================================

BidiagonalVectorType :: enum u8 {
	P = 'P', // Generate P (left orthogonal matrix)
	Q = 'Q', // Generate Q (right orthogonal matrix)
}

// Query workspace size for generating Q/P from bidiagonal reduction (ORGBR/UNGBR)
query_workspace_generate_from_bidiagonal :: proc(A: ^Matrix($T), vect: BidiagonalVectorType, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	vect_c := cast(u8)vect
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgbr_(&vect_c, &m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgbr_(&vect_c, &m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungbr_(&vect_c, &m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungbr_(&vect_c, &m, &n, &k_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Generate orthogonal/unitary matrix from bidiagonal reduction
// Works for both real (ORGBR) and complex (UNGBR) types
//
// Generates Q or P from the output of GEBRD (bidiagonal reduction).
//
// Parameters:
//   A: Matrix containing the factorization from GEBRD
//   tau: Scalar factors from GEBRD (length ≥ min(m,n))
//   vect: Which matrix to generate (.P or .Q)
//   k: Number of elementary reflectors
//   work: Workspace array (use query_workspace_generate_from_bidiagonal to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_from_bidiagonal :: proc(A: ^Matrix($T), tau: []T, vect: BidiagonalVectorType, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	vect_c := cast(u8)vect
	m := A.rows
	n := A.cols
	k_val := Blas_Int(k)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorgbr_(&vect_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorgbr_(&vect_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cungbr_(&vect_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zungbr_(&vect_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// HESSENBERG ORTHOGONAL GENERATION (ORGHR/UNGHR)
// ===================================================================================

// Query workspace size for generating Q from Hessenberg reduction (ORGHR/UNGHR)
query_workspace_generate_from_hessenberg :: proc(A: ^Matrix($T), ilo, ihi: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	n := A.rows
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorghr_(&n, &ilo_val, &ihi_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorghr_(&n, &ilo_val, &ihi_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunghr_(&n, &ilo_val, &ihi_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunghr_(&n, &ilo_val, &ihi_val, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Generate orthogonal/unitary matrix from Hessenberg reduction
// Works for both real (ORGHR) and complex (UNGHR) types
//
// Generates Q from the output of GEHRD (Hessenberg reduction).
//
// Parameters:
//   A: Matrix containing the factorization from GEHRD
//   tau: Scalar factors from GEHRD
//   ilo, ihi: Balancing range (1-indexed)
//   work: Workspace array (use query_workspace_generate_from_hessenberg to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_from_hessenberg :: proc(A: ^Matrix($T), tau: []T, ilo, ihi: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(A.rows == A.cols, "Matrix must be square for Hessenberg")
	assert(ilo >= 1 && ihi <= A.rows && ilo <= ihi, "Invalid balancing range")
	assert(len(tau) >= ihi - ilo, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	n := A.rows
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorghr_(&n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorghr_(&n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunghr_(&n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunghr_(&n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIDIAGONAL ORTHOGONAL GENERATION (ORGTR/UNGTR)
// ===================================================================================

// Query workspace size for generating Q from tridiagonal reduction (ORGTR/UNGTR)
query_workspace_generate_from_tridiagonal :: proc(A: ^Matrix($T), uplo: MatrixTriangle) -> (lwork: int) where is_float(T) || is_complex(T) {
	uplo_c := cast(u8)uplo
	n := A.rows
	lda := A.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sorgtr_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dorgtr_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cungtr_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zungtr_(&uplo_c, &n, nil, &lda, nil, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Generate orthogonal/unitary matrix from tridiagonal reduction
// Works for both real (ORGTR) and complex (UNGTR) types
//
// Generates Q from the output of SYTRD/HETRD (tridiagonal reduction).
//
// Parameters:
//   A: Matrix containing the factorization from SYTRD/HETRD
//   tau: Scalar factors from SYTRD/HETRD (length ≥ n-1)
//   uplo: Upper or lower triangular storage
//   work: Workspace array (use query_workspace_generate_from_tridiagonal to determine size)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_from_tridiagonal :: proc(A: ^Matrix($T), tau: []T, uplo: MatrixTriangle, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0, "Matrix cannot be empty")
	assert(A.rows == A.cols, "Matrix must be square for tridiagonal")
	assert(len(tau) >= A.rows - 1, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	uplo_c := cast(u8)uplo
	n := A.rows
	lda := A.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorgtr_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorgtr_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cungtr_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zungtr_(&uplo_c, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL ORTHOGONAL MULTIPLICATION (ORMBR/UNMBR)
// ===================================================================================

// Query workspace size for multiplying by Q/P from bidiagonal (ORMBR/UNMBR)
query_workspace_multiply_from_bidiagonal :: proc(A: ^Matrix($T), C: ^Matrix(T), vect: BidiagonalVectorType, side: MultiplicationSide, transpose: TransposeOperation, k: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	vect_c := cast(u8)vect
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Multiply by orthogonal/unitary matrix from bidiagonal reduction
// Works for both real (ORMBR) and complex (UNMBR) types
//
// Applies Q or P from GEBRD to matrix C.
//
// Parameters:
//   A: Matrix containing factorization from GEBRD
//   tau: Scalar factors from GEBRD
//   C: Matrix to multiply
//   vect: Which matrix to apply (.P or .Q)
//   side: Apply from left or right
//   transpose: Apply transpose or not
//   k: Number of elementary reflectors
//   work: Workspace array
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_from_bidiagonal :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), vect: BidiagonalVectorType, side: MultiplicationSide, transpose: TransposeOperation, k: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= min(A.rows, A.cols), "k cannot exceed min(m,n)")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	vect_c := cast(u8)vect
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmbr_(&vect_c, &side_c, &trans_c, &m, &n, &k_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// HESSENBERG ORTHOGONAL MULTIPLICATION (ORMHR/UNMHR)
// ===================================================================================

// Query workspace size for multiplying by Q from Hessenberg (ORMHR/UNMHR)
query_workspace_multiply_from_hessenberg :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, ilo, ihi: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Multiply by orthogonal/unitary matrix from Hessenberg reduction
// Works for both real (ORMHR) and complex (UNMHR) types
//
// Applies Q from GEHRD to matrix C.
//
// Parameters:
//   A: Matrix containing factorization from GEHRD
//   tau: Scalar factors from GEHRD
//   C: Matrix to multiply
//   side: Apply from left or right
//   transpose: Apply transpose or not
//   ilo, ihi: Balancing range (1-indexed)
//   work: Workspace array
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_from_hessenberg :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, ilo, ihi: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(A.rows == A.cols, "A must be square for Hessenberg")
	assert(ilo >= 1 && ihi <= A.rows && ilo <= ihi, "Invalid balancing range")
	assert(len(tau) >= ihi - ilo, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmhr_(&side_c, &trans_c, &m, &n, &ilo_val, &ihi_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RZ ORTHOGONAL MULTIPLICATION (ORMRZ/UNMRZ)
// ===================================================================================

// Query workspace size for multiplying by Q from RZ factorization (ORMRZ/UNMRZ)
query_workspace_multiply_from_rz :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k, l: int) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	l_val := Blas_Int(l)
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Multiply by orthogonal/unitary matrix from RZ factorization
// Works for both real (ORMRZ) and complex (UNMRZ) types
//
// Applies Q from TZRZF (trapezoidal RZ factorization) to matrix C.
//
// Parameters:
//   A: Matrix containing factorization from TZRZF
//   tau: Scalar factors from TZRZF
//   C: Matrix to multiply
//   side: Apply from left or right
//   transpose: Apply transpose or not
//   k: Number of elementary reflectors
//   l: Number of columns in trapezoidal part
//   work: Workspace array
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_from_rz :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, k, l: int, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(k <= A.rows && l <= A.cols, "Invalid k or l parameters")
	assert(len(tau) >= k, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k_val := Blas_Int(k)
	l_val := Blas_Int(l)
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmrz_(&side_c, &trans_c, &m, &n, &k_val, &l_val, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIDIAGONAL ORTHOGONAL MULTIPLICATION (ORMTR/UNMTR)
// ===================================================================================

// Query workspace size for multiplying by Q from tridiagonal (ORMTR/UNMTR)
query_workspace_multiply_from_tridiagonal :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation, uplo: MatrixTriangle) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	uplo_c := cast(u8)uplo
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	lda := A.ld
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T

	when T == f32 {
		lapack.sormtr_(&side_c, &uplo_c, &trans_c, &m, &n, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dormtr_(&side_c, &uplo_c, &trans_c, &m, &n, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cunmtr_(&side_c, &uplo_c, &trans_c, &m, &n, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zunmtr_(&side_c, &uplo_c, &trans_c, &m, &n, nil, &lda, nil, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Multiply by orthogonal/unitary matrix from tridiagonal reduction
// Works for both real (ORMTR) and complex (UNMTR) types
//
// Applies Q from SYTRD/HETRD to matrix C.
//
// Parameters:
//   A: Matrix containing factorization from SYTRD/HETRD
//   tau: Scalar factors from SYTRD/HETRD
//   C: Matrix to multiply
//   side: Apply from left or right
//   uplo: Upper or lower triangular storage
//   transpose: Apply transpose or not
//   work: Workspace array
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_from_tridiagonal :: proc(A: ^Matrix($T), tau: []T, C: ^Matrix(T), side: MultiplicationSide, uplo: MatrixTriangle, transpose: TransposeOperation, work: []T) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	assert(len(A.data) > 0 && len(C.data) > 0, "Matrices cannot be empty")
	assert(A.rows == A.cols, "A must be square for tridiagonal")
	assert(len(tau) >= A.rows - 1, "tau array too small")
	assert(len(work) > 0, "work array must be provided")

	side_c := cast(u8)side
	uplo_c := cast(u8)uplo
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	lda := A.ld
	ldc := C.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sormtr_(&side_c, &uplo_c, &trans_c, &m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dormtr_(&side_c, &uplo_c, &trans_c, &m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cunmtr_(&side_c, &uplo_c, &trans_c, &m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zunmtr_(&side_c, &uplo_c, &trans_c, &m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIANGULAR-PENTAGONAL QR FACTORIZATION (TPQRT family)
// ===================================================================================

// Query workspace size for triangular-pentagonal QR factorization
query_workspace_triangular_pentagonal_qr :: proc(A: ^Matrix($T), B: ^Matrix(T), nb: int = 32) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := B.rows
	n := B.cols
	l := min(A.rows, A.cols)
	nb_val := Blas_Int(nb)

	// Workspace size is n * nb for tpqrt
	lwork = int(n * nb_val)
	return lwork
}

// Triangular-pentagonal QR factorization (unified real/complex)
// Computes a QR factorization of a pentagonal matrix formed by coupling
// an n-by-n upper triangular tile A1 on top of an m-by-n pentagonal tile A2:
//
//     [ A1 ]  <-- n-by-n upper triangular
//     [ A2 ]  <-- m-by-n general
//
triangular_pentagonal_qr :: proc(
	A: ^Matrix($T), // Upper triangular matrix (n-by-n, overwritten with R)
	B: ^Matrix(T), // Pentagonal matrix (m-by-n, overwritten with Q info)
	T_out: ^Matrix(T), // T matrix for compact WY representation (nb-by-n, pre-allocated)
	work: []T, // Workspace (pre-allocated)
	l: int = 0, // Pentagonal width parameter (0 = min(m,n))
	nb: int = 32, // Block size
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := B.rows
	n := B.cols
	l_val := Blas_Int(l)
	if l == 0 {
		l_val = Blas_Int(min(int(m), int(n)))
	}
	nb_val := Blas_Int(nb)
	lda := A.ld
	ldb := B.ld
	ldt := T_out.ld

	assert(A.rows == A.cols, "A must be square")
	assert(A.rows == n, "A dimension mismatch")
	assert(T_out.rows >= int(min(nb_val, m)), "T_out rows too small")
	assert(T_out.cols >= int(n), "T_out cols too small")
	assert(len(work) >= int(n * nb_val), "work array too small")

	when T == f32 {
		lapack.stpqrt_(&m, &n, &l_val, &nb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == f64 {
		lapack.dtpqrt_(&m, &n, &l_val, &nb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == complex64 {
		lapack.ctpqrt_(&m, &n, &l_val, &nb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == complex128 {
		lapack.ztpqrt_(&m, &n, &l_val, &nb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MULTIPLY BY Q FROM TRIANGULAR-PENTAGONAL QR (TPMQRT family)
// ===================================================================================

// Query workspace size for multiplying by Q from triangular-pentagonal QR
query_workspace_multiply_triangular_pentagonal_qr :: proc(V: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, nb: int = 32) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := C.rows
	n := C.cols
	nb_val := Blas_Int(nb)

	switch side {
	case .Left:
		lwork = int(n * nb_val)
	case .Right:
		lwork = int(m * nb_val)
	}

	return lwork
}

// Multiply by Q from triangular-pentagonal QR (unified real/complex)
// Applies Q or Q**H from a triangular-pentagonal QR factorization to a matrix C
multiply_q_triangular_pentagonal_qr :: proc(
	V: ^Matrix($T), // Pentagonal matrix V from tpqrt (m-by-n)
	T_mat: ^Matrix(T), // T matrix from tpqrt (nb-by-n)
	A: ^Matrix(T), // Upper block of C (n-by-ncols or nrows-by-n)
	B: ^Matrix(T), // Lower block of C (m-by-ncols or nrows-by-m)
	work: []T, // Workspace (pre-allocated)
	side: MultiplicationSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = 0, // Number of elementary reflectors (0 = n)
	l: int = 0, // Pentagonal width parameter (0 = min(m,n))
	nb: int = 32, // Block size
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)trans
	m := B.rows
	n := B.cols
	k_val := Blas_Int(k)
	if k == 0 {
		k_val = Blas_Int(min(V.rows, V.cols))
	}
	l_val := Blas_Int(l)
	if l == 0 {
		l_val = Blas_Int(min(int(m), int(k_val)))
	}
	nb_val := Blas_Int(nb)
	ldv := V.ld
	ldt := T_mat.ld
	lda := A.ld
	ldb := B.ld

	assert(V.rows >= int(m), "V rows too small")
	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.stpmqrt_(&side_c, &trans_c, &m, &n, &k_val, &l_val, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == f64 {
		lapack.dtpmqrt_(&side_c, &trans_c, &m, &n, &k_val, &l_val, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex64 {
		lapack.ctpmqrt_(&side_c, &trans_c, &m, &n, &k_val, &l_val, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex128 {
		lapack.ztpmqrt_(&side_c, &trans_c, &m, &n, &k_val, &l_val, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIANGULAR-PENTAGONAL LQ FACTORIZATION (TPLQT family)
// ===================================================================================

// Query workspace size for triangular-pentagonal LQ factorization
query_workspace_triangular_pentagonal_lq :: proc(A: ^Matrix($T), B: ^Matrix(T), mb: int = 32) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := A.rows
	mb_val := Blas_Int(mb)

	// Workspace size is m * mb for tplqt
	lwork = int(m * mb_val)
	return lwork
}

// Triangular-pentagonal LQ factorization (unified real/complex)
// Computes an LQ factorization of a pentagonal matrix formed by coupling
// an m-by-m lower triangular tile A1 to the left of an m-by-n pentagonal tile A2:
//
//     [ A1  A2 ]  where A1 is m-by-m lower triangular, A2 is m-by-n general
//
triangular_pentagonal_lq :: proc(
	A: ^Matrix($T), // Lower triangular matrix (m-by-m, overwritten with L)
	B: ^Matrix(T), // Pentagonal matrix (m-by-n, overwritten with Q info)
	T_out: ^Matrix(T), // T matrix for compact WY representation (m-by-mb, pre-allocated)
	work: []T, // Workspace (pre-allocated)
	l: int = 0, // Pentagonal height parameter (0 = min(m,n))
	mb: int = 32, // Block size
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := B.cols
	l_val := Blas_Int(l)
	if l == 0 {
		l_val = min(int(m), int(n))
	}
	mb_val := Blas_Int(mb)
	lda := A.ld
	ldb := B.ld
	ldt := T_out.ld

	assert(A.rows == A.cols, "A must be square")
	assert(A.cols == m, "A dimension mismatch")
	assert(B.rows == m, "B rows must match A")
	assert(T_out.rows >= int(m), "T_out rows too small")
	assert(T_out.cols >= int(min(mb_val, n)), "T_out cols too small")
	assert(len(work) >= int(m * mb_val), "work array too small")

	when T == f32 {
		lapack.stplqt_(&m, &n, &l_val, &mb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == f64 {
		lapack.dtplqt_(&m, &n, &l_val, &mb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == complex64 {
		lapack.ctplqt_(&m, &n, &l_val, &mb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	} else when T == complex128 {
		lapack.ztplqt_(&m, &n, &l_val, &mb_val, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(T_out.data), &ldt, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MULTIPLY BY Q FROM TRIANGULAR-PENTAGONAL LQ (TPMLQT family)
// ===================================================================================

// Query workspace size for multiplying by Q from triangular-pentagonal LQ
query_workspace_multiply_triangular_pentagonal_lq :: proc(V: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, mb: int = 32) -> (lwork: int) where is_float(T) || is_complex(T) {
	m := C.rows
	n := C.cols
	mb_val := Blas_Int(mb)

	switch side {
	case .Left:
		lwork = int(n * mb_val)
	case .Right:
		lwork = int(m * mb_val)
	}

	return lwork
}

// Multiply by Q from triangular-pentagonal LQ (unified real/complex)
// Applies Q or Q**H from a triangular-pentagonal LQ factorization to a matrix C
multiply_q_triangular_pentagonal_lq :: proc(
	V: ^Matrix($T), // Pentagonal matrix V from tplqt (m-by-n)
	T_mat: ^Matrix(T), // T matrix from tplqt (m-by-mb)
	A: ^Matrix(T), // Left block of C (nrows-by-m or m-by-ncols)
	B: ^Matrix(T), // Right block of C (nrows-by-n or n-by-ncols)
	work: []T, // Workspace (pre-allocated)
	side: MultiplicationSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = 0, // Number of elementary reflectors (0 = m)
	l: int = 0, // Pentagonal height parameter (0 = min(m,n))
	mb: int = 32, // Block size
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)trans
	m := V.rows
	n := V.cols
	k_val := Blas_Int(k)
	if k == 0 {
		k_val = min(V.rows, V.cols)
	}
	l_val := Blas_Int(l)
	if l == 0 {
		l_val = min(m, n)
	}
	mb_val := Blas_Int(mb)
	ldv := V.ld
	ldt := T_mat.ld
	lda := A.ld
	ldb := B.ld

	// Use dimensions from B for the actual operation
	m_op := B.rows
	n_op := B.cols

	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.stpmlqt_(&side_c, &trans_c, &m_op, &n_op, &k_val, &l_val, &mb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == f64 {
		lapack.dtpmlqt_(&side_c, &trans_c, &m_op, &n_op, &k_val, &l_val, &mb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex64 {
		lapack.ctpmlqt_(&side_c, &trans_c, &m_op, &n_op, &k_val, &l_val, &mb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	} else when T == complex128 {
		lapack.ztpmlqt_(&side_c, &trans_c, &m_op, &n_op, &k_val, &l_val, &mb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(work), &info)
	}

	return info, info == 0
}


// ===================================================================================
// WORKSPACE QUERIES FOR COMPACT WY
// ===================================================================================

// Query workspace for QR factorization with compact WY
query_workspace_qr_compact :: proc(m, n: int, nb: int = 32) -> (lwork: int) {
	// Workspace for geqrt is nb*n
	return nb * n
}

// Query workspace for multiplying by Q from compact WY QR
query_workspace_multiply_compact_qr :: proc(m, n: int, nb: int = 32) -> (lwork: int) {
	// Workspace for gemqrt
	return max(m, n) * nb
}

// ===================================================================================
// QR FACTORIZATION WITH COMPACT WY (GEQRT)
// ===================================================================================

// QR factorization using compact WY representation (GEQRT)
//
// Computes a QR factorization of a real/complex m-by-n matrix A using the
// compact WY representation of Q. This is more efficient than the classical
// representation for blocked algorithms.
//
// Parameters:
//   A: On entry, the m-by-n matrix to factor. On exit, upper triangle contains R,
//      lower triangle contains elementary reflectors
//   T: Output matrix storing the triangular factors of the block reflectors (nb-by-n)
//   work: Workspace array (length nb*n)
//   nb: Block size (default 32)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
qr_factorization_compact :: proc(A: ^Matrix($T), Tmat: ^Matrix(T), work: []T, nb: int = 32) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nb_val := Blas_Int(nb)
	lda := A.ld
	ldt := Tmat.ld

	assert(Tmat.rows >= nb, "T matrix must have at least nb rows")
	assert(Tmat.cols >= min(int(m), int(n)), "T matrix must have at least min(m,n) columns")
	assert(len(work) >= nb * int(n), "work array too small")

	when T == f32 {
		lapack.sgeqrt_(&m, &n, &nb_val, raw_data(A.data), &lda, raw_data(Tmat.data), &ldt, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgeqrt_(&m, &n, &nb_val, raw_data(A.data), &lda, raw_data(Tmat.data), &ldt, raw_data(work), &info)
	} else when T == complex64 {
		lapack.cgeqrt_(&m, &n, &nb_val, raw_data(A.data), &lda, raw_data(Tmat.data), &ldt, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zgeqrt_(&m, &n, &nb_val, raw_data(A.data), &lda, raw_data(Tmat.data), &ldt, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MULTIPLY BY Q FROM COMPACT WY QR (GEMQRT)
// ===================================================================================

// Multiply by Q from compact WY QR factorization (GEMQRT)
//
// Applies the orthogonal/unitary matrix Q from a QR factorization computed by GEQRT
// to a matrix C using the compact WY representation.
//
// Computes:
//   - C := Q * C    (side = .Left, trans = .None)
//   - C := Q**T * C (side = .Left, trans = .Transpose for real, .ConjugateTranspose for complex)
//   - C := C * Q    (side = .Right, trans = .None)
//   - C := C * Q**T (side = .Right, trans = .Transpose/.ConjugateTranspose)
//
// Parameters:
//   V: Matrix containing elementary reflectors from GEQRT (m-by-k if side=Left, n-by-k if side=Right)
//   T: Triangular factors of block reflectors from GEQRT (nb-by-k)
//   C: Matrix to multiply (m-by-n)
//   work: Workspace array
//   side: Multiply from left or right
//   trans: Apply Q or Q**T/Q**H
//   nb: Block size (must match value used in GEQRT, default 32)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_compact_qr :: proc(V: ^Matrix($T), T_mat: ^Matrix(T), C: ^Matrix(T), work: []T, side: MultiplicationSide = .Left, trans: MatrixTranspose = .None, nb: int = 32) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)trans
	m := C.rows
	n := C.cols
	k := min(V.rows, V.cols)
	nb_val := Blas_Int(nb)
	ldv := V.ld
	ldt := T_mat.ld
	ldc := C.ld

	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.sgemqrt_(&side_c, &trans_c, &m, &n, &k, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgemqrt_(&side_c, &trans_c, &m, &n, &k, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == complex64 {
		lapack.cgemqrt_(&side_c, &trans_c, &m, &n, &k, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zgemqrt_(&side_c, &trans_c, &m, &n, &k, &nb_val, raw_data(V.data), &ldv, raw_data(T_mat.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERIES FOR FLEXIBLE/RECURSIVE
// ===================================================================================

// Query workspace for flexible QR multiplication
query_workspace_multiply_flexible_qr :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k := min(A.rows, A.cols)
	lda := A.ld
	tsize := QUERY_WORKSPACE
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T
	tsize_query: T

	when T == f32 {
		lapack.sgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// Query workspace for flexible LQ multiplication
query_workspace_multiply_flexible_lq :: proc(A: ^Matrix($T), C: ^Matrix(T), side: MultiplicationSide, transpose: TransposeOperation) -> (lwork: int) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)transpose
	m := C.rows
	n := C.cols
	k := min(A.rows, A.cols)
	lda := A.ld
	tsize := QUERY_WORKSPACE
	ldc := C.ld

	info: Info
	lwork_query := QUERY_WORKSPACE
	work_query: T
	tsize_query: T

	when T == f32 {
		lapack.sgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, &tsize_query, &tsize, nil, &ldc, &work_query, &lwork_query, &info)
	}

	when is_float(T) {
		return int(work_query)
	} else {
		return int(real(work_query))
	}
}

// ===================================================================================
// MULTIPLY BY Q FROM FLEXIBLE QR (GEMQR)
// ===================================================================================

// Multiply by Q from flexible/recursive QR factorization (GEMQR)
//
// Applies the orthogonal/unitary matrix Q from a flexible QR factorization (GEQR)
// to a matrix C. This is an improved version that handles various matrix shapes
// efficiently using recursive algorithms.
//
// Computes:
//   - C := Q * C    (side = .Left, trans = .None)
//   - C := Q**T * C (side = .Left, trans = .Transpose for real, .ConjugateTranspose for complex)
//   - C := C * Q    (side = .Right, trans = .None)
//   - C := C * Q**T (side = .Right, trans = .Transpose/.ConjugateTranspose)
//
// Parameters:
//   A: Matrix containing elementary reflectors from GEQR
//   T: Householder reflectors in compact form from GEQR (tsize elements)
//   C: Matrix to multiply (m-by-n)
//   work: Workspace array (use query_workspace_multiply_flexible_qr to determine size)
//   side: Multiply from left or right
//   trans: Apply Q or Q**T/Q**H
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_flexible_qr :: proc(A: ^Matrix($T), T_array: []T, C: ^Matrix(T), work: []T, side: MultiplicationSide = .Left, trans: MatrixTranspose = .None) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)trans
	m := C.rows
	n := C.cols
	k := min(A.rows, A.cols)
	lda := A.ld
	tsize := Blas_Int(len(T_array))
	ldc := C.ld
	lwork := Blas_Int(len(work))

	assert(len(T_array) > 0, "T array must be provided")
	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.sgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgemqr_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// MULTIPLY BY Q FROM FLEXIBLE LQ (GEMLQ)
// ===================================================================================

// Multiply by Q from flexible/recursive LQ factorization (GEMLQ)
//
// Applies the orthogonal/unitary matrix Q from a flexible LQ factorization (GELQ)
// to a matrix C. This is an improved version that handles various matrix shapes
// efficiently using recursive algorithms.
//
// Computes:
//   - C := Q * C    (side = .Left, trans = .None)
//   - C := Q**T * C (side = .Left, trans = .Transpose for real, .ConjugateTranspose for complex)
//   - C := C * Q    (side = .Right, trans = .None)
//   - C := C * Q**T (side = .Right, trans = .Transpose/.ConjugateTranspose)
//
// Parameters:
//   A: Matrix containing elementary reflectors from GELQ
//   T: Householder reflectors in compact form from GELQ (tsize elements)
//   C: Matrix to multiply (m-by-n)
//   work: Workspace array (use query_workspace_multiply_flexible_lq to determine size)
//   side: Multiply from left or right
//   trans: Apply Q or Q**T/Q**H
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_flexible_lq :: proc(A: ^Matrix($T), T_array: []T, C: ^Matrix(T), work: []T, side: MultiplicationSide = .Left, trans: MatrixTranspose = .None) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	side_c := cast(u8)side
	trans_c := cast(u8)trans
	m := C.rows
	n := C.cols
	k := min(A.rows, A.cols)
	lda := A.ld
	tsize := Blas_Int(len(T_array))
	ldc := C.ld
	lwork := Blas_Int(len(work))

	assert(len(T_array) > 0, "T array must be provided")
	assert(len(work) > 0, "work array must be provided")

	when T == f32 {
		lapack.sgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgemlq_(&side_c, &trans_c, &m, &n, &k, raw_data(A.data), &lda, raw_data(T_array), &tsize, raw_data(C.data), &ldc, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERIES
// ===================================================================================

// Query workspace for generating Q from packed Householder reflectors
query_workspace_generate_q_packed :: proc(n: int) -> (lwork: int) {
	// For opgtr/upgtr, workspace is (n-1) elements
	return max(1, n - 1)
}

// Query workspace for multiplying by Q from packed Householder
query_workspace_multiply_q_packed :: proc(m, n: int, side: MultiplicationSide) -> (lwork: int) {
	// For opmtr/upmtr, workspace depends on which dimension is operated on
	switch side {
	case .Left:
		return max(1, n)
	case .Right:
		return max(1, m)
	}
	return 1
}

// ===================================================================================
// GENERATE Q FROM PACKED HOUSEHOLDER (OPGTR/UPGTR - Unified)
// ===================================================================================

// Generate orthogonal/unitary matrix Q from packed symmetric/Hermitian Householder reflectors
//
// This generates the orthogonal/unitary matrix Q from elementary reflectors returned by
// the reduction of a symmetric/Hermitian matrix to tridiagonal form using packed storage.
//
// Parameters:
//   AP: Packed symmetric/Hermitian matrix containing elementary reflectors (n*(n+1)/2 elements)
//   tau: Scalar factors of elementary reflectors (length n-1)
//   Q: Output orthogonal/unitary matrix Q (n-by-n)
//   work: Workspace array (length n-1)
//   uplo: Whether AP contains upper or lower triangle
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
generate_q_packed :: proc(AP: []$T, tau: []T, Q: ^Matrix(T), work: []T, uplo: MatrixRegion = .Upper) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	n := Q.rows
	assert(Q.rows == Q.cols, "Q must be square")
	assert(len(AP) >= n * (n + 1) / 2, "AP array too small for packed storage")
	assert(len(tau) >= n - 1, "tau array too small")
	assert(len(work) >= n - 1, "work array too small")

	uplo_c := cast(u8)uplo
	n_val := Blas_Int(n)
	ldq := Q.ld

	when T == f32 {
		lapack.sopgtr_(&uplo_c, &n_val, raw_data(AP), raw_data(tau), raw_data(Q.data), &ldq, raw_data(work), &info)
	} else when T == f64 {
		lapack.dopgtr_(&uplo_c, &n_val, raw_data(AP), raw_data(tau), raw_data(Q.data), &ldq, raw_data(work), &info)
	} else when T == complex64 {
		lapack.cupgtr_(&uplo_c, &n_val, raw_data(AP), raw_data(tau), raw_data(Q.data), &ldq, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zupgtr_(&uplo_c, &n_val, raw_data(AP), raw_data(tau), raw_data(Q.data), &ldq, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MULTIPLY BY Q FROM PACKED HOUSEHOLDER (OPMTR/UPMTR - Unified)
// ===================================================================================

// Multiply by orthogonal/unitary matrix Q from packed symmetric/Hermitian Householder reflectors
//
// This multiplies a matrix C by the orthogonal/unitary matrix Q from elementary reflectors
// returned by the reduction of a symmetric/Hermitian matrix to tridiagonal form using packed storage.
//
// Computes:
//   - C := Q * C      (side = .Left, trans = .None)
//   - C := Q**T/H * C (side = .Left, trans = .Transpose/.ConjugateTranspose)
//   - C := C * Q      (side = .Right, trans = .None)
//   - C := C * Q**T/H (side = .Right, trans = .Transpose/.ConjugateTranspose)
//
// Parameters:
//   AP: Packed symmetric/Hermitian matrix containing elementary reflectors (n*(n+1)/2 elements)
//   tau: Scalar factors of elementary reflectors (length n-1)
//   C: Matrix to multiply (m-by-n)
//   work: Workspace array (length n if side=Left, m if side=Right)
//   side: Multiply from left or right
//   uplo: Whether AP contains upper or lower triangle
//   trans: Apply Q or Q**T (Q**H for complex)
//
// Returns:
//   info: 0 on success, < 0 if argument i had illegal value
//   ok: true if successful
multiply_q_packed :: proc(AP: []$T, tau: []T, C: ^Matrix(T), work: []T, side: MultiplicationSide = .Left, uplo: MatrixRegion = .Upper, trans: MatrixTranspose = .None) -> (info: Info, ok: bool) where is_float(T) || is_complex(T) {
	m := C.rows
	n := C.cols
	nq := side == .Left ? m : n // Dimension of Q

	assert(len(AP) >= nq * (nq + 1) / 2, "AP array too small for packed storage")
	assert(len(tau) >= nq - 1, "tau array too small")

	work_required := side == .Left ? n : m
	assert(len(work) >= work_required, "work array too small")

	side_c := cast(u8)side
	uplo_c := cast(u8)uplo
	trans_c := cast(u8)trans
	m_val := Blas_Int(m)
	n_val := Blas_Int(n)
	ldc := C.ld

	when T == f32 {
		lapack.sopmtr_(&side_c, &uplo_c, &trans_c, &m_val, &n_val, raw_data(AP), raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dopmtr_(&side_c, &uplo_c, &trans_c, &m_val, &n_val, raw_data(AP), raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == complex64 {
		lapack.cupmtr_(&side_c, &uplo_c, &trans_c, &m_val, &n_val, raw_data(AP), raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &info)
	} else when T == complex128 {
		lapack.zupmtr_(&side_c, &uplo_c, &trans_c, &m_val, &n_val, raw_data(AP), raw_data(tau), raw_data(C.data), &ldc, raw_data(work), &info)
	}

	return info, info == 0
}
