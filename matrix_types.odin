package openblas

import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// SPECIALIZED MATRIX STORAGE TYPES
// Different storage formats for efficient memory usage and cache performance
// ===================================================================================

// Standard dense matrix (already defined in matrix.odin)
// Included here for completeness of the type system
// Matrix(T) - full storage, row or column major

// ===================================================================================
// PACKED MATRIX - Triangular/Symmetric/Hermitian Storage
// Stores only upper or lower triangle, ~50% memory savings
// ===================================================================================

PackedMatrix :: struct($T: typeid) where is_float(T) || is_complex(T) {
	data:      []T, // Packed storage array, size n*(n+1)/2
	n:         int, // Matrix dimension (n×n)
	uplo:      UpLo, // Upper or Lower triangle stored
	// For symmetric/Hermitian matrices
	symmetric: bool, // True if matrix is symmetric (real) or Hermitian (complex)
}

// Create a packed matrix from dimensions
make_packed_matrix :: proc(n: int, uplo: UpLo, $T: typeid, allocator := context.allocator) -> PackedMatrix(T) {
	size := n * (n + 1) / 2
	return PackedMatrix(T){n = n, data = make([]T, size, allocator), uplo = uplo}
}

// Access element (i,j) in packed storage
// Column-major upper: AP[i + j*(j+1)/2] for i <= j
// Column-major lower: AP[i + (2*n-j-1)*j/2] for i >= j
packed_index :: proc(pm: ^PackedMatrix($T), i, j: int) -> int {
	assert(i >= 0 && i < pm.n && j >= 0 && j < pm.n, "Index out of bounds")

	if pm.uplo == .Upper {
		if i <= j {
			return i + j * (j + 1) / 2
		} else if pm.symmetric {
			return j + i * (i + 1) / 2 // Use symmetry
		}
	} else { 	// Lower
		if i >= j {
			return i + (2 * pm.n - j - 1) * j / 2
		} else if pm.symmetric {
			return j + (2 * pm.n - i - 1) * i / 2 // Use symmetry
		}
	}

	panic("Accessing unstored element in packed matrix")
}

// ===================================================================================
// BANDED MATRIX - Efficient storage for sparse banded systems
// ===================================================================================

BandedMatrix :: struct($T: typeid) where is_float(T) || is_complex(T) {
	data:      []T, // Band storage array, size ldab × cols
	rows:      Blas_Int, // Number of rows
	cols:      Blas_Int, // Number of columns
	kl:        Blas_Int, // Number of subdiagonals (lower bandwidth)
	ku:        Blas_Int, // Number of superdiagonals (upper bandwidth)
	ldab:      Blas_Int, // Leading dimension of band storage (>= kl + ku + 1)

	// For symmetric banded matrices
	symmetric: bool, // True if symmetric/Hermitian (only need ku or kl)
}

// Create a banded matrix
make_banded_matrix :: proc(rows, cols, kl, ku: int, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := kl + ku + 1
	return BandedMatrix(T){rows = Blas_Int(rows), cols = Blas_Int(cols), kl = Blas_Int(kl), ku = Blas_Int(ku), ldab = Blas_Int(ldab), data = make([]T, ldab * cols, allocator)}
}

// Create symmetric banded matrix (only stores upper or lower band)
make_symmetric_banded_matrix :: proc(n, k: int, uplo: UpLo, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := k + 1 // For symmetric, only need k+1 bands
	bm := BandedMatrix(T) {
		rows      = n,
		cols      = n,
		symmetric = true,
		ldab      = ldab,
		data      = make([]T, ldab * n, allocator),
	}

	if uplo == .Upper {
		bm.ku = k
		bm.kl = 0
	} else {
		bm.kl = k
		bm.ku = 0
	}

	return bm
}

// Access element (i,j) in banded storage
// Band storage format (column-major):
// AB(ku+1+i-j, j) = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl)
banded_index :: proc(bm: ^BandedMatrix($T), i, j: int) -> (idx: int, stored: bool) {
	assert(i >= 0 && i < bm.rows && j >= 0 && j < bm.cols, "Index out of bounds")

	// Check if element is within the band
	if i > j + bm.kl || i < j - bm.ku {
		return -1, false
	}

	// Column-major band storage
	idx = bm.ku + i - j + j * bm.ldab
	return idx, true
}

// ===================================================================================
// TRIDIAGONAL MATRIX - Special case of banded with kl=ku=1
// ===================================================================================

TridiagonalMatrix :: struct($T: typeid) where is_float(T) || is_complex(T) {
	n:         Blas_Int, // Matrix dimension (n×n)
	dl:        []T, // Subdiagonal, size n-1
	d:         []T, // Main diagonal, size n
	du:        []T, // Superdiagonal, size n-1

	// For symmetric tridiagonal (du = dl for real, du = conj(dl) for complex)
	symmetric: bool,
}

// Create a tridiagonal matrix
make_tridiagonal_matrix :: proc(n: int, $T: typeid, allocator := context.allocator) -> TridiagonalMatrix(T) {
	return TridiagonalMatrix(T){n = Blas_Int(n), dl = make([]T, n - 1, allocator), d = make([]T, n, allocator), du = make([]T, n - 1, allocator)}
}

// Create symmetric tridiagonal matrix (du is not allocated for real, or computed from dl for complex)
make_symmetric_tridiagonal_matrix :: proc(n: int, $T: typeid, allocator := context.allocator) -> TridiagonalMatrix(T) {
	tm := TridiagonalMatrix(T) {
		n         = Blas_Int(n),
		d         = make([]T, n, allocator),
		dl        = make([]T, n - 1, allocator),
		symmetric = true,
	}

	when is_float(T) {
		tm.du = tm.dl
	} else {
		// For complex Hermitian, du = conj(dl) (need separate storage)
		tm.du = make([]T, n - 1, allocator)
	}

	return tm
}

// Access element (i,j) in tridiagonal matrix
tridiagonal_get :: proc(tm: ^TridiagonalMatrix($T), i, j: int) -> (value: T, stored: bool) {
	assert(i >= 0 && i < tm.n && j >= 0 && j < tm.n, "Index out of bounds")

	if i == j {
		return tm.d[i], true
	} else if i == j + 1 && i > 0 {
		return tm.dl[j], true
	} else if i == j - 1 && j > 0 {
		when T == complex64 || T == complex128 {
			if tm.symmetric {
				return conj(tm.dl[i]), true // Hermitian case
			}
		}
		return tm.du[i], true
	}

	return T{}, false
}

// Set element (i,j) in tridiagonal matrix
tridiagonal_set :: proc(tm: ^TridiagonalMatrix($T), i, j: int, value: T) -> bool {
	assert(i >= 0 && i < tm.n && j >= 0 && j < tm.n, "Index out of bounds")

	if i == j {
		tm.d[i] = value
		return true
	} else if i == j + 1 && i > 0 {
		tm.dl[j] = value
		if tm.symmetric {
			when is_float(T) {
				// Real symmetric: du = dl
			} else {
				// Complex Hermitian: du = conj(dl)
				tm.du[j] = conj(value)
			}
		}
		return true
	} else if i == j - 1 && j > 0 {
		tm.du[i] = value
		if tm.symmetric {
			when is_float(T) {
				tm.dl[i] = value // Real symmetric
			} else {
				tm.dl[i] = conj(value) // Complex Hermitian
			}
		}
		return true
	}

	return false // Element not in tridiagonal structure
}

// ===================================================================================
// RFP (RECTANGULAR FULL PACKED) FORMAT
// Hybrid format combining benefits of packed and full storage
// Better cache performance than traditional packed storage
// ===================================================================================

RFPMatrix :: struct($T: typeid) where is_float(T) || is_complex(T) {
	data:        []T, // RFP storage array, size n*(n+1)/2
	n:           Blas_Int, // Matrix dimension (n×n)
	trans_state: TransposeState, // RFP format variant (Normal or Transpose/Conjugate)
	uplo:        UpLo, // Upper or Lower triangle of original matrix
}

// Create an RFP matrix
new_rfp_matrix :: proc(n: int, trans_state: TransposeState, uplo: UpLo, $T: typeid, allocator := context.allocator) -> RFPMatrix(T) {
	size := n * (n + 1) / 2
	return RFPMatrix(T){n = Blas_Int(n), data = make([]T, size, allocator), trans_state = trans_state, uplo = uplo}
}

// RFP storage is complex - the mapping depends on:
// - n even or odd
// - transr = 'N' or 'T'/'C'
// - uplo = 'U' or 'L'
// This creates 8 different storage schemes
rfp_index :: proc(rm: ^RFPMatrix($T), i, j: int) -> (idx: int, stored: bool) {
	assert(i >= 0 && i < rm.n && j >= 0 && j < rm.n, "Index out of bounds")

	n := rm.n
	k := n / 2

	// RFP format is quite complex, this is a simplified version
	// Full implementation would need all 8 cases
	// See LAPACK working note 199 for complete details

	if rm.transr == .Normal {
		if rm.uplo == .Upper {
			// Normal, Upper case
			if n % 2 == 0 {
				// Even n, Normal, Upper
				// ... complex indexing formula
			} else {
				// Odd n, Normal, Upper
				// ... complex indexing formula
			}
		} else {
			// Normal, Lower case
			// ... similar complex logic
		}
	} else {
		// Transposed cases
		// ... more complex logic
	}

	// Placeholder - would need full implementation
	return -1, false
}

// ===================================================================================
// PROPERTY ENUMS (shared across matrix types)
// ===================================================================================

// Triangle specification
UpLo :: enum u8 {
	Upper = 'U', // Upper triangle
	Lower = 'L', // Lower triangle
}

// Diagonal type for triangular matrices
Diag :: enum u8 {
	NonUnit = 'N', // Diagonal elements are stored
	Unit    = 'U', // Diagonal elements are assumed to be 1
}

// Side for operations like triangular solve
Side :: enum u8 {
	Left  = 'L', // Operation on the left
	Right = 'R', // Operation on the right
}

// Transpose operations
TransposeState :: enum u8 {
	NoTrans   = 'N', // No transpose
	Trans     = 'T', // Transpose
	ConjTrans = 'C', // Conjugate transpose (Hermitian transpose)
}

// ===================================================================================
// CONVERSION UTILITIES
// ===================================================================================

// Convert packed matrix to full matrix
packed_to_full :: proc(pm: ^PackedMatrix($T), allocator := context.allocator) -> Matrix(T) {
	// Create a standard matrix - assuming column-major for LAPACK compatibility
	m := Matrix(T) {
		rows   = Blas_Int(pm.n),
		cols   = Blas_Int(pm.n),
		data   = make([]T, pm.n * pm.n, allocator),
		ld     = Blas_Int(pm.n),
		format = .General,
	}

	for j in 0 ..< pm.n {
		for i in 0 ..< pm.n {
			if (pm.uplo == .Upper && i <= j) || (pm.uplo == .Lower && i >= j) {
				idx := packed_index(pm, i, j)
				// Column-major indexing
				m.data[i + j * pm.n] = pm.data[idx]
				if pm.symmetric && i != j {
					m.data[j + i * pm.n] = pm.data[idx] // Fill symmetric element
				}
			}
		}
	}

	return m
}

// Convert full matrix to packed format (symmetric/triangular only)
full_to_packed :: proc(m: ^Matrix($T), uplo: UpLo, check_symmetric := false, allocator := context.allocator) -> PackedMatrix(T) {
	assert(m.rows == m.cols, "Matrix must be square for packed storage")

	pm := new_packed_matrix(m.rows, uplo, T, allocator)

	if check_symmetric {
		pm.symmetric = is_symmetric(m)
	}

	for j in 0 ..< m.cols {
		for i in 0 ..< m.rows {
			if (uplo == .Upper && i <= j) || (uplo == .Lower && i >= j) {
				idx := packed_index(&pm, i, j)
				// Column-major indexing
				pm.data[idx] = m.data[i + j * m.cols]
			}
		}
	}

	return pm
}
