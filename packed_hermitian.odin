package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED HERMITIAN MATRIX BASE TYPES AND CORE OPERATIONS
// ===================================================================================
//
// This file provides the base types and core operations for packed Hermitian
// matrices. Hermitian matrices satisfy A* = A (conjugate transpose equals original).
// Packed storage stores only the upper or lower triangle in a 1D array.
//
// For Hermitian matrices:
// - Diagonal elements are always real
// - Off-diagonal elements satisfy A[i,j] = conj(A[j,i])
// - Only one triangle needs to be stored
//
// For conversion operations, see packed_hermitian_conversion.odin

// ===================================================================================
// BASE TYPES
// ===================================================================================

// Packed Hermitian matrix type (complex types only)
PackedHermitian :: struct($T: typeid) where is_complex(T) {
    data: []T, // Packed storage array (n*(n+1)/2 elements)
    n:    int, // Matrix dimension
    uplo: MatrixRegion, // Storage region (Upper or Lower)
}
// ===================================================================================
// PACKED MATRIX ELEMENT ACCESS
// ===================================================================================

// Get element from packed Hermitian matrix
pack_herm_get :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i, j: int, // Element indices
    uplo: MatrixRegion = .Upper,
) -> T where is_complex(T) {
    assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

    // Determine if we need to access stored triangle or compute conjugate
    row, col := i, j
    conjugate_result := false

    if uplo == .Upper {
        if i > j {
            row, col = j, i // Access upper triangle
            conjugate_result = true
        }
    } else if uplo == .Lower {
        if i < j {
            row, col = j, i // Access lower triangle
            conjugate_result = true
        }
    }

    // Calculate packed index
    idx: int
    switch uplo {
    case .Upper:
        idx = row + col * (col + 1) / 2
    case .Lower:
        idx = (row - col) + col * (2 * n - col - 1) / 2
    case .Full:
        panic("Full storage not supported for packed format")
    }

    val := AP[idx]

    // Apply conjugate if accessing mirrored element
    if conjugate_result && i != j {
        return conj(val)
    }

    return val
}

// Set element in packed Hermitian matrix
pack_herm_set :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i, j: int, // Element indices
    val: T, // Value to set
    uplo: MatrixRegion = .Upper,
) where is_complex(T) {
    assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

    // For diagonal elements, ensure they are real (Hermitian property)
    if i == j {
        // Diagonal elements must be real in Hermitian matrices
        real_val := real(val)
        val = complex(real_val, 0)
    }

    // Only set elements in the stored triangle
    should_set := false
    switch uplo {
    case .Upper:
        should_set = i <= j
    case .Lower:
        should_set = i >= j
    case .Full:
        panic("Full storage not supported for packed format")
    }

    if !should_set {
        return // Don't set elements outside stored triangle
    }

    // Calculate packed index
    idx: int
    switch uplo {
    case .Upper:
        idx = i + j * (j + 1) / 2
    case .Lower:
        idx = (i - j) + j * (2 * n - j - 1) / 2
    case .Full:
        panic("Full storage not supported for packed format")
    }

    AP[idx] = val
}

// ===================================================================================
// UTILITY FUNCTIONS
// ===================================================================================
// Copy packed Hermitian matrix
copy_packed_hermitian :: proc(
    src: ^PackedHermitian($T),
    allocator := context.allocator,
) -> PackedHermitian(T) where is_complex(T) {
    data := make([]T, len(src.data), allocator)
    copy(data, src.data)

    return PackedHermitian(T){data = data, n = src.n, uplo = src.uplo}
}

// Delete packed Hermitian matrix
delete_packed_hermitian :: proc(packed: ^PackedHermitian($T)) where is_complex(T) {
    if packed.data != nil {
        delete(packed.data)
        packed.data = nil
    }
}

// Validate packed Hermitian matrix consistency
validate_packed_hermitian :: proc(packed: ^PackedHermitian($T)) -> bool where is_complex(T) {
    if packed.n <= 0 {
        return false
    }

    expected_size := packed_storage_size(packed.n)
    if len(packed.data) < expected_size {
        return false
    }

    switch packed.uplo {
    case .Upper, .Lower:
        // Check that diagonal elements are real (Hermitian property)
        for i in 0 ..< packed.n {
            diag_val := pack_herm_get(packed.data, packed.n, i, i, packed.uplo)
            if abs(imag(diag_val)) > 1e-12 {     // Allow for small numerical errors
                return false
            }
        }
        return true
    case .Full:
        return false
    }

    return false
}

// ===================================================================================
// HERMITIAN-SPECIFIC OPERATIONS
// ===================================================================================

// Check if a packed Hermitian matrix appears to be positive definite
// (diagonal elements > 0 and real)
is_packed_hermitian_positive_definite_heuristic :: proc(packed: ^PackedHermitian($T)) -> bool where is_complex(T) {
    for i in 0 ..< packed.n {
        diag_val := pack_herm_get(packed.data, packed.n, i, i, packed.uplo)
        // Diagonal should be real and positive
        if real(diag_val) <= 0 || abs(imag(diag_val)) > 1e-12 {
            return false
        }
    }
    return true
}

// Get diagonal element efficiently (always real for Hermitian matrices)
pack_herm_diagonal_get :: proc(
    AP: []$Cmplx, // Packed array
    n: int, // Matrix dimension
    i: int, // Diagonal index
    result_out: ^$Real, // Output: diagonal element value
    uplo: MatrixRegion = .Upper,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
    assert(i >= 0 && i < n, "Diagonal index out of bounds")

    idx: int
    switch uplo {
    case .Upper:
        idx = i + i * (i + 1) / 2
    case .Lower:
        idx = i * (2 * n - i - 1) / 2
    case .Full:
        panic("Full storage not supported for packed format")
    }

    // Diagonal elements are always real in Hermitian matrices
    result_out^ = real(AP[idx])
}

// Set diagonal element efficiently (automatically makes it real)
pack_herm_diagonal_set :: proc(
    AP: []$Cmplx, // Packed array
    n: int, // Matrix dimension
    i: int, // Diagonal index
    val: $Real, // Real value to set
    uplo: MatrixRegion = .Upper,
) where Cmplx == complex64 || Cmplx == complex128,
    Real == real_type_of(Cmplx) {
    assert(i >= 0 && i < n, "Diagonal index out of bounds")

    idx: int
    switch uplo {
    case .Upper:
        idx = i + i * (i + 1) / 2
    case .Lower:
        idx = i * (2 * n - i - 1) / 2
    case .Full:
        panic("Full storage not supported for packed format")
    }

    // Store as real value (imaginary part = 0)
    AP[idx] = complex(val, 0)
}
