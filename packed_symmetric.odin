package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED SYMMETRIC MATRIX BASE TYPES AND UTILITIES
// ===================================================================================
//
// This file provides the base types and core utilities for packed symmetric
// matrices. Packed storage stores only the upper or lower triangle of symmetric
// matrices in a 1D array, saving approximately 50% memory.
//
// For conversion operations between packed and full formats, see:
// - packed_symmetric_conversion.odin
//
// Packed storage format:
// - Upper triangle: A[i,j] stored at AP[i + j*(j+1)/2] for i <= j
// - Lower triangle: A[i,j] stored at AP[i + (2*n-j-1)*j/2] for i >= j
//
// The packed format requires n*(n+1)/2 elements for an n×n symmetric matrix.

// ===================================================================================
// BASE TYPES
// ===================================================================================

// Packed symmetric matrix type
PackedSymmetric :: struct($T: typeid) {
    data: []T, // Packed storage array (n*(n+1)/2 elements)
    n:    int, // Matrix dimension
    uplo: MatrixRegion, // Storage region (Upper or Lower)
}

// Storage layout validation
validate_packed_storage :: proc(n: int, data_len: int) -> bool {
    expected_len := n * (n + 1) / 2
    return data_len >= expected_len
}

// Calculate packed storage size
packed_storage_size :: proc(n: int) -> int {
    return n * (n + 1) / 2
}

// ===================================================================================
// PACKED MATRIX CREATION AND MANAGEMENT
// ===================================================================================

// Create empty PackedSymmetric matrix
pack_sym_make :: proc(
    $T: typeid,
    n: int,
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> PackedSymmetric(T) {
    assert(n > 0, "Matrix dimension must be positive")

    packed_size := packed_storage_size(n)
    data := make([]T, packed_size, allocator)

    return PackedSymmetric(T){data = data, n = n, uplo = uplo}
}

// Create PackedSymmetric matrix initialized to zero
pack_sym_make_zero :: proc(
    $T: typeid,
    n: int,
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> PackedSymmetric(T) {
    result := pack_sym_make(T)(n, uplo, allocator)

    // Initialize to zero
    for i in 0 ..< len(result.data) {
        result.data[i] = T(0)
    }

    return result
}

// Create PackedSymmetric identity matrix
pack_sym_make_identity :: proc(
    $T: typeid,
    n: int,
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> PackedSymmetric(T) {
    result := pack_sym_make_zero(T)(n, uplo, allocator)

    // Set diagonal elements to 1
    for i in 0 ..< n {
        pack_sym_diagonal_set(result.data, n, i, T(1), uplo)
    }

    return result
}

// ===================================================================================
// PACKED MATRIX ELEMENT ACCESS
// ===================================================================================

// Get element from packed symmetric matrix
pack_sym_get :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i, j: int, // Element indices
    uplo: MatrixRegion = .Upper,
) -> T {
    assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

    // Ensure we access the stored triangle
    row, col := i, j
    if uplo == .Upper && i > j {
        row, col = j, i // Access upper triangle
    } else if uplo == .Lower && i < j {
        row, col = j, i // Access lower triangle
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

    return AP[idx]
}

// Set element in packed symmetric matrix
pack_sym_set :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i, j: int, // Element indices
    val: T, // Value to set
    uplo: MatrixRegion = .Upper,
) {
    assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

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
// MEMORY MANAGEMENT
// ===================================================================================

// Copy packed matrix
pack_sym_copy :: proc(src: ^PackedSymmetric($T), allocator := context.allocator) -> PackedSymmetric(T) {
    data := make([]T, len(src.data), allocator)
    copy(data, src.data)

    return PackedSymmetric(T){data = data, n = src.n, uplo = src.uplo}
}

// Delete packed matrix
pack_sym_delete :: proc(packed: ^PackedSymmetric($T)) {
    if packed.data != nil {
        delete(packed.data)
        packed.data = nil
    }
}

// Memory usage comparison
pack_sym_memory_savings :: proc(n: int) -> f64 {
    full_size := f64(n * n)
    packed_size := f64(n * (n + 1) / 2)
    return (full_size - packed_size) / full_size * 100.0
}

// Validate packed matrix consistency
pack_sym_validate :: proc(packed: ^PackedSymmetric($T)) -> bool {
    if packed.n <= 0 {
        return false
    }

    expected_size := packed_storage_size(packed.n)
    if len(packed.data) < expected_size {
        return false
    }

    switch packed.uplo {
    case .Upper, .Lower:
        return true
    case .Full:
        return false
    }

    return false
}


// ===================================================================================
// MATRIX PROPERTIES
// ===================================================================================

// Check if a packed matrix appears to be positive definite (diagonal elements > 0)
// Note: This is a necessary but not sufficient condition
pack_sym_is_positive_definite_heuristic :: proc(packed: ^PackedSymmetric($T)) -> bool {
    when is_complex(T) {
        // For complex matrices, check if diagonal elements have positive real parts
        for i in 0 ..< packed.n {
            diag_val := pack_sym_get(packed.data, packed.n, i, i, packed.uplo)
            if real(diag_val) <= 0 {
                return false
            }
        }
    } else {
        // For real matrices, check if diagonal elements are positive
        for i in 0 ..< packed.n {
            diag_val := pack_sym_get(packed.data, packed.n, i, i, packed.uplo)
            if diag_val <= 0 {
                return false
            }
        }
    }
    return true
}

// Get diagonal element efficiently
pack_sym_diagonal_get :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i: int, // Diagonal index
    uplo: MatrixRegion = .Upper,
) -> T {
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

    return AP[idx]
}

// Set diagonal element efficiently
pack_sym_diagonal_set :: proc(
    AP: []$T, // Packed array
    n: int, // Matrix dimension
    i: int, // Diagonal index
    val: T, // Value to set
    uplo: MatrixRegion = .Upper,
) {
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

    AP[idx] = val
}
