package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED SYMMETRIC MATRIX CONVERSIONS
// ===================================================================================
//
// This file provides conversion operations between packed symmetric matrices
// and other matrix formats. Packed storage stores only the upper or lower triangle
// of symmetric matrices in a 1D array, saving approximately 50% memory.
//
// Conversion operations:
// - pack_sym_from_dns_array: Convert full matrix array to packed storage array (non-allocating)
// - dns_from_pack_sym_array: Convert packed storage array to full matrix array (non-allocating)
// - pack_sym_make_from_dns: Allocating conversion from full array to PackedSymmetric
// - pack_sym_to_dns: Allocating conversion from PackedSymmetric to full array

// ===================================================================================
// MEMORY LAYOUT CONVERSIONS
// ===================================================================================

// Convert full symmetric matrix to packed storage
// Non-allocating version: requires pre-allocated packed array
pack_sym_from_dns_array :: proc(
    A: []$T, // Full matrix (n×n) in column-major order
    AP: []T, // Pre-allocated packed array (n*(n+1)/2 elements)
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
) where is_float(T) || is_complex(T) {
    assert(validate_packed_storage(n, len(AP)), "Packed array too small")
    assert(len(A) >= n * lda, "Full matrix array too small")

    idx := 0
    switch uplo {
    case .Upper:
        // Pack upper triangle column by column
        for j in 0 ..< n {
            for i in 0 ..= j {
                AP[idx] = A[i + j * lda]
                idx += 1
            }
        }
    case .Lower:
        // Pack lower triangle column by column
        for j in 0 ..< n {
            for i in j ..< n {
                AP[idx] = A[i + j * lda]
                idx += 1
            }
        }
    case .Full:
        panic("Full storage not supported for packed format")
    }
}

// Convert packed storage to full symmetric matrix
// Non-allocating version: requires pre-allocated full matrix
dns_from_pack_sym_array :: proc(
    AP: []$T, // Packed array
    A: []T, // Pre-allocated full matrix (n×n)
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
) where is_float(T) || is_complex(T) {
    assert(validate_packed_storage(n, len(AP)), "Packed array too small")
    assert(len(A) >= n * lda, "Full matrix array too small")

    // Initialize full matrix to zero
    for i in 0 ..< n * lda {
        A[i] = T(0)
    }

    idx := 0
    switch uplo {
    case .Upper:
        // Unpack upper triangle and mirror to lower
        for j in 0 ..< n {
            for i in 0 ..= j {
                val := AP[idx]
                A[i + j * lda] = val
                if i != j {
                    A[j + i * lda] = val // Mirror to lower triangle
                }
                idx += 1
            }
        }
    case .Lower:
        // Unpack lower triangle and mirror to upper
        for j in 0 ..< n {
            for i in j ..< n {
                val := AP[idx]
                A[i + j * lda] = val
                if i != j {
                    A[j + i * lda] = val // Mirror to upper triangle
                }
                idx += 1
            }
        }
    case .Full:
        panic("Full storage not supported for packed format")
    }
}

// ===================================================================================
// ALLOCATING CONVERSION FUNCTIONS
// ===================================================================================

// Create PackedSymmetric from full matrix (allocating version for convenience)
pack_sym_make_from_dns :: proc(
    A: []$T, // Full matrix
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> PackedSymmetric(T) where is_float(T) || is_complex(T) {
    packed_size := packed_storage_size(n)
    data := make([]T, packed_size, allocator)

    pack_sym_from_dns_array(A, data, n, lda, uplo)

    return PackedSymmetric(T){data = data, n = n, uplo = uplo}
}

// Extract full matrix from PackedSymmetric (allocating version for convenience)
pack_sym_to_dns :: proc(
    packed: ^PackedSymmetric($T),
    lda: int, // Leading dimension for output
    allocator := context.allocator,
) -> []T where is_float(T) || is_complex(T) {
    if lda < packed.n {
        panic("Leading dimension too small")
    }

    A := make([]T, packed.n * lda, allocator)
    dns_from_pack_sym_array(packed.data, A, packed.n, lda, packed.uplo)

    return A
}

// ===================================================================================
// BATCH CONVERSION OPERATIONS
// ===================================================================================

// Convert multiple packed matrices to full format in batch
pack_sym_batch_to_dns :: proc(
    packed_matrices: []^PackedSymmetric($T),
    lda: int,
    allocator := context.allocator,
) -> [][]T {
    results := make([][]T, len(packed_matrices), allocator)

    for i, packed in packed_matrices {
        results[i] = pack_sym_to_dns(packed, lda, allocator)
    }

    return results
}

// Convert multiple full matrices to packed format in batch
pack_sym_batch_from_dns :: proc(
    full_matrices: [][]$T,
    n: int,
    lda: int,
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> []PackedSymmetric(T) {
    results := make([]PackedSymmetric(T), len(full_matrices), allocator)

    for i, A in full_matrices {
        results[i] = pack_sym_make_from_dns(A, n, lda, uplo, allocator)
    }

    return results
}

// ===================================================================================
// IN-PLACE CONVERSION UTILITIES
// ===================================================================================

// Convert upper triangle storage to lower triangle storage (or vice versa)
// This operation modifies the packed matrix in-place
pack_sym_transpose :: proc(packed: ^PackedSymmetric($T)) {
    n := packed.n
    temp_data := make([]T, len(packed.data))
    defer delete(temp_data)

    if packed.uplo == .Upper {
        // Convert upper to lower storage
        idx := 0
        for j in 0 ..< n {
            for i in j ..< n {
                // Get element (i,j) from upper storage
                upper_idx := j + i * (i + 1) / 2
                temp_data[idx] = packed.data[upper_idx]
                idx += 1
            }
        }
        packed.uplo = .Lower
    } else {
        // Convert lower to upper storage
        idx := 0
        for j in 0 ..< n {
            for i in 0 ..= j {
                // Get element (i,j) from lower storage
                lower_idx := (j - i) + i * (2 * n - i - 1) / 2
                temp_data[idx] = packed.data[lower_idx]
                idx += 1
            }
        }
        packed.uplo = .Upper
    }

    copy(packed.data, temp_data)
}

// ===================================================================================
// CONVERSION VALIDATION
// ===================================================================================

// Verify that a packed matrix correctly represents a symmetric matrix
// by converting to full and checking symmetry
pack_sym_verify_symmetry :: proc(packed: ^PackedSymmetric($T), tolerance: T) -> bool {
    // Extract to full matrix
    A := pack_sym_to_dns(packed, packed.n) // This is janky;; fixme
    defer delete(A)

    // Check symmetry: A[i,j] == A[j,i]
    for i in 0 ..< packed.n {
        for j in 0 ..< packed.n {
            diff := A[i + j * packed.n] - A[j + i * packed.n]
            if abs(diff) > tolerance {
                return false
            }
        }
    }

    return true
}

// Compare packed and full representations for equivalence
pack_sym_compare_full :: proc(packed: ^PackedSymmetric($T), A: []T, lda: int, tolerance: T) -> bool {
    if lda < packed.n {
        return false
    }

    // Check each element
    for i in 0 ..< packed.n {
        for j in 0 ..< packed.n {
            packed_val := pack_sym_get(packed.data, packed.n, i, j, packed.uplo)
            full_val := A[i + j * lda]

            diff := packed_val - full_val
            if abs(diff) > tolerance {
                return false
            }
        }
    }

    return true
}
