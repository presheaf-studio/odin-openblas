package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// RFP MATRIX CONVERSION FUNCTIONS
//
// This file provides conversion functions for RFP (Rectangular Full Packed) matrices:
// - Dense Matrix ↔ RFP (format conversions)
// - Packed ↔ RFP (between different packed formats)
// - LAPACK format parameter extraction
// - RFP internal format conversions
//
// RFP format provides better cache performance than traditional packed storage
// while maintaining the memory efficiency of storing only one triangle.
// ===================================================================================

// ===================================================================================
// DENSE TO RFP CONVERSION
// ===================================================================================

// Convert dense symmetric/Hermitian matrix to RFP format using LAPACK trttf
dense_to_rfp :: proc(source: ^Matrix($T), trans_state: TransposeState, uplo: UpLo, allocator := context.allocator) -> RFP(T) where is_float(T) || is_complex(T) {
	assert(source.format == .Symmetric || source.format == .Hermitian, "Source matrix must be symmetric or Hermitian")
	assert(source.rows == source.cols, "Source matrix must be square")

	n := int(source.rows)
	dest := make_rfp(n, trans_state, uplo, T, allocator)

	// Use LAPACK conversion routines (STRTTF/DTRTTF for real, CTRTTF/ZTRTTF for complex)
	transr := cast(u8)trans_state
	uplo_c := cast(u8)uplo
	n_blas := source.rows
	lda := source.ld
	info: Info

	when T == f32 {
		lapack.strttf_(&transr, &uplo_c, &n_blas, raw_data(source.data), &lda, raw_data(dest.data), &info)
	} else when T == f64 {
		lapack.dtrttf_(&transr, &uplo_c, &n_blas, raw_data(source.data), &lda, raw_data(dest.data), &info)
	} else when T == complex64 {
		lapack.ctrttf_(&transr, &uplo_c, &n_blas, raw_data(source.data), &lda, raw_data(dest.data), &info)
	} else when T == complex128 {
		lapack.ztrttf_(&transr, &uplo_c, &n_blas, raw_data(source.data), &lda, raw_data(dest.data), &info)
	}

	assert(info == 0, "LAPACK dense to RFP conversion failed")
	return dest
}

// ===================================================================================
// RFP TO DENSE CONVERSION
// ===================================================================================

// Convert RFP matrix back to dense format using LAPACK tfttr
rfp_to_dense :: proc(source: ^RFP($T), allocator := context.allocator) -> Matrix(T) where is_float(T) || is_complex(T) {
	n := int(source.n)

	dest := Matrix(T) {
		data   = make([]T, n * n, allocator),
		rows   = source.n,
		cols   = source.n,
		ld     = source.n,
		format = .Symmetric, // RFP is always for symmetric/Hermitian matrices
		uplo   = source.uplo,
	}

	// Use LAPACK conversion routines (STFTTR/DTFTTR for real, CTFTTR/ZTFTTR for complex)
	transr := cast(u8)source.trans_state
	uplo := cast(u8)source.uplo
	n_blas := source.n
	lda := dest.ld
	info: Info

	when T == f32 {
		lapack.stfttr_(&transr, &uplo, &n_blas, raw_data(source.data), raw_data(dest.data), &lda, &info)
	} else when T == f64 {
		lapack.dtfttr_(&transr, &uplo, &n_blas, raw_data(source.data), raw_data(dest.data), &lda, &info)
	} else when T == complex64 {
		lapack.ctfttr_(&transr, &uplo, &n_blas, raw_data(source.data), raw_data(dest.data), &lda, &info)
	} else when T == complex128 {
		lapack.ztfttr_(&transr, &uplo, &n_blas, raw_data(source.data), raw_data(dest.data), &lda, &info)
	}

	assert(info == 0, "LAPACK RFP to dense conversion failed")
	return dest
}

// ===================================================================================
// PACKED TO RFP CONVERSION
// ===================================================================================

// Convert PackedSymmetric to RFP format
packed_symmetric_to_rfp :: proc(source: ^PackedSymmetric($T), trans_state: TransposeState, allocator := context.allocator) -> RFP(T) where is_float(T) || is_complex(T) {
	dest := make_rfp(source.n, trans_state, source.uplo, T, allocator)

	// Use LAPACK conversion routines (STPTTF/DTPTTF for real, CTPTTF/ZTPTTF for complex)
	n_blas := Blas_Int(source.n)
	transr_c := cast(u8)trans_state
	uplo_c := cast(u8)source.uplo
	info: Info

	when T == f32 {
		lapack.stpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == f64 {
		lapack.dtpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == complex64 {
		lapack.ctpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == complex128 {
		lapack.ztpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	}

	assert(info == 0, "LAPACK conversion failed")
	return dest
}

// Convert PackedHermitian to RFP format
packed_hermitian_to_rfp :: proc(source: ^PackedHermitian($T), trans_state: TransposeState, allocator := context.allocator) -> RFP(T) where T == complex64 || T == complex128 {
	dest := make_rfp(source.n, trans_state, source.uplo, T, allocator)

	// Use LAPACK conversion routines for Hermitian matrices
	n_blas := Blas_Int(source.n)
	transr_c := cast(u8)trans_state
	uplo_c := cast(u8)source.uplo
	info: Info

	when T == complex64 {
		lapack.ctpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == complex128 {
		lapack.ztpttf_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	}

	assert(info == 0, "LAPACK conversion failed")
	return dest
}

// ===================================================================================
// RFP TO PACKED CONVERSION
// ===================================================================================

// Convert RFP to PackedSymmetric format
rfp_to_packed_symmetric :: proc(source: ^RFP($T), allocator := context.allocator) -> PackedSymmetric(T) where is_float(T) || is_complex(T) {
	dest := make_packed_symmetric(int(source.n), source.uplo, T, allocator)

	// Use LAPACK conversion routines (STFTTP/DTFTTP for real, CTFTTP/ZTFTTP for complex)
	n_blas := source.n
	transr_c := cast(u8)source.trans_state
	uplo_c := cast(u8)source.uplo
	info: Info

	when T == f32 {
		lapack.stfttp_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == f64 {
		lapack.dtfttp_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == complex64 {
		lapack.ctfttp_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	} else when T == complex128 {
		lapack.ztfttp_(&transr_c, &uplo_c, &n_blas, raw_data(source.data), raw_data(dest.data), &info)
	}

	assert(info == 0, "LAPACK conversion failed")
	return dest
}

// ===================================================================================
// RFP INTERNAL FORMAT CONVERSIONS
// ===================================================================================

// Convert between different RFP storage formats (transpose state)
convert_rfp_format :: proc(source: ^RFP($T), new_trans_state: TransposeState, allocator := context.allocator) -> RFP(T) where is_float(T) || is_complex(T) {
	if source.trans_state == new_trans_state {
		// No conversion needed, just clone
		return clone_rfp(source, allocator)
	}

	dest := make_rfp(int(source.n), new_trans_state, source.uplo, T, allocator)

	// Use LAPACK conversion routines if available
	// For now, implement via packed format as intermediate
	packed := rfp_to_packed_symmetric(source, context.temp_allocator)
	defer delete(packed.data, context.temp_allocator)

	// Convert back to RFP with new format
	result := packed_symmetric_to_rfp(&packed, new_trans_state, allocator)
	return result
}

// Convert RFP uplo (upper/lower triangle storage)
convert_rfp_uplo :: proc(source: ^RFP($T), new_uplo: UpLo, allocator := context.allocator) -> RFP(T) where is_float(T) || is_complex(T) {
	if source.uplo == new_uplo {
		// No conversion needed, just clone
		return clone_rfp(source, allocator)
	}

	// Convert via dense format as intermediate
	dense := rfp_to_dense(source, context.temp_allocator)
	defer delete(dense.data, context.temp_allocator)

	// Update the uplo field and convert back
	dense.uplo = new_uplo
	result := dense_to_rfp(&dense, source.trans_state, new_uplo, allocator)
	return result
}

// ===================================================================================
// LAPACK FORMAT CONVERSIONS
// ===================================================================================

// Convert RFP to LAPACK TF format parameters
rfp_to_lapack_tf :: proc(rm: ^RFP($T)) -> (tf: []T, transr: TransposeState, uplo: UpLo, n: Blas_Int) {
	return rm.data, rm.trans_state, rm.uplo, rm.n
}

// Convert RFP for LAPACK routine that expects specific format
prepare_rfp_for_lapack :: proc(rm: ^RFP($T), required_trans: TransposeState, required_uplo: UpLo, allocator := context.allocator) -> (result: RFP(T), needs_cleanup: bool) {
	if rm.trans_state == required_trans && rm.uplo == required_uplo {
		// No conversion needed
		return rm^, false
	}

	// Need conversion
	if rm.trans_state != required_trans {
		if rm.uplo != required_uplo {
			// Convert both
			temp := convert_rfp_format(rm, required_trans, allocator)
			result = convert_rfp_uplo(&temp, required_uplo, allocator)
			delete_rfp(&temp, allocator)
		} else {
			// Convert only transpose state
			result = convert_rfp_format(rm, required_trans, allocator)
		}
	} else {
		// Convert only uplo
		result = convert_rfp_uplo(rm, required_uplo, allocator)
	}

	return result, true
}

// ===================================================================================
// BATCH OPERATIONS
// ===================================================================================

// Convert multiple matrices to RFP format in batch
batch_dense_to_rfp :: proc(sources: []^Matrix($T), trans_state: TransposeState, uplo: UpLo, allocator := context.allocator) -> []RFP(T) where is_float(T) || is_complex(T) {
	results := make([]RFP(T), len(sources), allocator)

	for source, i in sources {
		results[i] = dense_to_rfp(source, trans_state, uplo, allocator)
	}

	return results
}

// Convert multiple RFP matrices to dense format in batch
batch_rfp_to_dense :: proc(sources: []^RFP($T), allocator := context.allocator) -> []Matrix(T) where is_float(T) || is_complex(T) {
	results := make([]Matrix(T), len(sources), allocator)

	for source, i in sources {
		results[i] = rfp_to_dense(source, allocator)
	}

	return results
}

// ===================================================================================
// VALIDATION AND UTILITIES
// ===================================================================================

// Verify RFP conversion correctness by comparing with original
verify_rfp_conversion :: proc(original: ^Matrix($T), rfp: ^RFP(T), tolerance: f64 = 1e-12) -> bool where is_float(T) || is_complex(T) {
	// Convert RFP back to dense and compare
	reconstructed := rfp_to_dense(rfp, context.temp_allocator)
	defer delete(reconstructed.data, context.temp_allocator)

	n := int(original.rows)
	for j in 0 ..< n {
		for i in 0 ..< n {
			orig_val := original.data[i + j * int(original.ld)]
			recon_val := reconstructed.data[i + j * n]

			when is_float(T) {
				if abs(orig_val - recon_val) > T(tolerance) {
					return false
				}
			} else {
				if abs(orig_val - recon_val) > T(tolerance) {
					return false
				}
			}
		}
	}

	return true
}

// ===================================================================================
// HELPER FUNCTIONS
// ===================================================================================

// Check if conversion parameters are valid
validate_rfp_conversion_params :: proc(n: int, trans_state: TransposeState, uplo: UpLo) -> bool {
	if n <= 0 {
		return false
	}

	switch trans_state {
	case .NoTrans, .Trans, .ConjTrans:
	// Valid transpose states
	case:
		return false
	}

	switch uplo {
	case .Upper, .Lower:
	// Valid uplo values
	case:
		return false
	}

	return true
}
