package openblas

import "base:builtin"
import "core:math"
import "core:slice"

// Triangle specification
// ===================================================================================
// SHARED PROPERTY ENUMS
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
