package openblas

// ===================================================================================
// ENUMS FOR ORTHOGONAL TRANSFORMATIONS
// ===================================================================================
QUERY_WORKSPACE: Blas_Int : -1
ERROR_BOUND_TYPES :: 3 // Number of error bound types

// Side for orthogonal transformation
OrthogonalSide :: enum u8 {
    Left  = 'L', // 'L' - Multiply on the left (premultiply)
    Right = 'R', // 'R' - Multiply on the right (postmultiply)
    Both  = 'C', // 'C'/'T' - Multiply on both sides
}

// Initialization mode for random orthogonal matrix
OrthogonalInit :: enum u8 {
    Identity = 'I', // 'I' - Initialize to identity before applying transformation
    None     = 'N', // 'N' - No initialization, apply to existing matrix
}


// Householder reflector side application
ReflectorSide :: enum u8 {
    Left  = 'L', // Apply from the left
    Right = 'R', // Apply from the right
}

// Householder reflector transpose operation
ReflectorTranspose :: enum u8 {
    None      = 'N', // No transpose
    Transpose = 'T', // Transpose
    Conjugate = 'C', // Conjugate transpose (for complex matrices)
}

// Householder reflector direction
ReflectorDirection :: enum u8 {
    Forward  = 'F', // Forward direction
    Backward = 'B', // Backward direction
}

// Householder reflector storage
ReflectorStorage :: enum u8 {
    ColumnWise = 'C', // Store reflectors column-wise
    RowWise    = 'R', // Store reflectors row-wise
}

// Random distribution types
RandomDistribution :: enum Blas_Int {
    Uniform          = 1, // Uniform distribution on (0,1)
    UniformMinus1To1 = 2, // Uniform distribution on (-1,1)
    Normal           = 3, // Normal distribution
    ComplexUniform   = 5, // Complex uniform distribution on unit circle
}

distribution_to_char :: proc(dist: RandomDistribution) -> u8 {
    switch dist {
    case .Uniform:
        return 'U' // Uniform(0,1) - IDIST=1
    case .UniformMinus1To1:
        return 'S' // Uniform(-1,1) - IDIST=2 (S for Symmetric)
    case .Normal:
        return 'N' // Normal(0,1) - IDIST=3
    case .ComplexUniform:
        // IDIST=5 - Complex uniform on unit circle
        // Only supported by DLARNV/CLARNV/ZLARNV (not DLATMS/DLATMT)
        // There is no character code for this distribution
        return ' ' // Return empty string to indicate not supported for character-based functions
    }
    unreachable()
}

// Equilibration type for matrix scaling
EquilibrationRequest :: enum u8 {
    None   = 'N', // No equilibration applied
    Row    = 'R', // Row equilibration applied
    Column = 'C', // Column equilibration applied
    Both   = 'B', // Both row and column equilibration applied
}

EquilibrationState :: enum u8 {
    None    = 'N', // "N" - No equilibration
    Applied = 'Y', // "Y" - Equilibration was applied
}

TransposeMode :: enum u8 {
    None               = 'N', // No transpose
    Transpose          = 'T', // Transpose
    ConjugateTranspose = 'C', // Conjugate transpose (Hermitian)
}

// Matrix initialization region enumeration
MatrixRegion :: enum u8 {
    Full  = 'A', // "A" - Full matrix
    Upper = 'U', // "U" - Upper triangular part
    Lower = 'L', // "L" - Lower triangular part
}

// Matrix triangle specification (for symmetric/Hermitian/triangular matrices)
MatrixTriangle :: enum u8 {
    Upper = 'U', // "U" - Upper triangle
    Lower = 'L', // "L" - Lower triangle
}

// Diagonal type for triangular matrices
DiagonalType :: enum u8 {
    NonUnit = 'N', // Non-unit diagonal (general triangular)
    Unit    = 'U', // Unit diagonal (diagonal elements assumed to be 1)
}

// Matrix diagonal specification (alias for DiagonalType)
MatrixDiagonal :: DiagonalType

SortDirection :: enum u8 {
    Increasing = 'I', // "I" - Sort in increasing order
    Decreasing = 'D', // "D" - Sort in decreasing order
}

// Factorization option
FactorizationOption :: enum u8 {
    Equilibrate     = 'E', // "E" - Equilibrate, then factor
    NoFactorization = 'N', // "N" - Matrix already factored
    Factor          = 'F', // "F" - Factor the matrix
}

// Factorization type for expert drivers
Factorization_Type :: enum u8 {
    New      = 'N', // "N" - Factor the matrix A
    Factored = 'F', // "F" - Matrix A has already been factored
}

// Equilibration type (output from expert drivers)
Equilibration_Type :: enum u8 {
    None   = 'N', // "N" - No equilibration
    Row    = 'R', // "R" - Row equilibration
    Column = 'C', // "C" - Column equilibration
    Both   = 'B', // "B" - Both row and column equilibration
}

// Transpose operation
Transpose :: enum u8 {
    None               = 'N', // "N" - No transpose
    Transpose          = 'T', // "T" - Transpose
    ConjugateTranspose = 'C', // "C" - Conjugate transpose (Hermitian)
}

// Matrix transpose (alias for Transpose)
MatrixTranspose :: Transpose

// ===================================================================================
// SYMMETRY TYPES
// ===================================================================================

// Matrix symmetry types for test matrix generation
MatrixSymmetry :: enum u8 {
    None              = 'N', // "N" - No symmetry (general matrix)
    Positive_Definite = 'P', // "P" - Symmetric positive definite
    Symmetric         = 'S', // "S" - Symmetric
    Hermitian         = 'H', // "H" - Hermitian (complex matrices)
    Hermitian_Pos_Def = 'R', // "R" - Hermitian positive definite
}

// ===================================================================================
// PACKING TYPES
// ===================================================================================

// Matrix packing/storage types
MatrixPacking :: enum u8 {
    No_Packing   = 'N', // "N" - No packing (full storage)
    Upper_Packed = 'U', // "U" - Upper triangular packed
    Lower_Packed = 'L', // "L" - Lower triangular packed
    Banded       = 'B', // "B" - Band storage
    Rectangular  = 'Q', // "Q" - Rectangular band
    Zero_Band    = 'Z', // "Z" - Zero off-diagonal bands
}

// RFP (Rectangular Full Packed) format transpose options
RFPTranspose :: enum u8 {
    NORMAL    = 'N', // 'N' - Normal form
    TRANSPOSE = 'T', // 'T' - Transpose form
    CONJUGATE = 'C', // 'C' - Conjugate transpose (complex only)
}

// ===================================================================================
// MACHINE PARAMETERS
// ===================================================================================

// Machine parameter types that can be queried from LAPACK
MachineParameter :: enum u8 {
    Epsilon        = 'E', // Relative machine precision
    SafeMinimum    = 'S', // Safe minimum (smallest normalized positive number)
    Base           = 'B', // Base of the machine (radix)
    Precision      = 'P', // Precision (eps*base)
    MantissaDigits = 'N', // Number of mantissa digits in base
    Rounding       = 'R', // 1.0 when rounding occurs in addition, 0.0 otherwise
    MinExponent    = 'M', // Minimum exponent before underflow
    Underflow      = 'U', // Underflow threshold
    MaxExponent    = 'L', // Largest exponent before overflow
    Overflow       = 'O', // Overflow threshold
}

machine_parameter :: proc($T: typeid, param: MachineParameter) -> f64 {
    when T == f32 || T == complex64 {
        // For single precision and complex single
        return f64(lapack.slamch_(cast(u8)param_char))
    } else when T == f64 || T == complex128 {
        // For double precision and complex double
        return lapack.dlamch_(cast(u8)param_char)
    } else {
        // Default to double precision
        return lapack.dlamch_(cast(u8)param_char)
    }
}


MatrixNorm :: enum u8 {
    OneNorm       = '1', // 1-norm (maximum column sum)
    InfinityNorm  = 'I', // infinity-norm (maximum row sum)
    FrobeniusNorm = 'F', // Frobenius norm (sqrt of sum of squares)
    MaxNorm       = 'M', // max norm (largest absolute value)
}

// Eigenvalue Params:

// Vector computation option for reduction
VectorOption :: enum u8 {
    NO_VECTORS   = 'N', // 'N' - No vectors computed
    FORM_VECTORS = 'V', // 'V' - Form transformation matrix
}
// Jobz
EigenJobOption :: enum u8 {
    VALUES_ONLY        = 'N', // 'N' - Compute eigenvalues only
    VALUES_AND_VECTORS = 'V', // 'V' - Compute eigenvalues and eigenvectors
}

// Eigenvector computation side specification
EigenvectorSide :: enum u8 {
    Right = 'R', // Compute right eigenvectors only
    Left  = 'L', // Compute left eigenvectors only
    Both  = 'B', // Compute both left and right eigenvectors
}

// Eigenvector selection specification
EigenvectorSelection :: enum u8 {
    All           = 'A', // Compute all eigenvectors
    Backtransform = 'B', // Backtransform using DTGEVC
    Selected      = 'S', // Compute selected eigenvectors
}

// Compz
CompzOption :: enum u8 {
    None     = 'N', // Eigenvalues only
    Identity = 'I', // Eigenvectors of tridiagonal, Z initialized to identity
    Vectors  = 'V', // Eigenvectors and update Z matrix
}

// Range option for eigenvalue/singular value selection
EigenRangeOption :: enum u8 {
    ALL   = 'A', // 'A' - All eigenvalues/singular values
    VALUE = 'V', // 'V' - Eigenvalues/singular values in range [vl, vu]
    INDEX = 'I', // 'I' - Eigenvalues/singular values with indices il to iu
}
// Alias for SVD functions (same selection mechanism)
SVDRangeOption :: EigenRangeOption

// ===================================================================================
// MATRIX SCALING TYPE (for DLASCL/SLASCL/CLASCL/ZLASCL)
// ===================================================================================

// Matrix scaling type enumeration
MatrixScalingType :: enum u8 {
    General     = 'G', // General matrix
    Lower       = 'L', // Lower triangular/trapezoidal
    Upper       = 'U', // Upper triangular/trapezoidal
    Hessenberg  = 'H', // Upper Hessenberg
    LowerBanded = 'B', // Lower banded triangular
    UpperBanded = 'Q', // Upper banded triangular
    ZeroBanded  = 'Z', // Band matrix with KL=KU=0
}

// ===================================================================================
// TYPE UTILITIES
// ===================================================================================
