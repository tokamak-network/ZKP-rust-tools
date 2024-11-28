use crate::bikzg::srs::StructuredReferenceString;
use crate::bikzg::utils::{multi_scalar_multiplication};

use lambdaworks_math::{
    field::{element::FieldElement},
};

use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;

use crate::bikzg::G1Point;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;

use srs::G2Point;

/// Generate a commitment for a bivariate polynomial
///
/// # Parameters:
/// - `srs`: The Structured Reference String (SRS).
/// - `bp`: The bivariate polynomial to commit to.
///
/// # Returns:
/// - `G1Point`: The commitment to the polynomial.
pub fn commit_bivariate<F: lambdaworks_math::field::traits::IsField>(
    srs: &StructuredReferenceString<G1Point, G2Point>,
    bp: &BivariatePolynomial<FieldElement<F>>,
) -> G1Point {
    // Flatten coefficients of the bivariate polynomial
    let coefficients = bp.flatten_out()
        .iter()
        .map(|coefficient| coefficient.representative())
        .collect();

    // Perform Multi-Scalar Multiplication (MSM) using the flattened coefficients and SRS points
    multi_scalar_multiplication(&coefficients, &srs.powers_main_group)
}

/// Generate a commitment for a univariate polynomial
///
/// # Parameters:
/// - `srs`: The Structured Reference String (SRS).
/// - `poly`: The univariate polynomial to commit to.
///
/// # Returns:
/// - `G1Point`: The commitment to the polynomial.
pub fn commit_univariate<F: lambdaworks_math::field::traits::IsField>(
    srs: &StructuredReferenceString<G1Point, G2Point>,
    poly: &UnivariatePolynomial<FieldElement<F>>,
) -> G1Point {
    // Extract coefficients of the univariate polynomial
    let coefficients = poly.coefficients.iter().map(|c| c.representative()).collect();

    // Use only the first column of the SRS for univariate commitment
    let first_column_powers = srs.powers_main_group.iter().step_by(srs.dimention_x).collect();

    // Perform Multi-Scalar Multiplication (MSM)
    multi_scalar_multiplication(&coefficients, &first_column_powers)
}
