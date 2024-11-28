use lambdaworks_math::{
    // field::{element::FieldElement},
    msm::pippenger::msm,
};

// use rayon::prelude::*;
use crate::bikzg::G1Point;
// use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;


use crate::bikzg::srs::{G1Point};

// /// Flatten the coefficients of a bivariate polynomial
// ///
// /// # Parameters:
// /// - `bp`: The bivariate polynomial to flatten.
// ///
// /// # Returns:
// /// - `Vec<F>`: A flattened vector of coefficients.
// pub fn flatten_bivariate_coefficients<F: lambdaworks_math::field::traits::IsField>(bp: &BivariatePolynomial<FieldElement<F>>) -> Vec<F> {
//     bp.flatten_out()
//         .iter()
//         .map(|coefficient| coefficient.representative())
//         .collect()
// }

/// Perform Multi-Scalar Multiplication (MSM)
///
/// # Parameters:
/// - `scalars`: The scalar values for multiplication.
/// - `points`: The group elements to be multiplied.
///
/// # Returns:
/// - `G1Point`: The resulting group element after MSM.
pub fn multi_scalar_multiplication<F>(scalars: &[F], points: &[G1Point]) -> G1Point {

    msm(scalars, points).expect("MSM failed: Scalars and points must have the same length.")
}
