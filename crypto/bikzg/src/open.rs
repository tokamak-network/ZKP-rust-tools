use crate::srs::StructuredReferenceString;
use crate::commit::{commit_bivariate, commit_univariate};
use lambdaworks_math::field::element::FieldElement;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;

/// Generate an opening proof for a bivariate polynomial
///
/// # Parameters:
/// - `srs`: The Structured Reference String (SRS).
/// - `x`: The x-coordinate of the evaluation point.
/// - `y`: The y-coordinate of the evaluation point.
/// - `evaluation`: The value of the polynomial at `(x, y)`.
/// - `bp`: The bivariate polynomial.
///
/// # Returns:
/// - `(G1Point, G1Point)`: The opening proof as two commitments.
///   - \( \pi_{xy} \): Commitment to \( q_{xy}(x, y) \).
///   - \( \pi_y \): Commitment to \( q_y(y) \).
pub fn open<F>(
    srs: &StructuredReferenceString<G1Point, G2Point>,
    x: &FieldElement<F>,
    y: &FieldElement<F>,
    evaluation: &FieldElement<F>,
    bp: &BivariatePolynomial<FieldElement<F>>,
) -> (G1Point, G1Point) {
    // Subtract the evaluation value from the polynomial
    let shifted_poly = bp.sub_by_field_element(evaluation);

    // Perform Ruffini division to compute q_{xy} and q_y
    let (q_xy, q_y) = shifted_poly.ruffini_division(x, y);

    // Generate commitments for q_{xy} and q_y
    let q_xy_commitment = commit_bivariate(srs, &q_xy);
    let q_y_commitment = commit_univariate(srs, &q_y);

    (q_xy_commitment, q_y_commitment)
}
