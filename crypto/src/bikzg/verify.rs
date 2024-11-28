use lambdaworks_math::{
    cyclic_group::IsGroup,
    field::{element::FieldElement, traits::IsPrimeField}, 
};

use crate::bikzg::G1Point;
use crate::bikzg::srs::StructuredReferenceString;
use lambdaworks_groth16::common::G2Point;

/// Verify an opening proof for a bivariate polynomial
///
/// # Parameters:
/// - `srs`: The Structured Reference String (SRS).
/// - `x`: The x-coordinate of the evaluation point.
/// - `y`: The y-coordinate of the evaluation point.
/// - `evaluation`: The claimed value of the polynomial at `(x, y)`.
/// - `p_commitment`: The commitment to the polynomial \( p(x, y) \).
/// - `proofs`: The opening proofs \((\pi_{xy}, \pi_y)\):
///   - \( \pi_{xy} \): Commitment to \( q_{xy}(x, y) \).
///   - \( \pi_y \): Commitment to \( q_y(y) \).
///
/// # Returns:
/// - `bool`: True if the proof is valid, false otherwise.
pub fn verify<F>(
    srs: &StructuredReferenceString<G1Point, G2Point>,
    x: &FieldElement<F>,
    y: &FieldElement<F>,
    evaluation: &FieldElement<F>,
    p_commitment: &G1Point,
    proofs: &(G1Point, G1Point),
) -> bool where F: IsPrimeField {
    // Extract the proofs
    let (q_xy_commitment, q_y_commitment) = proofs;

    // Extract SRS components
    let g1 = &srs.powers_main_group[0];
    let g2 = &srs.powers_secondary_group[0];
    let tau_g2 = &srs.powers_secondary_group[1];
    let theta_g2 = &srs.powers_secondary_group[2];

    // Compute pairings
    let pairing_result = P::compute_batch(&[
        (
            &p_commitment.operate_with(&(g1.operate_with_self(evaluation.representative())).neg()),
            g2,
        ),
        (
            &q_xy_commitment.neg(),
            &tau_g2.operate_with(&(g2.operate_with_self(x.representative())).neg()),
        ),
        (
            &q_y_commitment.neg(),
            &theta_g2.operate_with(&(g2.operate_with_self(y.representative())).neg()),
        ),
    ]);

    // Check if the pairing result equals the identity element
    pairing_result == Ok(FieldElement::one())
}
