use lambdaworks_math::field::element::FieldElement;
use crate::bikzg::srs::{StructuredReferenceString, G1Point, G2Point};
use lambdaworks_math::elliptic_curve::traits;
use crate::bikzg::verify::traits::IsPairing;

use lambdaworks_math::field::traits::IsPrimeField;

use super::traits::IsCommitmentScheme;
use super::BivariateKateZaveruchaGoldberg;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;

#[cfg(feature = "verify")]
impl<
    const N: usize, 
    F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>, 
    P: IsPairing<G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>>
>
    IsCommitmentScheme<F> for BivariateKateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;
    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proofs: &(Self::Commitment, Self::Commitment),
    ) -> bool {
         // Extract G2 points from the SRS
        let g2 = &self.srs.powers_secondary_group[0];
        let tau_g2 = &self.srs.powers_secondary_group[1];
        let theta_g2 = &self.srs.powers_secondary_group[2];

        // Compute the pairing result using P::compute_batch
        let pairing_result = P::compute_batch(&[
            (
                &p_commitment.operate_with(&(&self.srs.powers_main_group[0].operate_with_self(evaluation.representative())).neg()),
                g2,
            ),
            (
                &proofs.0.neg(),
                &(tau_g2.operate_with(&(g2.operate_with_self(x.representative())).neg())),
            ),
            (
                &proofs.1.neg(),
                &(theta_g2.operate_with(&(g2.operate_with_self(y.representative())).neg())),
            ),
        ]);

        // The pairing result should equal one in the target field
        pairing_result == Ok(FieldElement::one())
        
    }
}
