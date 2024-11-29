use lambdaworks_math::{
    field::{element::FieldElement, traits::IsPrimeField},
    elliptic_curve::traits::IsPairing,
};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use super::traits::IsCommitmentScheme;
use super::BivariateKateZaveruchaGoldberg;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;

#[cfg(feature = "open")]
impl<
    const N: usize, 
    F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>, 
    P: IsPairing<G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>>
>
    IsCommitmentScheme<F> for BivariateKateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p: &BivariatePolynomial<FieldElement<F>>,
    ) -> (Self::Commitment, Self::Commitment) {
        // Compute q_xy(x, y) = (p(x, y) - evaluation) / ((x - X)(y - Y))
        let adjusted_poly = p.sub_by_field_element(evaluation);
        let (q_xy, q_y) = adjusted_poly.ruffini_division(x, y);

        // Commit to q_xy and q_y
        let q_xy_commitment = self.commit_bivariate(&q_xy);
        let q_y_commitment = self.commit_univariate(&q_y);

        (q_xy_commitment, q_y_commitment)
    }
}
