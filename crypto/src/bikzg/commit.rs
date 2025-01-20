use lambdaworks_math::{
    elliptic_curve::{
        traits::IsPairing,
        short_weierstrass::{
            curves::bls12_381::curve::BLS12381Curve,
            point::ShortWeierstrassProjectivePoint,
        },
    },
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    unsigned_integer::element::UnsignedInteger,
    cyclic_group::IsGroup, 
};
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use super::traits::IsCommitmentScheme;
use super::BivariateKateZaveruchaGoldberg;

impl<
    const N: usize,
    F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    P: IsPairing<G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>>,
> IsCommitmentScheme<F> for BivariateKateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;

    fn commit_bivariate(&self, poly: &BivariatePolynomial<FieldElement<F>>) -> Self::Commitment {
        let coefficients: Vec<_> = poly
            .flatten_out()
            .iter()
            .map(|c| c.representative())
            .collect();

        let g1_points = self.srs.flatten_partitioned_g1_points(poly.x_degree, poly.y_degree);

        println!("commit_bivariate: coefficients: {:?}, {:?}", coefficients.len(), g1_points.len());
        println!("coefficients: {:?}", coefficients);

        msm(&coefficients, &g1_points)
            .expect("MSM failed: Scalars and points must have the same length.")
    }

    fn commit_univariate(
        &self,
        poly: &UnivariatePolynomial<FieldElement<F>>,
    ) -> Self::Commitment {
        let coefficients_y: Vec<_> = poly
            .coefficients
            .iter()
            .map(|c| c.representative())
            .collect();

        let first_col_powers_main_group: Vec<_> = self
            .srs
            .powers_main_group
            .iter()
            .step_by(self.srs.dimention_x)
            .cloned()
            .collect();

        msm(
            &coefficients_y,
            &first_col_powers_main_group[..coefficients_y.len()],
        )
        .expect("MSM failed: Scalars and points must have the same length.")
    }

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p: &BivariatePolynomial<FieldElement<F>>,
    ) -> (Self::Commitment, Self::Commitment) {
        let adjusted_poly = p.sub_by_field_element(evaluation);
        let (q_xy, q_y) = adjusted_poly.ruffini_division(x, y);

        let q_xy_commitment = self.commit_bivariate(&q_xy);
        let q_y_commitment = self.commit_univariate(&q_y);

        (q_xy_commitment, q_y_commitment)
    }

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proofs: &(Self::Commitment, Self::Commitment),
    ) -> bool {
        let g2 = &self.srs.powers_secondary_group[0];
        let tau_g2 = &self.srs.powers_secondary_group[1];
        let theta_g2 = &self.srs.powers_secondary_group[2];

        // println!("g2: {:?}", g2);

        let pairing_result = P::compute_batch(&[
            (
                &p_commitment.operate_with(
                    &(&self.srs.powers_main_group[0]
                        .operate_with_self(evaluation.representative()))
                    .neg(),
                ),
                g2,
            ),
            (
                &proofs.0.neg(),
                &(tau_g2.operate_with(
                    &(g2.operate_with_self(x.representative())).neg(),
                )),
            ),
            (
                &proofs.1.neg(),
                &(theta_g2.operate_with(
                    &(g2.operate_with_self(y.representative())).neg(),
                )),
            ),
        ]);

        pairing_result == Ok(FieldElement::one())
    }
}
