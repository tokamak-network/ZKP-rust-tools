use icicle_core::{curve::G1Affine, field::Field};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use crate::bikzg_icicle::commit::BivariateKateZaveruchaGoldberg;

impl<const NUM_LIMBS: usize, F: icicle_core::traits::FieldImpl, CurveCfg>
    BivariateKateZaveruchaGoldberg<NUM_LIMBS, F, CurveCfg>
{
    /// Bivariate polynomial을 기반으로 열기 증명 생성
    pub fn open(
        &self,
        x: &Field<NUM_LIMBS, F>,
        y: &Field<NUM_LIMBS, F>,
        evaluation: &Field<NUM_LIMBS, F>,
        poly: &BivariatePolynomial<Field<NUM_LIMBS, F>>,
    ) -> (G1Affine<CurveCfg>, G1Affine<CurveCfg>) {
        // f(x, y) - evaluation
        let adjusted_poly = poly.sub_by_field_element(evaluation);

        // Ruffini 분할 (x와 y에 대해)
        let (q_xy, q_y) = adjusted_poly.ruffini_division(x, y);

        // 각각의 다항식에 대해 커밋 생성
        let q_xy_commitment = self.commit_bivariate(&q_xy);
        let q_y_commitment = self.commit_univariate(&q_y);

        (q_xy_commitment, q_y_commitment)
    }
}
