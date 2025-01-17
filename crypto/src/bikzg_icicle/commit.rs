use icicle_core::{field::Field, curve::G1Affine, msm};
use icicle_core::traits::FieldImpl;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;
use crate::bikzg_icicle::srs::StructuredReferenceString;

/// KZG commit 함수를 구현한 구조체
pub struct BivariateKateZaveruchaGoldberg<const NUM_LIMBS: usize, F, CurveCfg> {
    srs: StructuredReferenceString<G1Affine<CurveCfg>, G1Affine<CurveCfg>>,
    _field_marker: std::marker::PhantomData<F>,
}

impl<const NUM_LIMBS: usize, F: FieldImpl, CurveCfg> BivariateKateZaveruchaGoldberg<NUM_LIMBS, F, CurveCfg> {
    /// Bivariate polynomial에 대한 commit 생성
    pub fn commit_bivariate(
        &self,
        poly: &BivariatePolynomial<Field<NUM_LIMBS, F>>,
    ) -> G1Affine<CurveCfg> {
        let scalars = poly
            .flatten_out()
            .iter()
            .map(|scalar| scalar.clone())
            .collect::<Vec<_>>();
        let points = self.srs.powers_main_group.clone();

        msm::msm(&scalars, &points)
    }

    /// Univariate polynomial에 대한 commit 생성
    pub fn commit_univariate(
        &self,
        poly: &UnivariatePolynomial<Field<NUM_LIMBS, F>>,
    ) -> G1Affine<CurveCfg> {
        let scalars = poly
            .coefficients
            .iter()
            .map(|scalar| scalar.clone())
            .collect::<Vec<_>>();
        let points = self
            .srs
            .powers_main_group
            .iter()
            .step_by(self.srs.dimension_x)
            .cloned()
            .collect::<Vec<_>>();

        msm::msm(&scalars, &points)
    }
}
