// use icicle_core::{field::Field, curve::G1Affine, msm};
use icicle_core::traits::FieldImpl;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;
use crate::bikzg_icicle::srs::StructuredReferenceString;

// use zkp_rust_tools_math::icicle_bipolynomial::BivariatePolynomial;
// use super::BivariateKateZaveruchaGoldberg;

// impl<const NUM_LIMBS: usize, F: FieldImpl, CurveCfg> BivariateKateZaveruchaGoldberg<NUM_LIMBS, F, CurveCfg> {
//     /// Bivariate polynomial에 대한 commit 생성
//     pub fn commit_bivariate(
//         &self,
//         poly: &BivariatePolynomial<Field<NUM_LIMBS, F>>,
//     ) -> G1Affine<CurveCfg> {
//         let scalars = poly
//             .flatten_out()
//             .iter()
//             .map(|scalar| scalar.clone())
//             .collect::<Vec<_>>();
//         let points = self.srs.powers_main_group.clone();

//         msm::msm(&scalars, &points)
//     }

//     /// Univariate polynomial에 대한 commit 생성
//     pub fn commit_univariate(
//         &self,
//         poly: &UnivariatePolynomial<Field<NUM_LIMBS, F>>,
//     ) -> G1Affine<CurveCfg> {
//         let scalars = poly
//             .coefficients
//             .iter()
//             .map(|scalar| scalar.clone())
//             .collect::<Vec<_>>();
//         let points = self
//             .srs
//             .powers_main_group
//             .iter()
//             .step_by(self.srs.dimension_x)
//             .cloned()
//             .collect::<Vec<_>>();

//         msm::msm(&scalars, &points)
//     }
// }
