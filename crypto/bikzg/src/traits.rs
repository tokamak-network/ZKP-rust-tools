use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;
use lambdaworks_math::errors::ByteConversionError;

use icicle_bls12_381::curve;
use icicle_bls12_381::curve::ScalarCfg;
use icicle_core::field::Field;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::bikzg::BLS12381Curve;


pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit_bivariate(&self, bp: &BivariatePolynomial<FieldElement<F>>) -> Self::Commitment;
    fn commit_univariate(&self, bp: &UnivariatePolynomial<FieldElement<F>>) -> Self::Commitment;
    fn icicle_commit_bivariate(&self, bp: &BivariatePolynomial<FieldElement<F>>) -> Self::Commitment;
    fn icicle_msm(scalar: Vec<Field<8, ScalarCfg>>, points: &Vec<curve::G1Affine> ) -> ShortWeierstrassProjectivePoint<BLS12381Curve>;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,//f(x,y)
        p: &BivariatePolynomial<FieldElement<F>>,
    ) -> (Self::Commitment,Self::Commitment);


    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proofs: &(Self::Commitment,Self::Commitment),
    ) -> bool;

}

pub trait PointConversion {
    fn to_icicle(&self) -> curve::G1Affine;
    fn from_icicle(icicle: &curve::G1Projective) -> Result<Self, ByteConversionError>
    where
        Self: Sized;
}

pub trait ToIcicle {
    fn to_icicle_scalar(&self) -> curve::ScalarField;
    fn to_icicle(&self) -> curve::BaseField;
    fn from_icicle(icicle: &curve::BaseField) -> Result<Self, ByteConversionError>
    where
        Self: Sized;
}