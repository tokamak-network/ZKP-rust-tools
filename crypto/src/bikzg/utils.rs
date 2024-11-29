use icicle_bls12_381::curve;
use icicle_core::{error::IcicleError, msm, traits::FieldImpl};
use crate::bikzg::G1Point;
use lambdaworks_math::traits::{AsBytes, Deserializable, ByteConversion};
use lambdaworks_math::errors::{DeserializationError, ByteConversionError};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;

use crate::bikzg::traits::{IsCommitmentScheme, PointConversion, ToIcicle};
type BlsG1point = ShortWeierstrassProjectivePoint<BLS12381Curve>;

impl PointConversion for BlsG1point {
    fn to_icicle(&self) -> curve::G1Affine {
        let s = self.to_affine();
        let x = s.x().to_icicle();
        let y = s.y().to_icicle();
        curve::G1Affine { x, y }
    }

    fn from_icicle(icicle: &curve::G1Projective) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            ToIcicle::from_icicle(&icicle.x)?,
            ToIcicle::from_icicle(&icicle.y)?,
            ToIcicle::from_icicle(&icicle.z)?,
        ]))
    }
}

impl ToIcicle for BLS12381FieldElement {
    fn to_icicle_scalar(&self) -> curve::ScalarField {
        let scalar_bytes = self.to_bytes_le();
        curve::ScalarField::from_bytes_le(&scalar_bytes)
    }

    fn to_icicle(&self) -> curve::BaseField {
        curve::BaseField::from_bytes_le(&self.to_bytes_le())
    }

    fn from_icicle(icicle: &curve::BaseField) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(&icicle.to_bytes_le())
    }
}
