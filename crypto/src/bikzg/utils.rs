use icicle_bls12_381::curve::{
    BaseField, G1Affine, G1Projective, G2Affine, G2Projective, ScalarField
};
use icicle_core::traits::FieldImpl;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::{
            curve::{BLS12381Curve, BLS12381FieldElement},
            default_types::FrElement,
            twist::BLS12381TwistCurve,
            field_extension::Degree2ExtensionField,
        },
        point::ShortWeierstrassProjectivePoint,
    },
    errors::ByteConversionError,
    field::element::FieldElement,
    traits::ByteConversion,
};
use num_bigint::BigUint;

use crate::bikzg::traits::{PointConversion, ToIcicle};

pub type BlsG1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
pub type BlsG2Point = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

#[inline]
fn from_u64_to_u32_array<const N: usize>(value: u64) -> [u32; N] {
    let low = value as u32;
    let high = (value >> 32) as u32;
    let mut arr = [0u32; N];
    arr[0] = low;
    arr[1] = high;
    arr
}

//-------------------------------------
// Point conversion implementations
//-------------------------------------
impl PointConversion for BlsG1Point {
    fn to_icicle(&self) -> G1Affine {
        let s = self.to_affine();
        let x = s.x().to_icicle();
        let y = s.y().to_icicle();
        G1Affine { x, y }
    }

    fn from_icicle(icicle: &G1Projective) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            ToIcicle::from_icicle(&icicle.x)?,
            ToIcicle::from_icicle(&icicle.y)?,
            ToIcicle::from_icicle(&icicle.z)?,
        ]))
    }
}

//-------------------------------------
// Field element conversion implementations
//-------------------------------------
impl ToIcicle for BLS12381FieldElement {
    fn to_icicle_scalar(&self) -> ScalarField {
        let scalar_bytes = self.to_bytes_le();
        ScalarField::from_bytes_le(&scalar_bytes)
    }

    fn to_icicle(&self) -> BaseField {
        BaseField::from_bytes_le(&self.to_bytes_le())
    }

    fn from_icicle(icicle: &BaseField) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(&icicle.to_bytes_le())
    }
}

//-------------------------------------
// Public conversion functions
//-------------------------------------
pub fn icicle_scalar_to_lambdaworks(
    icicle_scalar: &ScalarField,
) -> Result<FrElement, ByteConversionError> {
    let bytes_le = icicle_scalar.to_bytes_le();
    let int_value = BigUint::from_bytes_le(&bytes_le);
    let hex_string = int_value.to_str_radix(16);
    FrElement::from_hex(&hex_string).map_err(|_| ByteConversionError::InvalidValue)
        
}

pub fn icicle_g1_to_lambdaworks(
    g1proj: &G1Projective
) -> Result<BlsG1Point, ByteConversionError> {
    BlsG1Point::from_icicle(g1proj)
}

pub fn icicle_proof_to_tuple(
    proof: &(G1Projective, G1Projective),
) -> Result<(BlsG1Point, BlsG1Point), ByteConversionError> {
    Ok((
        icicle_g1_to_lambdaworks(&proof.0)?,
        icicle_g1_to_lambdaworks(&proof.1)?,
    ))
}

pub fn icicle_g2_projective_to_lw(
    g2_aff: &G2Affine
) -> Result<BlsG2Point, ByteConversionError> {
    let x_bytes = g2_aff.x.to_bytes_le();
    let y_bytes = g2_aff.y.to_bytes_le();
    
    let (x0_base, x1_base) = x_bytes.split_at(x_bytes.len()/2);
    let (y0_base, y1_base) = y_bytes.split_at(y_bytes.len()/2);

    let x0_fe = BLS12381FieldElement::from_bytes_le(x0_base)?;
    let x1_fe = BLS12381FieldElement::from_bytes_le(x1_base)?;
    let y0_fe = BLS12381FieldElement::from_bytes_le(y0_base)?;
    let y1_fe = BLS12381FieldElement::from_bytes_le(y1_base)?;

    let x_fe2 = FieldElement::<Degree2ExtensionField>::new([x0_fe, x1_fe]);
    let y_fe2 = FieldElement::<Degree2ExtensionField>::new([y0_fe, y1_fe]);
    let z_fe2 = FieldElement::<Degree2ExtensionField>::one();

    Ok(ShortWeierstrassProjectivePoint::new([x_fe2, y_fe2, z_fe2]))
}

//-------------------------------------
// Tests
//-------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_icicle_scalar_to_lambdaworks() {
        let s_arr = from_u64_to_u32_array::<8>(123u64);
        let icicle_s = ScalarField::from(s_arr);
        let lw_s = icicle_scalar_to_lambdaworks(&icicle_s).unwrap();
        
        let expected_hex = FrElement::from(123u64).to_hex();
        let actual_hex = lw_s.to_hex();
        assert_eq!(expected_hex, actual_hex);
    }

    #[test]
    fn test_icicle_g1_to_lambdaworks() {
        let x12 = from_u64_to_u32_array::<12>(1);
        let y12 = from_u64_to_u32_array::<12>(2);
        let z12 = from_u64_to_u32_array::<12>(1);

        let g1_icicle = G1Projective {
            x: BaseField::from(x12),
            y: BaseField::from(y12),
            z: BaseField::from(z12),
        };

        let lw_point = icicle_g1_to_lambdaworks(&g1_icicle).unwrap();
        let coords = lw_point.coordinates();

        assert_eq!(
            BigUint::from_bytes_le(&coords[0].to_bytes_le()),
            BigUint::from(1u64)
        );
        assert_eq!(
            BigUint::from_bytes_le(&coords[1].to_bytes_le()),
            BigUint::from(2u64)
        );
    }
}