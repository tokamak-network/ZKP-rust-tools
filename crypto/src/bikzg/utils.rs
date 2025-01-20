use icicle_bls12_381::curve::{
    G1Projective,
    ScalarField,
    G1Affine,
    G2Affine,
    BaseField,
    G2Projective
};
use icicle_core::traits::FieldImpl;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::{curves::bls12_381::{curve::{BLS12381Curve, BLS12381FieldElement}, default_types::FrElement, twist::BLS12381TwistCurve}, point::ShortWeierstrassProjectivePoint}, errors::ByteConversionError, field::element::FieldElement, traits::ByteConversion
};
use icicle_bls12_381::curve::ScalarField as IcicleScalar;
use num_bigint::BigUint;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::Degree2ExtensionField;

use crate::bikzg::traits::{PointConversion, ToIcicle};

// lambdaworks BLS12-381 G1
type BlsG1point = ShortWeierstrassProjectivePoint<BLS12381Curve>;

impl PointConversion for BlsG1point {
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

impl ToIcicle for BLS12381FieldElement {
    fn to_icicle_scalar(&self) -> ScalarField {
        // lambdaworks FieldElement -> icicle ScalarField
        let scalar_bytes = self.to_bytes_le();
        ScalarField::from_bytes_le(&scalar_bytes)
    }

    fn to_icicle(&self) -> BaseField {
        // lambdaworks FieldElement -> icicle BaseField
        BaseField::from_bytes_le(&self.to_bytes_le())
    }

    fn from_icicle(icicle: &BaseField) -> Result<Self, ByteConversionError> {
        // icicle BaseField -> lambdaworks FieldElement
        Self::from_bytes_le(&icicle.to_bytes_le())
    }
}

pub fn icicle_scalar_to_lambdaworks(
    icicle_scalar: &IcicleScalar,
) -> Result<FrElement, ByteConversionError> {
    let bytes_le = icicle_scalar.to_bytes_le();
    let int_value = BigUint::from_bytes_le(&bytes_le);
    let hex_string = int_value.to_str_radix(16);
    
    let fr = FrElement::from_hex(&hex_string).map_err(|_| {
        panic!("Failed to convert hex string to FrElement: {:?}", hex_string)
    })?;

    Ok(fr)
}

pub fn icicle_g1_to_lambdaworks(
    g1proj: &G1Projective
) -> Result<BlsG1point, ByteConversionError> {
    let lw_point = BlsG1point::from_icicle(g1proj)?;
    Ok(lw_point)
}

pub fn icicle_proof_to_tuple(
    proof: &(G1Projective, G1Projective),
) -> Result<(BlsG1point, BlsG1point), ByteConversionError> {
    // 각각 icicle -> lambdaworks G1 변환
    let p1 = icicle_g1_to_lambdaworks(&proof.0)?;
    let p2 = icicle_g1_to_lambdaworks(&proof.1)?;
    Ok((p1, p2))
}

/// Lambdaworks BLS12-381 G1 alias
pub type BlsG1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
/// Lambdaworks BLS12-381 G2 alias (있다면)
pub type BlsG2Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;


/// (2) icicle G1Affine -> lambdaworks G1Point
///     (Affine → Projective 변환 + 필드 좌표 변환)
pub fn icicle_g1_affine_to_lw(
    g1_aff: &G1Affine
) -> Result<BlsG1Point, ByteConversionError> {
    // G1Affine { x: BaseField, y: BaseField } → bytes → lambdaworks field
    // 간단 예시:
    let x_bytes = g1_aff.x.to_bytes_le();
    let y_bytes = g1_aff.y.to_bytes_le();

    let x_int = BigUint::from_bytes_le(&x_bytes);
    let y_int = BigUint::from_bytes_le(&y_bytes);

    let x_hex = x_int.to_str_radix(16);
    let y_hex = y_int.to_str_radix(16);

    let x_fe = BLS12381FieldElement::from_hex(&x_hex)
        .map_err(|_| panic!("Failed to convert hex string to FrElement: {:?}", x_hex))?;
    let y_fe = BLS12381FieldElement::from_hex(&y_hex)
        .map_err(|_| panic!("Failed to convert hex string to FrElement: {:?}", y_hex))?;

    // ShortWeierstrassProjectivePoint::from_affine(x, y)
    let p = ShortWeierstrassProjectivePoint::new([x_fe, y_fe, BLS12381FieldElement::one()]);
    Ok(p)
}

/// (3) icicle G2Affine -> lambdaworks G2Point
///     (동일한 방식, 단 G2가 twist 필드를 쓴다면 추가 고려가 필요하지만
///      예시상 간단히 "같은 BLS12381Curve"로 처리)
fn icicle_g2_affine_to_lw(
    g2_aff: &icicle_bls12_381::curve::G2Affine
) -> Result<ShortWeierstrassProjectivePoint<BLS12381TwistCurve>, ByteConversionError> {
    // Suppose g2_aff.x and g2_aff.y each store Fq2 in two base-limb fields: (x.c0, x.c1), (y.c0, y.c1).
    // Pseudocode: adjust to your actual icicle representation.

    let x_bytes = g2_aff.x.to_bytes_le();
    let y_bytes = g2_aff.y.to_bytes_le();
    let (x0_base, x1_base) = x_bytes.split_at(48);
    let (y0_base, y1_base) = y_bytes.split_at(48);

    // Convert each base part to lambdaworks BLS12381FieldElement
    let x0_fe = BLS12381FieldElement::from_bytes_le(x0_base)?;
    let x1_fe = BLS12381FieldElement::from_bytes_le(x1_base)?;
    let y0_fe = BLS12381FieldElement::from_bytes_le(y0_base)?;
    let y1_fe = BLS12381FieldElement::from_bytes_le(y1_base)?;

    // Now build extension field elements
    let x_fe2 = FieldElement::<Degree2ExtensionField>::new([x0_fe, x1_fe]);
    let y_fe2 = FieldElement::<Degree2ExtensionField>::new([y0_fe, y1_fe]);
    let z_fe2 = FieldElement::<Degree2ExtensionField>::one();

    Ok(ShortWeierstrassProjectivePoint::new([
        x_fe2, 
        y_fe2, 
        z_fe2
    ]))
}


// pub fn icicle_g1_projective_to_lw(
//     g1_proj: &G1Projective
// ) -> Result<BlsG1Point, ByteConversionError> {
//     // 1) icicle G1Projective -> G1Affine
//     let aff: G1Affine = (*g1_proj).into();
//     // 2) 이미 구현된 G1Affine -> Lambdaworks 변환 재사용
//     icicle_g1_affine_to_lw(&aff)
// }

// pub fn icicle_g2_projective_to_lw(
//     g2_proj: &G2Projective
// ) -> Result<BlsG2Point, ByteConversionError> {
//     // 1) icicle G2Projective -> G2Affine
//     let aff: G2Affine = (*g2_proj).into();
//     // 2) 이미 구현된 G2Affine -> Lambdaworks 변환 재사용
//     icicle_g2_affine_to_lw(&aff)
// }

pub fn icicle_g1_projective_to_lw(
    g1_proj: &G1Projective,
) -> Result<ShortWeierstrassProjectivePoint<BLS12381Curve>, ByteConversionError> {
    // 1) Projective -> Affine (icicle)
    let aff: G1Affine = (*g1_proj).into();
    // 2) Now call your existing “icicle_g1_affine_to_lw(&aff)” if you have it
    //    or do direct logic here to turn it into a `ShortWeierstrassProjectivePoint<BLS12381Curve>`.
    icicle_g1_affine_to_lw(&aff)
}

pub fn icicle_g2_projective_to_lw(
    g2_proj: &G2Projective,
) -> Result<ShortWeierstrassProjectivePoint<BLS12381TwistCurve>, ByteConversionError> {
    let aff: G2Affine = (*g2_proj).into();
    icicle_g2_affine_to_lw(&aff)
}
