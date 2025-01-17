use icicle_core::{curve::Affine, field::Field};
use icicle_core::traits::FieldImpl;
use icicle_bls12_381::curve::{CurveCfg, ScalarCfg};

/// PointConversion 트레이트 구현
impl<CurveCfg> Affine<CurveCfg>
where
    CurveCfg: icicle_core::curve::CurveConfig,
{
    /// 현재 점을 Icicle 포맷으로 변환
    pub fn to_icicle(&self) -> Self {
        self.clone()
    }

    /// Icicle 점을 현재 점 포맷으로 변환
    pub fn from_icicle(point: &Self) -> Self {
        point.clone()
    }
}

/// Field 관련 변환 트레이트 구현
impl<const NUM_LIMBS: usize, F: FieldImpl> Field<NUM_LIMBS, F> {
    /// Icicle ScalarField로 변환
    pub fn to_icicle_scalar(&self) -> ScalarField {
        ScalarField::from_bytes_le(&self.to_bytes_le()).expect("Icicle ScalarField 변환 실패")
    }

    /// Icicle BaseField로 변환
    pub fn to_icicle(&self) -> ScalarField {
        ScalarField::from_bytes_le(&self.to_bytes_le()).expect("Icicle BaseField 변환 실패")
    }

    /// Icicle BaseField에서 변환
    pub fn from_icicle(icicle: &ScalarField) -> Self {
        Self::from_bytes_le(&icicle.to_bytes_le()).expect("Field 변환 실패")
    }
}
