// srs.rs
use icicle_bls12_381::curve::{
    G1Affine, G2Affine, ScalarField
};
use icicle_core::traits::Arithmetic;
use rand::Rng;

use lambdaworks_math::{elliptic_curve::{
    short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, twist::BLS12381TwistCurve
        }, traits::IsEllipticCurve
}, traits::ByteConversion};

// Generator를 Icicle 타입으로 변환하는 새로운 함수들
fn lambdaworks_g1_generator_to_icicle() -> G1Affine {
    let g1_gen = BLS12381Curve::generator();
    let coords = g1_gen.coordinates();
    
    // G1의 경우 단순 필드 요소로 구성
    let x_bytes = coords[0].to_bytes_le();
    let y_bytes = coords[1].to_bytes_le();
    
    let mut x_limbs = [0u32; 12];
    let mut y_limbs = [0u32; 12];
    
    for (i, chunk) in x_bytes.chunks(4).enumerate() {
        x_limbs[i] = u32::from_le_bytes(chunk.try_into().expect("Chunk size mismatch"));
    }
    
    for (i, chunk) in y_bytes.chunks(4).enumerate() {
        y_limbs[i] = u32::from_le_bytes(chunk.try_into().expect("Chunk size mismatch"));
    }
    
    G1Affine::from_limbs(x_limbs, y_limbs)
}

fn lambdaworks_g2_generator_to_icicle() -> G2Affine {
    let g2_gen = BLS12381TwistCurve::generator();
    let coords = g2_gen.coordinates();
    
    // G2의 경우 각 좌표가 2개의 기본 필드 요소로 구성
    let x_fe_vec = coords[0].to_bytes_le();
    let y_fe_vec = coords[1].to_bytes_le();
    
    let mut x_fe = [0u32; 24];
    let mut y_fe = [0u32; 24];
    
    for (i, chunk) in x_fe_vec.chunks(4).enumerate() {
        x_fe[i] = u32::from_le_bytes(chunk.try_into().expect("Chunk size mismatch"));
    }
    
    for (i, chunk) in y_fe_vec.chunks(4).enumerate() {
        y_fe[i] = u32::from_le_bytes(chunk.try_into().expect("Chunk size mismatch"));
    }
    
    G2Affine::from_limbs(x_fe, y_fe)        
}

pub struct StructuredReferenceString {
    pub dimension_x: usize,
    pub dimension_y: usize,
    pub powers_main_group: Vec<G1Affine>,
    pub powers_secondary_group: [G2Affine; 3],
}

impl StructuredReferenceString {
    pub fn create_srs(
        dimension_x: usize,
        dimension_y: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // 1. 랜덤 값 생성
        let mut tau_bytes = [0u32; 8];
        let mut theta_bytes = [0u32; 8];
        for i in 0..8 {
            tau_bytes[i] = rng.gen();
            theta_bytes[i] = rng.gen();
        }
        let tau = ScalarField::from(tau_bytes);
        let theta = ScalarField::from(theta_bytes);

        // 2. Generator를 Icicle 타입으로 변환
        let g1_base = lambdaworks_g1_generator_to_icicle().to_projective();
        let g2_base = lambdaworks_g2_generator_to_icicle().to_projective();

        // 3. G1 points 생성 - 크기 수정
        let total_needed = dimension_x * dimension_y;  // inclusive range 제거
        let mut powers_main_group = Vec::with_capacity(total_needed);

        // 0부터 dimension-1까지만 순회
        for i in 0..dimension_y {
            for j in 0..dimension_x {
                let exponent = tau.pow(j) * theta.pow(i);
                let point_proj = g1_base * exponent;
                let point_aff = G1Affine::from(point_proj);
                powers_main_group.push(point_aff);
            }
        }

        // 4. G2 points 생성
        let g2_points = [
            G2Affine::from(g2_base),
            G2Affine::from(g2_base * tau),
            G2Affine::from(g2_base * theta),
        ];

        Self {
            dimension_x,
            dimension_y,
            powers_main_group,
            powers_secondary_group: g2_points,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_srs() {
        let srs = StructuredReferenceString::create_srs(2, 2);
        assert_eq!(srs.powers_main_group.len(), 4);  // 2 x 2 = 4
        assert_eq!(srs.powers_secondary_group.len(), 3);
    }

    #[test]
    fn test_various_dimensions() {
        let srs1 = StructuredReferenceString::create_srs(2, 3);
        assert_eq!(srs1.powers_main_group.len(), 6);  // 2 x 3 = 6

        let srs2 = StructuredReferenceString::create_srs(3, 2);
        assert_eq!(srs2.powers_main_group.len(), 6);  // 3 x 2 = 6
    }
}