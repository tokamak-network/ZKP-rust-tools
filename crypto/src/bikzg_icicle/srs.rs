// srs.rs
use icicle_bls12_381::curve::{
    BaseField, CurveCfg, G1Projective as IcicleG1Projective, G2Affine, ScalarField
};
use icicle_core::curve::Affine;
use icicle_core::traits::{Arithmetic, FieldImpl};
use rand::Rng;

use lambdaworks_math::{elliptic_curve::{
    short_weierstrass::{curves::bls12_381::{
            curve::BLS12381Curve, twist::BLS12381TwistCurve
        }, point::ShortWeierstrassProjectivePoint}, traits::IsEllipticCurve
}, traits::ByteConversion};

use crate::bikzg::srs::{StructuredReferenceString as LambdaSrs, G1Point, G2Point};


pub struct StructuredReferenceString {
    pub dimension_x: usize,
    pub dimension_y: usize,
    pub powers_main_group: Vec<Affine<CurveCfg>>,
}

impl StructuredReferenceString {
    // pub fn create_srs(
    //     dimension_x: usize,
    //     dimension_y: usize,
    // ) -> Self {
    //     let mut rng = rand::thread_rng();

    //     // 1. 랜덤 값 생성
    //     let mut tau_bytes = [0u32; 8];
    //     let mut theta_bytes = [0u32; 8];
    //     for i in 0..8 {
    //         tau_bytes[i] = rng.gen();
    //         theta_bytes[i] = rng.gen();
    //     }
    //     let tau = ScalarField::from(tau_bytes);
    //     let theta = ScalarField::from(theta_bytes);

    //     // 2. Generator를 Icicle 타입으로 변환
    //     let g1_base = lambdaworks_g1_generator_to_icicle().to_projective();
    //     let g2_base = lambdaworks_g2_generator_to_icicle().to_projective();

    //     // 3. G1 points 생성 - 크기 수정
    //     let total_needed = dimension_x * dimension_y;  // inclusive range 제거
    //     let mut powers_main_group = Vec::with_capacity(total_needed);

    //     // 0부터 dimension-1까지만 순회
    //     for i in 0..dimension_y {
    //         for j in 0..dimension_x {
    //             let exponent = tau.pow(j) * theta.pow(i);
    //             let point_proj = g1_base * exponent;
    //             let point_aff = G1Affine::from(point_proj);
    //             powers_main_group.push(point_aff);
    //         }
    //     }

    //     // 4. G2 points 생성
    //     let g2_points = [
    //         G2Affine::from(g2_base),
    //         G2Affine::from(g2_base * tau),
    //         G2Affine::from(g2_base * theta),
    //     ];

    //     Self {
    //         dimension_x,
    //         dimension_y,
    //         powers_main_group,
    //         // powers_secondary_group: g2_points,
    //     }
    // }

    pub fn to_icicle_srs(lambda_srs: &LambdaSrs<G1Point, G2Point>, dimension_x: usize, dimension_y: usize) -> StructuredReferenceString {
        // 1) G1 변환
        let icicle_main: Vec<Affine<CurveCfg>> = lambda_srs
            .powers_main_group
            .iter()
            .map(|lw_g1| Self::lambda_g1_to_icicle_g1(lw_g1))
            .map(|proj| Affine::from(proj))
            .collect();
    
        Self {
            dimension_x,
            dimension_y,
            powers_main_group: icicle_main,   
        }
    }

    pub fn lambda_g1_to_icicle_g1(
        pt_lw: &ShortWeierstrassProjectivePoint<BLS12381Curve>
    ) -> IcicleG1Projective {
        let coords = pt_lw.coordinates(); // [x, y, z]
    
        let x_bytes = coords[0].to_bytes_le();
        let y_bytes = coords[1].to_bytes_le();
        let z_bytes = coords[2].to_bytes_le();
    
        IcicleG1Projective {
            x: BaseField::from_bytes_le(&x_bytes),
            y: BaseField::from_bytes_le(&y_bytes),
            z: BaseField::from_bytes_le(&z_bytes),
        }
    }

    pub fn flatten_partitioned_g1_points_icicle(&self, x_len: usize, y_len: usize) -> Vec<Affine<CurveCfg>> {
        println!("powers_main_group: {:?}", self.powers_main_group);
        let mut chunk_iter = self.powers_main_group.chunks(self.dimension_x);
        let mut output: Vec<Affine<CurveCfg>> = vec![];
        for _ in 0..y_len{
            output.extend( chunk_iter.next().unwrap().iter().take(x_len).cloned());
        }
        println!("output: {:?}", output);
        output
    }
}
