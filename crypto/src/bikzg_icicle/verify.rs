use icicle_bls12_381::curve::ScalarField;
use icicle_bls12_381::curve::G1Projective as IcicleG1Projective;
use icicle_core::traits::FieldImpl;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing,
        traits::IsPairing
    },
    field::element::FieldElement,
    unsigned_integer::element::U256
};

use crate::bikzg::utils::{
    icicle_g1_to_lambdaworks, 
    icicle_g2_projective_to_lw, 
    icicle_proof_to_tuple
};

use super::BivariateKateZaveruchaGoldbergIcicle;

impl BivariateKateZaveruchaGoldbergIcicle {
    fn scalar_to_uint(scalar: &ScalarField) -> U256 {
        let bytes = scalar.to_bytes_le();
        println!("Converting bytes: {:?}", bytes);
        
        // 값을 첫 번째 limb에 넣고 나머지는 0으로 설정
        let mut limbs = [0u64; 4];
        if bytes.len() >= 8 {
            let mut byte_chunk = [0u8; 8];
            byte_chunk.copy_from_slice(&bytes[0..8]);
            // 첫 번째 limb에만 값을 저장하고 나머지는 0으로 유지
            limbs[3] = u64::from_le_bytes(byte_chunk);
        }
        
        let result = U256::from_limbs(limbs);
        println!("Converted to U256: {:?}", result);
        result
    }


    pub fn verify(
        &self,
        x: &ScalarField,
        y: &ScalarField,
        evaluation: &ScalarField,
        p_commit: &IcicleG1Projective,
        proof: &(IcicleG1Projective, IcicleG1Projective),
    ) -> bool {
        use crate::bikzg::utils::{icicle_g1_to_lambdaworks, icicle_proof_to_tuple};

        // Icicle 타입을 Lambdaworks 타입으로 변환
        let lw_p_commit = icicle_g1_to_lambdaworks(p_commit).unwrap();
        let (lw_proof0, lw_proof1) = icicle_proof_to_tuple(proof).unwrap();

        // 스칼라 값들을 변환
        let lw_x = Self::scalar_to_fr(x);
        let lw_y = Self::scalar_to_fr(y);
        let lw_eval = Self::scalar_to_fr(evaluation);

        // Lambdaworks verify 호출
        crate::bikzg::verify::verify(
            &self.srs,
            &lw_x,
            &lw_y,
            &lw_eval,
            &lw_p_commit,
            &(lw_proof0, lw_proof1)
        )
    }
}

fn scalar_to_fr(scalar: &ScalarField) -> FrElement {
        let bytes = scalar.to_bytes_le();
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            if start + 8 <= bytes.len() {
                let mut bytes_chunk = [0u8; 8];
                bytes_chunk.copy_from_slice(&bytes[start..start + 8]);
                limbs[i] = u64::from_le_bytes(bytes_chunk);
            }
        }
        FrElement::new(lambdaworks_math::unsigned_integer::element::U256 { limbs })
    }