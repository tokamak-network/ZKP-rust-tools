use icicle_bls12_381::curve::ScalarField;
use icicle_bls12_381::curve::G1Projective as IcicleG1Projective;
use icicle_core::traits::FieldImpl;
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
        
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            if i * 8 + 8 <= bytes.len() {
                let mut byte_chunk = [0u8; 8];
                byte_chunk.copy_from_slice(&bytes[i*8..i*8+8]);
                limbs[i] = u64::from_le_bytes(byte_chunk);
            }
        }
        let result = U256::from_limbs(limbs);
        println!("Converted to U256: {:?}", result);
        result
    }

    pub fn verify(
        &self,
        x: &ScalarField,
        y: &ScalarField,
        eval: &ScalarField,
        p_commit: &IcicleG1Projective,
        proof: &(IcicleG1Projective, IcicleG1Projective),
    ) -> bool {
        println!("\nStarting verification with inputs:");
        println!("x: {:?}", x.to_bytes_le());
        println!("y: {:?}", y.to_bytes_le());
        println!("eval: {:?}", eval.to_bytes_le());

        // Convert scalar values to U256
        let x_val = Self::scalar_to_uint(x);
        let y_val = Self::scalar_to_uint(y);
        let eval_val = Self::scalar_to_uint(eval);

        println!("\nConverted scalar values:");
        println!("x_val: {:?}", x_val);
        println!("y_val: {:?}", y_val);
        println!("eval_val: {:?}", eval_val);

        // Convert SRS points
        let g2 = icicle_g2_projective_to_lw(&self.srs.powers_secondary_group[0]).unwrap();
        let tau_g2 = icicle_g2_projective_to_lw(&self.srs.powers_secondary_group[1]).unwrap();
        let theta_g2 = icicle_g2_projective_to_lw(&self.srs.powers_secondary_group[2]).unwrap();
        
        println!("\nSRS points converted:");
        println!("g2: {:?}", g2);
        println!("tau_g2: {:?}", tau_g2);
        println!("theta_g2: {:?}", theta_g2);

        // Convert commitment and proof points
        let lw_p_commit = icicle_g1_to_lambdaworks(p_commit).unwrap();
        let (lw_proof0, lw_proof1) = icicle_proof_to_tuple(proof).unwrap();

        println!("\nCommitment and proof points:");
        println!("lw_p_commit: {:?}", lw_p_commit);
        println!("lw_proof0: {:?}", lw_proof0);
        println!("lw_proof1: {:?}", lw_proof1);

        // Convert G1 base point and calculate evaluation term
        let g1_base = icicle_g1_to_lambdaworks(
            &self.srs.powers_main_group[0].to_projective()
        ).unwrap();
        println!("\nG1 base point: {:?}", g1_base);

        // Calculate evaluation term
        let eval_base = g1_base.operate_with_self(eval_val);
        println!("Evaluation base: {:?}", eval_base);
        
        // Calculate G2 points with scalar multiplication
        let x_g2 = g2.operate_with_self(x_val);
        let y_g2 = g2.operate_with_self(y_val);

        println!("\nScalar multiplication results:");
        println!("x_g2: {:?}", x_g2);
        println!("y_g2: {:?}", y_g2);

        // Compute pairing components
        let commitment_term = lw_p_commit.operate_with(&eval_base.neg());
        let tau_term = tau_g2.operate_with(&x_g2.neg());
        let theta_term = theta_g2.operate_with(&y_g2.neg());
        let proof0_term = lw_proof0.neg();
        let proof1_term = lw_proof1.neg();

        println!("\nPairing components:");
        println!("commitment_term: {:?}", commitment_term);
        println!("tau_term: {:?}", tau_term);
        println!("theta_term: {:?}", theta_term);
        println!("proof0_term: {:?}", proof0_term);
        println!("proof1_term: {:?}", proof1_term);

        // Compute batch pairing
        // e(C - eval·G₁, g₂) · e(-π₁, τg₂ - xg₂) · e(-π₂, θg₂ - yg₂) = 1
        let pairing_result = BLS12381AtePairing::compute_batch(&[
            (&commitment_term, &g2),
            (&proof0_term, &tau_term),
            (&proof1_term, &theta_term)
        ]);

        println!("\nPairing result: {:?}", pairing_result);
        
        matches!(pairing_result, Ok(f) if f == FieldElement::one())
    }
}