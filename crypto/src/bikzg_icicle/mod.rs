// src/bikzg_icicle/mod.rs
use crate::bikzg;

pub mod srs;
pub mod commit;
pub mod open;
// pub mod verify;

pub struct BivariateKateZaveruchaGoldbergIcicle {
    pub srs: srs::StructuredReferenceString,    
}

impl BivariateKateZaveruchaGoldbergIcicle {
    pub fn new(srs: srs::StructuredReferenceString) -> Self {
        Self { srs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::srs::StructuredReferenceString as SrsIcicle;
    use crate::bikzg::{srs as LambdaSRS, BivariateKateZaveruchaGoldberg, IsCommitmentScheme, PointConversion};
    use crate::bikzg::srs::StructuredReferenceString as SrsLambda;
    use icicle_bls12_381::curve::ScalarField;
    use icicle_core::traits::FieldImpl;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField};
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::traits::ByteConversion;
    use ndarray::array;
    use zkp_rust_tools_math::icicle_bipolynomial::BivariatePolynomial;
    use zkp_rust_tools_math::bipolynomial::BivariatePolynomial as LambdaBivariatePolynomial;

    #[allow(clippy::upper_case_acronyms)]
    type KZG = BivariateKateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
    
    use crate::bikzg;

    #[test]
    fn test_kzg() {
        // Lambdaworks SRS 생성
        let lambda_srs = bikzg::srs::create_srs((2, 2));
        let lambda_bikzg = KZG::new(lambda_srs.clone());

        let icicle_srs = srs::StructuredReferenceString::to_icicle_srs(&lambda_srs.clone(), 2,2);
        let icicle_bikzg = BivariateKateZaveruchaGoldbergIcicle::new(icicle_srs);

        // 테스트 다항식 생성
        let coeffs_2d = array![
            [ScalarField::from_u32(1), ScalarField::from_u32(1)],
            [ScalarField::from_u32(1), ScalarField::from_u32(1)]
        ];
        let coefficients = array![
            [FrElement::from(1), FrElement::from(1)],
            [FrElement::from(1), FrElement::from(1)]
        ];
        let bp = LambdaBivariatePolynomial::new(coefficients);

        let coeffs_vec: Vec<Vec<ScalarField>> = coeffs_2d
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let poly = BivariatePolynomial::new(coeffs_vec);

        // Commitment 생성
        let p_commitment= icicle_bikzg.commit_bivariate(&poly);
        let lambda_p_commit = lambda_bikzg.commit_bivariate(&bp);

        let p_commitment_converted = PointConversion::from_icicle(&p_commitment).unwrap();

        
        assert_eq!(p_commitment_converted, lambda_p_commit);

        let x = ScalarField::from_u32(0);
        let y = ScalarField::from_u32(10);
        let evaluation = poly.evaluate(&x, &y);

        // Opening 증명 생성
        let proof = icicle_bikzg.open(&x, &y, &evaluation, &poly);

        // Verification 검증
        let x_converted = FieldElement::from_bytes_le(&x.to_bytes_le()).unwrap();
        let y_converted = FieldElement::from_bytes_le(&y.to_bytes_le()).unwrap();
        let evaluation_converted = FieldElement::from_bytes_le(&evaluation.to_bytes_le()).unwrap();

        let lambda_proof = lambda_bikzg.open(&x_converted, &y_converted, &evaluation_converted, &bp);
        
        let proof_converted = (PointConversion::from_icicle(&proof.0).unwrap(), PointConversion::from_icicle(&proof.1).unwrap());
        assert_eq!(proof_converted.0, lambda_proof.0);

        let is_valid = lambda_bikzg.verify(
            &x_converted,
            &y_converted,
            &evaluation_converted,
            &p_commitment_converted,
            &proof_converted,
        );
        // assert!(is_valid, "BiKZG verification failed");
    }


}