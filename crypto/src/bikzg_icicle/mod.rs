// src/bikzg_icicle/mod.rs
pub mod srs;
pub mod commit;
pub mod open;
pub mod verify;

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
    use crate::bikzg_icicle::{
        srs::StructuredReferenceString as SrsIcicle, BivariateKateZaveruchaGoldbergIcicle as BIKZG_Icicle
    };
    
    use icicle_bls12_381::curve::ScalarField;
    use icicle_core::traits::FieldImpl;
    use ndarray::array;
    use zkp_rust_tools_math::icicle_bipolynomial::BivariatePolynomial;


    #[test]
    fn test_kzg() {        
        let icicle_srs = SrsIcicle::create_srs(2, 2);
        let icicle_bikzg = BIKZG_Icicle::new(icicle_srs);

        let coeffs_2d = array![
            [ScalarField::from_u32(1), ScalarField::from_u32(1)],
            [ScalarField::from_u32(1), ScalarField::from_u32(1)]
        ];
        let coeffs_vec: Vec<Vec<ScalarField>> = coeffs_2d
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let poly = BivariatePolynomial::new(coeffs_vec);
        let p_commitment = icicle_bikzg.commit_bivariate(&poly);

        let x = ScalarField::from_u32(0); 
        let y = ScalarField::from_u32(10); // 10으로 초기화
        let evaluation = poly.evaluate(&x, &y);

        let proof = icicle_bikzg.open(&x, &y, &evaluation, &poly);

        let is_valid = icicle_bikzg.verify(&x, &y, &evaluation, &p_commitment, &proof);
        println!("is_valid: {:?}", is_valid);

        assert!(is_valid, "bikzg verify should pass, but it failed.");
    }
}