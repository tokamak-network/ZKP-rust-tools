// open.rs

use icicle_bls12_381::curve::ScalarField;
use icicle_bls12_381::curve::G1Projective as BLS12381G1Projective;
use zkp_rust_tools_math::icicle_bipolynomial::BivariatePolynomial;
use super::BivariateKateZaveruchaGoldbergIcicle; 


impl BivariateKateZaveruchaGoldbergIcicle {
    pub fn open(
        &self,
        x: &ScalarField,
        y: &ScalarField,
        evaluation: &ScalarField,
        poly: &BivariatePolynomial,
    ) -> (BLS12381G1Projective, BLS12381G1Projective) {
        let adjusted_poly = poly.sub_by_field_element(*evaluation);
        // println!("adjusted_poly.coefficients: {:?}", adjusted_poly.coefficients);
        for coeff in &adjusted_poly.coefficients {
            coeff.print();
        }
        let (q_xy, q_y) = adjusted_poly
            .ruffini_division(x, y)
            .expect("Ruffini division error");
        
        let q_xy_commitment = self.commit_bivariate(&q_xy);
        let q_y_commitment = self.commit_univariate(&q_y);

        (q_xy_commitment, q_y_commitment)
    }
}