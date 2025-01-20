// Export the SRS-related utilities and structures
pub mod srs;
pub mod commit;
pub mod traits;
pub mod utils;
pub use srs::{StructuredReferenceString, G2Point};
pub use traits::{IsCommitmentScheme, PointConversion, ToIcicle};
use core::marker::PhantomData;

// Re-export types from external libraries for convenience
pub use lambdaworks_math::{
    field::element::FieldElement,
    elliptic_curve::short_weierstrass::curves::bls12_381::{
                curve::BLS12381Curve, 
                pairing::BLS12381AtePairing,
            },
};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::elliptic_curve::traits::IsPairing;
use lambdaworks_math::field::traits::IsPrimeField;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;

#[derive(Clone)]
pub struct BivariateKateZaveruchaGoldberg<F: IsPrimeField, P: IsPairing> {
    srs: StructuredReferenceString<P::G1Point, P::G2Point>,
    phantom: PhantomData<F>,
}

impl<F: IsPrimeField, P: IsPairing> BivariateKateZaveruchaGoldberg<F, P> {
    pub fn new(srs: StructuredReferenceString<P::G1Point, P::G2Point>) -> Self {
        Self {
            srs,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    // use alloc::vec::Vec;
    use lambdaworks_math::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::{
                    default_types::{FrElement, FrField},
                    pairing::BLS12381AtePairing,
                },
            traits::IsPairing,
        },
        field::element::FieldElement,
    };
    use ndarray::array;
    use srs::create_srs;
    use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;


    use crate::bikzg::traits::IsCommitmentScheme;

    // type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;

    use super::*;

    #[allow(clippy::upper_case_acronyms)]
    type KZG = BivariateKateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

    #[test]
    fn kzg_1() {
        // (x+1)(y+1) = xy + y + x + 1 
        let srs = create_srs((2, 2));
        let bikzg = KZG::new(srs);
        // let p = Polynomial::<FrElement>::new(&[FieldElement::one(), FieldElement::one()]);
        let coefficients = array![
            [FrElement::from(1), FrElement::from(1)],
            [FrElement::from(1), FrElement::from(1)]
        ];
        let bp = BivariatePolynomial::new(coefficients);
        // let (qxy, qy) = bp.ruffini_division(&-FieldElement::<FrField>::one(),& -FieldElement::<FrField>::one());
        let p_commitment: <BLS12381AtePairing as IsPairing>::G1Point = bikzg.commit_bivariate(&bp);
        let x = FieldElement::zero(); 
        let y = FrElement::from(10);
        let evaluation = bp.evaluate(&x, &y);
        let proof = bikzg.open(&x, &y, &evaluation,&bp);
        
        println!("proof: {:?}", proof.0);
        assert!(bikzg.verify(&x, &y,&evaluation, &p_commitment, &proof));
    }


    #[test]
    fn kzg_2() {
        let bikzg = KZG::new(create_srs((4, 4)));
        // let p = Polynomial::<FrElement>::new(&[FieldElement::one(), FieldElement::one()]);

        let coefficients = array![
            [FrElement::from(2),FrElement::from(1), FrElement::from(1)],//(2+x+x2) =2 
            [FrElement::from(1), FrElement::from(1),FrElement::from(1)],//1
            [FrElement::from(5), FrElement::from(2),FrElement::from(0)],//1
            [FrElement::from(3), FrElement::from(0),FrElement::from(1)],//1
        ];

        let bp = BivariatePolynomial::new(coefficients);
        // let (qxy, qy) = bp.ruffini_division(&-FieldElement::<FrField>::one(),& -FieldElement::<FrField>::one());
        let p_commitment: <BLS12381AtePairing as IsPairing>::G1Point = bikzg.commit_bivariate(&bp);
        let x = -FieldElement::one();
        let y = -FieldElement::one();
        let evaluation = bp.evaluate(&x, &y);
        let fake_evaluation = FrElement::from(1000);
        let proof = bikzg.open(&x, &y, &fake_evaluation,&bp);
        // let fake_proof = (BLS12381Curve::generator(),BLS12381Curve::generator());
        
        assert!(bikzg.verify(&x, &y,&evaluation, &p_commitment, &proof));

    }
}