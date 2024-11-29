// Export the SRS-related utilities and structures
pub mod srs;
pub mod setup;
pub mod commit;
pub mod traits;
pub mod utils;
pub use srs::{StructuredReferenceString, G2Point};
pub use traits::{IsCommitmentScheme, PointConversion, ToIcicle};
use core::{marker::PhantomData, mem};

// Re-export types from external libraries for convenience
pub use lambdaworks_math::{
    field::element::FieldElement,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve, 
                pairing::BLS12381AtePairing,
            },
        },
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