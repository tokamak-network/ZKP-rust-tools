// pub mod prover;
pub mod srs;
pub mod utils;
pub mod traits;
pub mod commit;
pub mod open;
pub mod setup;
pub mod verify;

use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
};

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
