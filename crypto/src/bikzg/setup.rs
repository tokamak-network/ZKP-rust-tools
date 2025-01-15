use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            default_types::FrElement,
        };
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
// use lambdaworks_groth16::common::G2Point;
type BlsG1point = ShortWeierstrassProjectivePoint<BLS12381Curve>;

/// Generate a Vandermonde Matrix
pub fn vandermonde_matrix(
    tau: &FrElement,
    theta: &FrElement,
    row_len: usize,
    col_len: usize,
) -> Vec<Vec<FrElement>> {
    let mut vec = Vec::with_capacity(row_len);

    for i in 0..row_len {
        let row_base = theta.pow(i);
        let mut row = Vec::with_capacity(col_len);

        for j in 0..col_len {
            row.push(row_base.clone() * tau.pow(j));
        }

        vec.push(row);
    }

    vec
}
