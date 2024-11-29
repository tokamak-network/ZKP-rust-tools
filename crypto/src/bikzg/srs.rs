pub mod srs {
    pub use crate::bikzg::G1Point;
    pub use crate::bikzg::G2Point;
    pub use crate::bikzg::StructuredReferenceString;
}
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve, 
        twist::BLS12381TwistCurve,
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
// use lambdaworks_math::field::traits::IsField;
use rayon::prelude::*;
use rand::Rng;
use icicle_bls12_381::curve::CurveCfg;
use icicle_core::curve::Affine;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;

#[derive(PartialEq, Clone, Debug)]
pub struct StructuredReferenceString<G1Point, G2Point> {
    pub dimention_x: usize, 
    pub dimention_y: usize,
    pub powers_main_group: Vec<G1Point>,
    pub powers_secondary_group: [G2Point; 3],// 1 , tau, theta 
    // pub converted_g1_points: Vec<Affine<CurveCfg>>
}

pub type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
pub type G2Point = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    pub fn new(
        dim_x: usize, 
        dim_y: usize,
        powers_main_group: &[G1Point], 
        powers_secondary_group: &[G2Point; 3], 
        // converted_g1_points: &Vec<Affine<CurveCfg>>,
    ) -> Self {
        Self {
            dimention_x: dim_x, 
            dimention_y: dim_y,
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
            // converted_g1_points: converted_g1_points.clone()
        }
    }

    pub fn flatten_partitioned_g1_points(&self, x_len: usize, y_len: usize) -> Vec<G1Point> {
        let mut chunk_iter = self.powers_main_group.chunks(self.dimention_x);
        let mut output: Vec<G1Point> = vec![];
        for _ in 0..y_len{
            // let dd = chunk_iter.next();
            // dd.iter().take(x_len).cloned().collect();
            output.extend( chunk_iter.next().unwrap().iter().take(x_len).cloned());
        }

        output
    }

    // pub fn flatten_partitioned_g1_points_icicle(&self, x_len: usize, y_len: usize) -> Vec<Affine<CurveCfg>> {
    //     let mut chunk_iter = self.converted_g1_points.chunks(self.dimention_x);
    //     let mut output: Vec<Affine<CurveCfg>> = vec![];
    //     for _ in 0..y_len{
    //         // let dd = chunk_iter.next();
    //         // dd.iter().take(x_len).cloned().collect();
    //         output.extend( chunk_iter.next().unwrap().iter().take(x_len).cloned());
    //     }

    //     output
    // }
}



/// Generates a structured reference string (SRS) for the KZG scheme.
///
/// # Parameters:
/// - `dims`: Tuple `(rows, cols)` indicating the dimensions of the SRS.
/// - `taus`: Tuple `(tau, theta)` where `tau` and `theta` are toxic waste elements.
///
/// # Returns:
/// - A `StructuredReferenceString` containing G1 and G2 elements.
pub fn create_srs(
    dims: (usize, usize),
) -> StructuredReferenceString<G1Point, G2Point> {
    let mut rng = rand::thread_rng();

    // Random toxic wastes (tau, theta)
    let tau = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });

    let theta = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });

    // Generate G1 and G2 generators
    let g1_generator = BLS12381Curve::generator();
    let g2_generator = BLS12381TwistCurve::generator();

    // Compute powers of tau and theta for G1 points
    let powers_of_tau_theta = compute_vandemonde(&tau, &theta, dims.0, dims.1);

    let g1_points = powers_of_tau_theta
        .par_iter()
        .map(|row| {
            row.iter()
                .map(|scalar| g1_generator.operate_with_self(scalar.representative()))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();
    
    // let converted_g1_points = powers_main_group
    //     .iter()
    //     .map(|point| PointConversion::to_icicle(point))
    //     .collect::<Vec<_>>();

    // Compute powers of tau and theta for G2 points
    let g2_points = [
        g2_generator.clone(),
        g2_generator.operate_with_self(tau.representative()),
        g2_generator.operate_with_self(theta.representative()),
    ];

    // Return the structured reference string
    StructuredReferenceString::new(dims.0, dims.1, &g1_points, &g2_points)
}

/// Computes a Vandermonde matrix for toxic wastes tau and theta.
///
/// # Parameters:
/// - `tau`: Toxic waste scalar.
/// - `theta`: Another toxic waste scalar.
/// - `rows`: Number of rows.
/// - `cols`: Number of columns.
///
/// # Returns:
/// - A 2D vector containing the computed Vandermonde matrix.
fn compute_vandemonde(
    tau: &FrElement,
    theta: &FrElement, 
    row_len: usize,
    col_len: usize
) -> Vec<Vec<FrElement>> {

    let mut vec: Vec<Vec<FrElement>> = Vec::with_capacity(row_len);

    for _ in 0..row_len {
        vec.push(Vec::with_capacity(col_len));
    }

    for i in 0..row_len {
        let y_row = theta.pow(i);

        let mut row: Vec<FrElement> = Vec::with_capacity(row_len);
        for j in 0..col_len{
            row.push(y_row.clone() * tau.pow(j));// TODO check for not being wrong.
        }
        vec[i]=row;
    }
    vec
}
