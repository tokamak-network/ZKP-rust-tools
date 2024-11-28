use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::{BLS12381Curve},
    },
    cyclic_group::IsGroup,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
// use serde::{Serialize, Deserialize};

/// G1Point represents a point on the primary elliptic curve group.
pub type G1Point = <BLS12381Curve as IsGroup>::Element;

/// G2Point represents a point on the secondary elliptic curve group.
pub type G2Point = <BLS12381TwistCurve as IsGroup>::Element;

/// Structured Reference String (SRS)
///
/// SRS contains precomputed elliptic curve points used in KZG commitment schemes.
#[derive(PartialEq, Clone, Debug)]
pub struct StructuredReferenceString<G1Point, G2Point> {
    pub dimention_x: usize,
    pub dimention_y: usize,
    pub powers_main_group: Vec<G1Point>, // G1 group points
    pub powers_secondary_group: [G2Point; 3], // G2 group points: [G2, τG2, θG2]
}

impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    /// Create a new SRS
    pub fn new(
        dimention_x: usize,
        dimention_y: usize,
        powers_main_group: &[G1Point],
        powers_secondary_group: &[G2Point; 3],
    ) -> Self {
        Self {
            dimention_x,
            dimention_y,
            powers_main_group: powers_main_group.to_vec(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }

    /// Flatten partitioned G1 points into a 1D vector for a specific dimension
    ///
    /// # Parameters:
    /// - `x_len`: The number of points to extract in the x-dimension.
    /// - `y_len`: The number of points to extract in the y-dimension.
    ///
    /// # Returns:
    /// - `Vec<G1Point>`: Flattened G1 points.
    pub fn flatten_partitioned_g1_points(&self, x_len: usize, y_len: usize) -> Vec<G1Point> {
        let mut chunk_iter = self.powers_main_group.chunks(self.dimention_x);
        let mut output: Vec<G1Point> = vec![];
        for _ in 0..y_len {
            output.extend(chunk_iter.next().unwrap().iter().take(x_len).cloned());
        }
        output
    }
}
