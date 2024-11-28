use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::{BLS12381Curve},
        default_types::FrElement,
    },
    // field::element::FieldElement,
};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use crate::bikzg::srs::{StructuredReferenceString, G1Point, G2Point};
use lambdaworks_math::unsigned_integer::element::U256;

/// Generate a Vandermonde Matrix
///
/// # Parameters:
/// - `tau`: Random scalar value representing the toxic waste.
/// - `theta`: Random scalar value representing the secondary toxic waste.
/// - `row_len`: Number of rows in the matrix.
/// - `col_len`: Number of columns in the matrix.
///
/// # Returns:
/// - `Vec<Vec<FrElement>>`: A 2D vector representing the Vandermonde matrix.
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

/// Generate G1 Points for SRS
///
/// # Parameters:
/// - `dims`: Dimensions of the SRS (number of rows and columns).
/// - `taus`: A tuple containing \( \tau \) and \( \theta \).
///
/// # Returns:
/// - `Vec<Vec<G1Point>>`: A 2D vector of G1 points for the SRS.
pub fn g1_points_srs(dims: (usize, usize), taus: (FrElement, FrElement)) -> Vec<Vec<G1Point>> {
    let (tau, theta) = taus;
    let powers_of_tau_theta = vandermonde_matrix(&tau, &theta, dims.0, dims.1);

    let g1_generator: G1Point = <BLS12381Curve as IsEllipticCurve>::generator();
    let mut g1_points_2d = Vec::with_capacity(dims.0);

    for i in 0..dims.0 {
        let mut row = vec![g1_generator.clone(); dims.1];
        row.par_iter_mut()
            .zip(&powers_of_tau_theta[i])
            .for_each(|(g1, tau_theta)| {
                *g1 = g1.operate_with_self(tau_theta.representative());
            });
        g1_points_2d.push(row);
    }

    g1_points_2d
}

/// Create Structured Reference String (SRS)
///
/// # Parameters:
/// - `dim_x`: Number of rows in the SRS.
/// - `dim_y`: Number of columns in the SRS.
///
/// # Returns:
/// - `StructuredReferenceString<G1Point, G2Point>`: The SRS containing G1 and G2 points.
pub fn create_srs(dim_x: usize, dim_y: usize) -> StructuredReferenceString<G1Point, G2Point> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
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

    let g1_points = g1_points_srs((dim_x, dim_y), (tau.clone(), theta.clone()));
    let g1_flattened: Vec<_> = g1_points.into_iter().flatten().collect();

    let g2_generator: G2Point = <BLS12381Curve as IsEllipticCurve>::twist_generator();
    let g2_points = [
        g2_generator.clone(),
        g2_generator.operate_with_self(tau.representative()),
        g2_generator.operate_with_self(theta.representative()),
    ];

    StructuredReferenceString::new(dim_x, dim_y, &g1_flattened, &g2_points)
}
