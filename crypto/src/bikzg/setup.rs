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

// /// Generate G1 Points for SRS
// pub fn g1_points_srs(dims: (usize, usize), taus: (FrElement, FrElement)) -> Vec<Vec<G1Point>> {
//     let (tau, theta) = taus;
//     let powers_of_tau_theta = vandermonde_matrix(&tau, &theta, dims.0, dims.1);

//     let g1_generator: BlsG1point = <BLS12381Curve as IsEllipticCurve>::generator();
//     let mut g1_points_2d: Vec<Vec<BlsG1point>> = Vec::with_capacity(dims.0);

//     for i in 0..dims.0 {
//         let mut row = vec![g1_generator.clone(); dims.1];
//         row.par_iter_mut()
//             .zip(&powers_of_tau_theta[i])
//             .for_each(|(g1, tau_theta)| {
//                 *g1 = g1.operate_with_self(tau_theta.representative());
//             });
//         g1_points_2d.push(row);
//     }

//     g1_points_2d
// }

// /// Create Structured Reference String (SRS)
// pub fn create_srs(dim_x: usize, dim_y: usize) -> StructuredReferenceString<
//     <BLS12381AtePairing as IsPairing>::G1Point,
//     <BLS12381AtePairing as IsPairing>::G2Point,
// > {
//     use rand::Rng;

//     let mut rng = rand::thread_rng();
//     let tau = FrElement::new(U256 {
//         limbs: [
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//         ],
//     });

//     let theta = FrElement::new(U256 {
//         limbs: [
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//             rng.gen::<u64>(),
//         ],
//     });

//     let g1_points = g1_points_srs((dim_x, dim_y), (tau.clone(), theta.clone()));
//     let g1_flattened: Vec<_> = g1_points.into_iter().flatten().collect::<Vec<_>>();

//     // let powers_main_group: Vec<_> = g1_points.clone().into_iter().flatten().collect::<Vec<_>>();

//     // let converted_g1_points = g1_flattened
//     //     .iter()
//     //     .map(|point| PointConversion::to_icicle(point))
//     //     .collect::<Vec<_>>();

//     let g2_generator = BLS12381TwistCurve::generator();
//     let g2_points = [
//         g2_generator.clone(),
//         g2_generator.operate_with_self(tau.representative()),
//         g2_generator.operate_with_self(theta.representative()),
//     ];

//     StructuredReferenceString::new(
//         dim_x,
//         dim_y,
//         &g1_flattened,
//         &g2_points,
//         // converted_g1_points,
//     )
// }
