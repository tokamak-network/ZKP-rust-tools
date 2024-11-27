mod setup;
mod commit;
mod open;
mod verify;
mod srs;
mod utils;

use lambdaworks_math::{
    field::element::FieldElement,
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField},
};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;

fn main() {
    // Step 1: Setup (Generate SRS)
    let dim_x = 1024;
    let dim_y = 1024;
    let srs = setup::create_srs(dim_x, dim_y);
    println!("SRS successfully generated.");

    // Step 2: Create a bivariate polynomial
    let coefficients = vec![
        vec![FrElement::from(2), FrElement::from(3)],
        vec![FrElement::from(5), FrElement::from(7)],
    ];
    let bp = BivariatePolynomial::new(coefficients);
    println!("Bivariate polynomial created.");

    // Step 3: Commit to the polynomial
    let p_commitment = commit::commit_bivariate(&srs, &bp);
    println!("Commitment generated: {:?}", p_commitment);

    // Step 4: Generate a proof
    let x = FieldElement::<FrField>::from(1);
    let y = FieldElement::<FrField>::from(2);
    let evaluation = bp.evaluate(&x, &y);
    println!("Evaluation at ({}, {}): {:?}", x, y, evaluation);

    let proof = open::open(&srs, &x, &y, &evaluation, &bp);
    println!("Proof generated: {:?}", proof);

    // Step 5: Verify the proof
    let is_valid = verify::verify(&srs, &x, &y, &evaluation, &p_commitment, &proof);
    println!("Verification result: {}", is_valid);

    // Final assertion
    if is_valid {
        println!("Proof successfully verified!");
    } else {
        println!("Proof verification failed.");
    }
}
