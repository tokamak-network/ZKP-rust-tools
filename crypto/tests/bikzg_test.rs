use zkp_rust_tools_crypto::bikzg::{
    BivariateKateZaveruchaGoldberg, srs::create_srs, traits::IsCommitmentScheme,
};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        default_types::{FrElement, FrField},
        pairing::BLS12381AtePairing,
    },
    field::element::FieldElement,
};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use ndarray::Array2;

#[allow(clippy::upper_case_acronyms)]
type KZG = BivariateKateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

#[test]
fn test_kzg() {
    let test_size = 4;

    // Create the SRS
    let srs = create_srs((test_size, test_size));
    let bikzg = KZG::new(srs);

    // Create a bivariate polynomial
    let matrix = Array2::from_elem((test_size, test_size), FrElement::from(1));
    let bp = BivariatePolynomial::new(matrix);

    // Commit to the polynomial
    let p_commitment = bikzg.commit_bivariate(&bp);

    // Test evaluation, opening, and verification
    let x = FieldElement::zero();
    let y = FrElement::from(10);
    let evaluation = bp.evaluate(&x, &y);

    let proof = bikzg.open( &x, &y, &evaluation, &bp);

    println!("proof: {:?}", proof);
    assert!(bikzg.verify(
        &x,
        &y,
        &evaluation,
        &p_commitment,
        &proof
    ));
}
