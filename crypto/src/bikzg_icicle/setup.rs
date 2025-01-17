use icicle_runtime::{runtime, Device};
use icicle_core::poly::{DensePoly, Polynomial};
use icicle_core::scalar::ScalarField;
use icicle_core::pairing::{PairingEngine, G1Affine, G2Affine};

struct Setup<E: PairingEngine> {
    g1_powers: Vec<G1Affine<E>>, // G1에서의 지수들
    g2_powers: Vec<G2Affine<E>>, // G2에서의 지수들
}

impl<E: PairingEngine> Setup<E> {
    /// Generate the public parameters for KZG
    pub fn new(max_degree: usize, secret: E::ScalarField) -> Self {
        let mut g1_powers = vec![];
        let mut g2_powers = vec![];

        // G1과 G2의 생성자
        let g1 = E::G1Affine::prime_subgroup_generator();
        let g2 = E::G2Affine::prime_subgroup_generator();

        let mut current_power = secret;
        for _ in 0..=max_degree {
            g1_powers.push(g1.mul(current_power).into_affine()); // G1에서의 지수 \( g1^{s^i} \)
            g2_powers.push(g2.mul(current_power).into_affine()); // G2에서의 지수 \( g2^{s^i} \)
            current_power *= secret; // 다음 지수로 진행
        }

        Setup { g1_powers, g2_powers }
    }
}