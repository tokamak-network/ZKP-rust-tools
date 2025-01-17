use icicle_bls12_381::curve::{
    self, CurveCfg, G1Affine, G1Projective, G2Affine, G2CurveCfg, G2Projective, ScalarField
};
use std::ops::Mul;
use icicle_core::{curve::Affine, traits::FieldImpl};
use icicle_core::{curve::Curve, msm, traits::GenerateRandom};

/// Structured Reference String (SRS) 데이터 구조
pub struct StructuredReferenceString {
    pub dimension_x: usize,
    pub dimension_y: usize,
    /// We store G1 in **affine** form
    pub powers_main_group: Vec<G1Affine>,
    /// We store just 3 G2 points in **affine** form
    pub powers_secondary_group: [G2Affine; 3],
}

impl StructuredReferenceString {
    /// SRS 생성
    /// - `dimension_x, dimension_y`: (x차, y차)에 맞춰 필요한 G1 점 개수 = (x+1)*(y+1).
    /// - `tau`: 임의(신뢰) 스칼라
    pub fn create_srs(
        dimension_x: usize,
        dimension_y: usize,
        tau: &ScalarField,
    ) -> Self {
        // 1) Generate a vector of random projective points for G1, but we only actually need one
        //    or we might use g1[0] as the "generator"
        let g1 = CurveCfg::generate_random_projective_points(dimension_y + 1);
        let g2 = G2CurveCfg::generate_random_projective_points(dimension_y + 1);

        // 2) total_needed = (x+1)*(y+1)
        let total_needed = (dimension_x + 1) * (dimension_y + 1);

        // 3) G1 상에서 [ (g1[0]) * (tau^i ) ] -> Affine
        let mut powers_main_group = Vec::with_capacity(total_needed);

        let mut cur_tau = ScalarField::one();
        for _ in 0..total_needed {
            // Step A: take the first G1Projective as "base"
            let base_g1_proj: G1Projective = g1[0];
            // Step B: multiply by tau^i (cur_tau)
            let point_proj = base_g1_proj * cur_tau;
            // Step C: convert to G1Affine
            let point_aff: G1Affine = point_proj.into(); 
            powers_main_group.push(point_aff);

            // Next power
            cur_tau = cur_tau * *tau;
        }

        // 4) G2에서 3개만 [ g2[0]*tau^0, g2[0]*tau^1, g2[0]*tau^2 ], all Affine
        let mut tmp_g2 = Vec::with_capacity(3);
        let mut cur_tau2 = ScalarField::one();
        for _ in 0..3 {
            let base_g2_proj: G2Projective = g2[0];
            let pt_proj = base_g2_proj * cur_tau2;
            let pt_aff: G2Affine = pt_proj.into(); 
            tmp_g2.push(pt_aff);
            cur_tau2 = cur_tau2 * *tau;
        }
        // convert Vec of length 3 -> array of length 3
        let powers_secondary_group: [G2Affine; 3] = [tmp_g2[0], tmp_g2[1], tmp_g2[2]];

        Self {
            dimension_x,
            dimension_y,
            powers_main_group,
            powers_secondary_group,
        }
    }

    /// G1 포인트를 (y행 x열) 순서로 평탄화
    /// - x_len: x방향 개수
    /// - y_len: y방향 개수
    pub fn flatten_partitioned_g1_points(&self, x_len: usize, y_len: usize) -> Vec<G1Affine> {
        let mut chunk_iter = self.powers_main_group.chunks(self.dimension_x + 1);
        let mut output = Vec::new();

        for _ in 0..y_len {
            if let Some(chunk) = chunk_iter.next() {
                output.extend(chunk.iter().take(x_len).cloned());
            }
        }
        output
    }
}

/// 간단한 Vandermonde
pub fn compute_vandemonde(bases: &[ScalarField], max_degree: usize) -> Vec<Vec<ScalarField>> {
    let rows = max_degree + 1;
    let cols = bases.len();

    let mut table = vec![vec![ScalarField::one(); cols]; rows];

    for j in 0..cols {
        let base_j = bases[j];
        let mut power = ScalarField::one();
        for i in 0..rows {
            table[i][j] = power;
            power = power * base_j;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_381::curve::CurveCfg;
    use icicle_core::traits::FieldImpl;

    #[test]
    fn test_create_srs() {
        let tau = ScalarField::from_u32(5);
        let srs = StructuredReferenceString::create_srs(2, 2, &tau);

        // We expect total_needed = (2+1)*(2+1) = 9
        assert_eq!(srs.powers_main_group.len(), 9);
        assert_eq!(srs.powers_secondary_group.len(), 3);

        // The first is g1[0]* tau^0 => effectively g1[0]. We can't directly assume generator,
        // because we used generate_random_projective_points. 
        // But we can check that we indeed have 9 distinct points, or do any consistent check.
        // e.g. we can check srs.powers_main_group[0] != srs.powers_main_group[1], etc.
    }

    #[test]
    fn test_flatten_partitioned_g1_points() {
        let tau = ScalarField::from_u32(3);
        let srs = StructuredReferenceString::create_srs(3, 1, &tau);
        assert_eq!(srs.powers_main_group.len(), 8);

        let flattened = srs.flatten_partitioned_g1_points(2, 1);
        assert_eq!(flattened.len(), 2);

        // They should match the first 2 points in the first "row"
        assert_eq!(flattened[0], srs.powers_main_group[0]);
        assert_eq!(flattened[1], srs.powers_main_group[1]);
    }

    #[test]
    fn test_compute_vandemonde() {
        let bases = [ScalarField::from_u32(2), ScalarField::from_u32(3)];
        let table = compute_vandemonde(&bases, 2);

        assert_eq!(table.len(), 3);
        assert_eq!(table[0].len(), 2);

        // row0 => [1,1]
        assert_eq!(table[0][0], ScalarField::one());
        assert_eq!(table[0][1], ScalarField::one());

        // row1 => [2,3]
        assert_eq!(table[1][0], ScalarField::from_u32(2));
        assert_eq!(table[1][1], ScalarField::from_u32(3));

        // row2 => [4,9]
        assert_eq!(table[2][0], ScalarField::from_u32(4));
        assert_eq!(table[2][1], ScalarField::from_u32(9));
    }
}