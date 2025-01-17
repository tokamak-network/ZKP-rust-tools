use icicle_bls12_381::curve::{
    CurveCfg, G1Affine, G1Projective, G2Affine, G2CurveCfg, ScalarField
};
use icicle_core::traits::Arithmetic;
use icicle_core::traits::FieldImpl;
use icicle_core::curve::Curve;
use rand::Rng; // 랜덤 생성에 필요

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
    /// - `dimension_x, dimension_y`: (x차, y차)에 맞춰 G1 점이 (x+1)*(y+1)개 필요
    /// - 내부에서 tau, theta 를 랜덤으로 뽑는다.
    pub fn create_srs(
        dimension_x: usize,
        dimension_y: usize,
    ) -> Self {
        // 1) tau, theta 랜덤 생성
        let mut rng = rand::thread_rng();
        let tau_array: [u32; 8] = rng.gen();
        let theta_array: [u32; 8] = rng.gen();
        let tau = ScalarField::from(tau_array);
        let theta = ScalarField::from(theta_array);

        // 2) G1, G2용 임의 projective points를 생성
        //    여기서는 "단 1개씩"만 생성해 base로 삼는다.
        let g1_points = CurveCfg::generate_random_projective_points(1);
        let g2_points = G2CurveCfg::generate_random_projective_points(1);

        // 3) (dimension_x+1)*(dimension_y+1)개의 G1 점을 만들자
        //    G1_{i,j} = g1_points[0] * ( tau^j * theta^i )
        let total_needed = (dimension_x + 1) * (dimension_y + 1);
        let mut powers_main_group = Vec::with_capacity(total_needed);

        // g1_points[0]가 projective 형태이므로, 여기에 (tau^j * theta^i) 곱해가며 만든다
        let base_g1 = g1_points[0];

        for i in 0..=dimension_y {
            for j in 0..=dimension_x {
                let exponent = tau.pow(j) * theta.pow(i);
                let point_proj: G1Projective = base_g1 * exponent;
                let point_aff: G1Affine = point_proj.into();
                powers_main_group.push(point_aff);
            }
        }

        // 4) G2에서 3개만 만들기: [base_g2, base_g2 * tau, base_g2 * theta]
        let base_g2 = g2_points[0];
        let g2_0 = base_g2;
        let g2_1 = base_g2 * tau;
        let g2_2 = base_g2 * theta;
        let powers_secondary_group: [G2Affine; 3] = [
            g2_0.into(),
            g2_1.into(),
            g2_2.into(),
        ];

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

/// 간단한 Vandermonde (필요하다면 그대로 유지하거나, 사용하지 않으면 지워도 됨)
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

    #[test]
    fn test_create_srs_random_tau_theta() {
        // 이제 create_srs 함수가 내부적으로 tau, theta를 뽑기 때문에
        // 따로 tau를 입력하지 않아도 됨
        let srs = StructuredReferenceString::create_srs(2, 2);

        // G1 포인트 개수 = (2+1)*(2+1) = 9
        assert_eq!(srs.powers_main_group.len(), 9);

        // G2 포인트는 3개
        assert_eq!(srs.powers_secondary_group.len(), 3);
    }
}