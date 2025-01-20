use icicle_bls12_381::curve::{
    G1Projective as BLS12381G1Projective, 
    ScalarField,
    G2Projective as BLS12381G2Projective,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::ate_pairing;
use lambdaworks_math::field::traits::IsField; // Fq12 비교 등
use crate::bikzg_icicle::{srs::StructuredReferenceString, BivariateKateZaveruchaGoldbergIcicle};

impl BivariateKateZaveruchaGoldbergIcicle {
    /// 실제 Pairing 기반 검증 로직 (2단계 univariate KZG) 예시
    /// - x, y: 증명하고 싶은 좌표 (x^*, y^*)
    /// - eval: p(x^*, y^*) = 평가값
    /// - p_commit: bivariate poly p(X,Y)의 커밋
    /// - proof: open() 결과 (q_xy_commit, q_y_commit)
    pub fn verify(
        &self,
        x: &ScalarField,
        y: &ScalarField,
        eval: &ScalarField,
        p_commit: &BLS12381G1Projective,
        proof: &(BLS12381G1Projective, BLS12381G1Projective),
    ) -> bool {
        let (q_xy_commit, q_y_commit) = proof;

        //-------------------------------------
        // 1) X 단계 검증
        //
        //   p'(X) = p(X, y^*)           (y를 y^*로 고정한 단변수)
        //   p'(x^*) = eval
        //   => p'(X) - eval = (X - x^*) * q'(X)
        //
        //   pairing(p'(X) - eval·g1(??), g2_{(X-x^*)}) == pairing(q_xy_commit, g2_base)
        //
        //   *주의*: 실제로 "p'(X) - eval"에 대한 커밋(p'(X) 커밋 - eval * g1(0))과
        //   "X-x^*"에 해당하는 G2 포인트를 SRS에서 찾아야 합니다.
        //-------------------------------------

        // (a) p'(X) 커밋을 구하기 위해 partial_eval_y(...) 같은 함수를 만들 수도 있음
        //     여기서는 "가상의 함수"로 처리
        let p_x_commit = match self.partial_eval_y(p_commit, y) {
            Some(commit) => commit,
            None => return false,
        };

        // (b) p'(X) - eval
        //     g1_index(0) 가 "X^0Y^0"에 해당한다고 가정해서, 그 점을 빼는 식.
        //     (실제로는 'p'(x^*)= eval'을 g1_base * eval 로 계산해 빼줄 수도 있음)
        let g1_identity = self.srs.powers_main_group[0]; 
        let g1_identity_proj: BLS12381G1Projective = g1_identity.into();
        let p_x_minus_eval = p_x_commit - (g1_identity_proj * (*eval));

        // (c) (X-x^*)에 해당하는 g2 포인트 (예시: srs.powers_secondary_group[1]이 (tau), etc.)
        //     여기서는 "가상의 로직"으로, 임의로 정함
        let g2_for_x_diff: BLS12381G2Projective = self.srs.powers_secondary_group[1].into();

        // (d) pairing 비교
        let lhs_x = ate_pairing(&p_x_minus_eval, &g2_for_x_diff);
        let rhs_x = ate_pairing(q_xy_commit, &self.srs.powers_secondary_group[0].into()); 
        //  ↑ 여기서 srs.powers_secondary_group[0] = g2 base?

        let pairing_check_x = lhs_x == rhs_x;

        //-------------------------------------
        // 2) Y 단계 검증
        //
        //   p''(Y) = p(x^*, Y)
        //   p''(y^*) = eval
        //   => p''(Y) - eval = (Y - y^*) * q''(Y)
        //
        //   pairing( p''(Y) - eval, g2_{(Y-y^*)} ) == pairing( q_y_commit, g2_base )
        //-------------------------------------

        let p_y_commit = match self.partial_eval_x(p_commit, x) {
            Some(commit) => commit,
            None => return false,
        };
        let p_y_minus_eval = p_y_commit - (g1_identity_proj * (*eval));

        // (가정) srs.powers_secondary_group[2]가 (Y-y^*) 역할?
        let g2_for_y_diff: BLS12381G2Projective = self.srs.powers_secondary_group[2].into();

        let lhs_y = ate_pairing(&p_y_minus_eval, &g2_for_y_diff);
        let rhs_y = ate_pairing(q_y_commit, &self.srs.powers_secondary_group[0].into());

        let pairing_check_y = lhs_y == rhs_y;

        pairing_check_x && pairing_check_y
    }
}

impl BivariateKateZaveruchaGoldbergIcicle {
    /// (예시) "bivariate 커밋" p_commit에서 y를 y^*로 부분평가한 "단변수 커밋"을 구하는 가상의 함수
    /// 실제로는 p_commit을 전개한 다음 y^* 항들을 합산해야 하므로, 
    /// 보통 p(X,Y) = Σ_i Σ_j coeff_{i,j} * X^j Y^i => Y^i -> (y^*)^i
    /// 식으로 다시 합쳐야 합니다.
    pub fn partial_eval_y(
        &self,
        p_commit: &BLS12381G1Projective,
        y_value: &ScalarField,
    ) -> Option<BLS12381G1Projective> {
        // 구현 난이도가 있어 "None" 처리하거나, 
        // 혹은 bivariate->univariate 변환 로직을 직접 짜야 함.
        //
        // 여기서는 "데모"로써, 그냥 p_commit를 그대로 반환하도록 하겠습니다. (실제로는 잘못됨)
        Some(*p_commit)
    }

    /// (예시) p_commit에서 x를 x^*로 부분평가
    pub fn partial_eval_x(
        &self,
        p_commit: &BLS12381G1Projective,
        x_value: &ScalarField,
    ) -> Option<BLS12381G1Projective> {
        // 동일한 이유로 실제 구현 없음
        Some(*p_commit)
    }
}