use icicle_bls12_381::curve::{G1Affine, G2Affine};
use icicle_core::field::Field;
use icicle_core::curve::Affine;

/// Structured Reference String (SRS) 데이터 구조
pub struct StructuredReferenceString {
    pub dimension_x: usize,
    pub dimension_y: usize,
    pub powers_main_group: Vec<G1Affine>,
    pub powers_secondary_group: [G2Affine; 3],
}

impl StructuredReferenceString {
    /// 새로운 SRS를 생성합니다.
    pub fn new(
        dimension_x: usize,
        dimension_y: usize,
        powers_main_group: Vec<G1Affine>,
        powers_secondary_group: [G2Affine; 3],
    ) -> Self {
        Self {
            dimension_x,
            dimension_y,
            powers_main_group,
            powers_secondary_group,
        }
    }

    /// G1 포인트를 평탄화합니다.
    pub fn flatten_partitioned_g1_points(&self, x_len: usize, y_len: usize) -> Vec<G1Affine> {
        let mut chunk_iter = self.powers_main_group.chunks(self.dimension_x);
        let mut output = Vec::new();
        for _ in 0..y_len {
            if let Some(chunk) = chunk_iter.next() {
                output.extend(chunk.iter().take(x_len).cloned());
            }
        }
        output
    }
}
