// bikzg 모듈의 최상위 관리 파일

pub mod setup;
pub mod commit;
pub mod open;
pub mod verify;
pub mod srs;
// pub mod utils;

// SRS 및 관련 타입들을 모듈 외부로 노출
pub use srs::{StructuredReferenceString, G1Point, G2Point};

// 공통 트레이트 및 타입들 노출
// pub use utils::{PointConversion, ToIcicle};

// Bivariate KZG 구조체 및 구현 노출
pub use commit::BivariateKateZaveruchaGoldberg;

// 외부 라이브러리 타입 재노출
pub use icicle_bls12_381::curve::{G1Affine, G2Affine};
