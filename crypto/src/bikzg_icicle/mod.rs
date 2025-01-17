//! Example module that uses `icicle_bipolynomial` for bivariate polynomial logic,
//! instead of the lambdaworks-based `bipolynomial/mod.rs`.

pub mod srs;
pub mod commit;
// 만약 verify를 따로 작성한다면, 여기서 pub mod verify; 로 선언 가능(현재 제외)

// 아래는 KZG 스킴의 예시 구조체
use srs::StructuredReferenceString;

/// Bivariate KZG 스킴 (icicle 버전)
///  - `F` : 필드(Generic),  `P` : 페어링 구조
pub struct BivariateKateZaveruchaGoldbergIcicle<F> {
    pub srs: StructuredReferenceString,
    pub _marker: core::marker::PhantomData<F>,
}

impl<P> BivariateKateZaveruchaGoldbergIcicle<P> {
    /// SRS로부터 스킴 생성
    pub fn new(srs: StructuredReferenceString) -> Self {
        Self {
            srs,
            _marker: core::marker::PhantomData,
        }
    }
}

// 여기까지는 verify 등의 로직 없이, 스킴 구조만 선언해 둠.