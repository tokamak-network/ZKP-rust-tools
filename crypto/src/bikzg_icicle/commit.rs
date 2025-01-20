use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};
use icicle_core::msm;
use icicle_bls12_381::{curve::{
    G1Projective as BLS12381G1Projective, ScalarField,
}, polynomials::DensePolynomial};
use zkp_rust_tools_math::icicle_bipolynomial::{BivariatePolynomial, DensePolynomialExt};

use super::BivariateKateZaveruchaGoldbergIcicle; 

impl BivariateKateZaveruchaGoldbergIcicle {
    pub fn commit_bivariate(
        &self,
        poly: &BivariatePolynomial, 
    ) -> BLS12381G1Projective {
        let scalars: Vec<ScalarField> = poly
            .flatten_out()
            .iter()
            .cloned()
            .collect();

        let points = &self.srs.powers_main_group;
        assert_eq!(scalars.len(), points.len());

        let host_scalars = HostSlice::from_slice(&scalars);
        let host_points = HostSlice::from_slice(points);

        let mut msm_result_dev = DeviceVec::<BLS12381G1Projective>::device_malloc(1).unwrap();

        let mut stream = IcicleStream::create().unwrap();
        let mut cfg = msm::MSMConfig::default();
        cfg.stream_handle = *stream;
        cfg.is_async = true;

        msm::msm(
            host_scalars, 
            host_points,
            &cfg,
            &mut msm_result_dev[..]
        ).unwrap();

        let mut msm_result_host = vec![BLS12381G1Projective::zero(); 1];
        stream.synchronize().unwrap();
        msm_result_dev
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_result_host[..]))
            .unwrap();
        stream.destroy().unwrap();

        // 7) 최종 결과 반환
        msm_result_host[0]
    }

    pub fn commit_univariate(
        &self,
        poly: &DensePolynomial, 
    ) -> BLS12381G1Projective {
        let scalars: Vec<ScalarField> = poly.get_coefficients();

        let points = &self.srs.powers_main_group;

        assert!(
            scalars.len() <= points.len(),
            "Polynomial degree is too large for the current SRS!"
        );

        // 3) Host -> Device 복사
        let host_scalars = HostSlice::from_slice(&scalars);
        let host_points = HostSlice::from_slice(&points[..scalars.len()]);

        // GPU/디바이스 메모리에 결과 1개짜리 버퍼 할당
        let mut msm_result_dev = DeviceVec::<BLS12381G1Projective>::device_malloc(1)
            .expect("device malloc failed");

        // 4) MSM 설정
        let mut stream = IcicleStream::create().unwrap();
        let mut cfg = msm::MSMConfig::default();
        cfg.stream_handle = *stream;
        cfg.is_async = true;

        // 5) MSM 실행: ∑ ( scalars[i] * points[i] )
        msm::msm(
            host_scalars,
            host_points,
            &cfg,
            &mut msm_result_dev[..],
        )
        .expect("MSM failed on device");

        // 6) Device -> Host로 결과 가져오기
        let mut msm_result_host = vec![BLS12381G1Projective::zero(); 1];
        stream.synchronize().unwrap();
        msm_result_dev
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_result_host[..]))
            .unwrap();
        stream.destroy().unwrap();

        // 7) 최종 Commitment (G1 점) 반환
        msm_result_host[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_381::curve::ScalarField;
    use icicle_core::traits::FieldImpl;
    use zkp_rust_tools_math::icicle_bipolynomial::BivariatePolynomial;
    use ndarray::array;

    // 만약 create_srs가 다른 모듈에 있다면 적절히 import
    use crate::bikzg_icicle::srs::StructuredReferenceString;

    #[test]
    fn test_commit_bivariate_simple() {
        // 1) (x_degree=1, y_degree=1) 정도로 작은 SRS 생성 (2x2)
        let srs = StructuredReferenceString::create_srs(1, 1);
        
        // 2) BivariateKateZaveruchaGoldbergIcicle 인스턴스 생성
        let bikzg = BivariateKateZaveruchaGoldbergIcicle::new(srs);

        // 3) 이변다항식 (x+1)(y+1) = xy + x + y + 1 => 계수 4개
        //    하지만 여기서는 ScalarField가 BLS12-381의 Fr이므로,
        //    ScalarField::from(n) 형태로 값을 넣는다.
        let coeffs = array![
            [ScalarField::from_u32(1), ScalarField::from_u32(1)],
            [ScalarField::from_u32(1), ScalarField::from_u32(1)]
        ];
        let coeffs_vec: Vec<Vec<ScalarField>> = coeffs.outer_iter().map(|row| row.to_vec()).collect();
        let poly = BivariatePolynomial::new(coeffs_vec);

        // 4) commit_bivariate 호출
        let commitment = bikzg.commit_bivariate(&poly);

        // 5) 간단한 검증: 커밋 결과가 영점(G1 identity)이 아닌지 확인
        assert_ne!(
            commitment, 
            icicle_bls12_381::curve::G1Projective::zero(),
            "Commitment should not be the identity point"
        );
    }
}