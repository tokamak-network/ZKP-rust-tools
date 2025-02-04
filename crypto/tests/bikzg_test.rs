// tests/cross_implementation_test.rs

use icicle_core::traits::FieldImpl;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use zkp_rust_tools_crypto::{
    bikzg::{
        self,
        StructuredReferenceString as LambdaSRS,
    },
    bikzg_icicle::BivariateKateZaveruchaGoldbergIcicle,
};
use icicle_bls12_381::curve::ScalarField;
use ndarray::array;
// use zkp_rust_tools_math::icicle_bipolynomial::bipolynomial::BivariatePolynomial;

// #[test]
// fn test_cross_implementation_verification() {
//     // 먼저 테스트에 사용할 이변수 다항식을 정의합니다.
//     // f(x,y) = 1 + x + y + xy와 같은 간단한 다항식을 사용합니다.
//     let coeffs_2d = array![
//         [ScalarField::from_u32(1), ScalarField::from_u32(1)],
//         [ScalarField::from_u32(1), ScalarField::from_u32(1)]
//     ];
//     let coeffs_vec: Vec<Vec<ScalarField>> = coeffs_2d
//         .outer_iter()
//         .map(|row| row.to_vec())
//         .collect();
//     let polynomial = BivariatePolynomial::new(coeffs_vec);

//     // 1. Lambdaworks의 SRS 생성
//     // 다항식의 차수에 맞춰 SRS를 생성합니다.
//     println!("Step 1: Generating Lambdaworks SRS...");
//     let lambda_srs = bikzg::srs::create_srs((2, 2));

//     // 2. Icicle BiKZG 인스턴스 생성과 commitment/proof 생성
//     println!("Step 2: Creating Icicle commitment and proof...");
//     let icicle_srs = bikzg::srs::convert_to_icicle_srs(&lambda_srs).unwrap();
//     let icicle_bikzg = BivariateKateZaveruchaGoldbergIcicle::new(icicle_srs);

//     // 평가점 설정
//     let x = ScalarField::from_u32(2);  // x = 2
//     let y = ScalarField::from_u32(3);  // y = 3

//     // 다항식 평가
//     let evaluation = polynomial.evaluate(&x, &y);
//     println!("Polynomial evaluated at x={}, y={}: {:?}", 2, 3, evaluation);

//     // commitment와 proof 생성
//     let p_commitment = icicle_bikzg.commit_bivariate(&polynomial);
//     let proof = icicle_bikzg.open(&x, &y, &evaluation, &polynomial);

//     // 검증을 위해 중간 결과 출력
//     println!("Commitment generated: {:?}", p_commitment);
//     println!("Proof generated: {:?}", proof);

//     // 3. Lambdaworks를 사용한 검증
//     println!("Step 3: Verifying with Lambdaworks...");

//     // Icicle의 타입을 Lambdaworks 타입으로 변환
//     let lw_p_commit = bikzg::utils::icicle_g1_to_lambdaworks(&p_commitment).unwrap();
//     let (lw_proof0, lw_proof1) = bikzg::utils::icicle_proof_to_tuple(&proof).unwrap();

//     // Lambdaworks의 verify 함수를 사용하여 검증
//     let is_valid = bikzg::commit::verify(
//         &lambda_srs,
//         &icicle_bikzg.scalar_to_fr(&x),
//         &icicle_bikzg.scalar_to_fr(&y),
//         &icicle_bikzg.scalar_to_fr(&evaluation),
//         &lw_p_commit,
//         &(lw_proof0, lw_proof1)
//     );

//     // 검증 결과 확인
//     println!("Verification result: {}", is_valid);
//     assert!(is_valid, "Cross-implementation verification failed");
// }

// fn scalar_to_fr(scalar: &ScalarField) -> FrElement {
//     let bytes = scalar.to_bytes_le();
//     let mut limbs = [0u64; 4];
//     for i in 0..4 {
//         let start = i * 8;
//         if start + 8 <= bytes.len() {
//             let mut bytes_chunk = [0u8; 8];
//             bytes_chunk.copy_from_slice(&bytes[start..start + 8]);
//             limbs[i] = u64::from_le_bytes(bytes_chunk);
//         }
//     }
//     FrElement::new(lambdaworks_math::unsigned_integer::element::U256 { limbs })
// }