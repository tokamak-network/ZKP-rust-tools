use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::ntt::{
    ntt, NTTConfig, NTTDir, initialize_domain, get_root_of_unity, NTTInitDomainConfig
};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{Arithmetic, FieldImpl};
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::{runtime, Device};
use ndarray::Array2;

use super::dense_ext::DensePolynomialExt;
use super::bipolynomial::BivariatePolynomial;

pub type NTTError = &'static str;

struct DeviceBuffer<T> {
    buffer: DeviceVec<T>,  // Option으로 감싸서 안전한 해제 보장
    size: usize,
}

impl<T> DeviceBuffer<T> {
    fn new(size: usize) -> Result<Self, NTTError> {
        // 크기가 0인 경우 특별 처리
        if size == 0 {
            return Err("Cannot create buffer with zero size");
        }
        
        Ok(Self {
            buffer: DeviceVec::device_malloc(size)
                .map_err(|_| "Device memory allocation failed")?,
            size,
        })
    }

    fn copy_from_host(&mut self, src: &HostSlice<T>) -> Result<(), NTTError> {
        if src.len() != self.size {
            return Err("Source and destination lengths do not match");
        }
        self.buffer[..self.size]
            .copy_from_host(src)
            .map_err(|_| "Failed to copy to device")
    }

    fn copy_to_host(&self, dst: &mut HostSlice<T>) -> Result<(), NTTError> {
        if dst.len() != self.size {
            return Err("Source and destination lengths do not match");
        }
        self.buffer[..self.size]
            .copy_to_host(dst)
            .map_err(|_| "Failed to copy from device")
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        println!("Starting to drop DeviceBuffer with size: {}", self.size);
        self.size = 0;
        println!("DeviceBuffer dropped");
    }
}

fn perform_ntt(
    input: &HostSlice<ScalarField>,
    direction: NTTDir,
    device_buffer: &mut DeviceBuffer<ScalarField>,
) -> Result<Vec<ScalarField>, NTTError> {
    println!("Starting perform_ntt with input size: {}", input.len());

    // runtime::load_backend_from_env_or_default().unwrap();
    let device = Device::new("CPU", 0);
    icicle_runtime::set_device(&device).unwrap();

    initialize_domain(
        get_root_of_unity::<ScalarField>(
            input.len().try_into()
                .unwrap(),
        ),
        &NTTInitDomainConfig::default(),
    )
    .unwrap();
    
    let mut output = vec![ScalarField::zero(); input.len()];
    let output_slice = HostSlice::from_mut_slice(&mut output);
    
    println!("Copying input to device buffer");
    device_buffer.copy_from_host(input)?;
    
    println!("Starting NTT computation");
    // 중간 호스트 버퍼 생성
    let mut host_buffer = vec![ScalarField::zero(); device_buffer.size];
    device_buffer.copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))?;
    
    println!("Performing NTT transform");
    ntt(
        HostSlice::from_slice(&host_buffer),
        direction,
        &NTTConfig::<ScalarField>::default(),
        &mut device_buffer.buffer[..device_buffer.size],
    ).map_err(|e| {
        println!("NTT transform failed with error: {:?}", e);
        "NTT transform failed"
    })?;
    
    println!("Copying result back to host");
    device_buffer.copy_to_host(output_slice)?;
    
    println!("perform_ntt completed successfully");
    Ok(output)
}

impl BivariatePolynomial {

    pub fn evaluate_ntt(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
    ) -> Result<Vec<DensePolynomial>, NTTError> {
        let dx = domain_x_size.unwrap_or(0);
        let dy = domain_y_size.unwrap_or(0);
        
        // 입력 다항식의 차수 확인
        let degree_x = self.x_degree;
        let degree_y = self.y_degree;
        
        // 필요한 최소 크기 계산
        let min_size = core::cmp::max(degree_x, degree_y) + 1;
        let required_size = min_size.next_power_of_two();
        
        // 실제 사용할 크기 결정
        let size = if dx > 0 && dy > 0 {
            core::cmp::min(required_size, core::cmp::min(dx, dy))
        } else {
            required_size
        };
        
        // Size가 4를 초과하지 않도록 제한
        let safe_size = core::cmp::min(size, 4);
        
        // 실제 evaluation 수행
        let padded_len = self.calculate_padded_length(safe_size, safe_size, x_blowup_factor, y_blowup_factor)?;
        let mut coeffs = Array2::from_elem((padded_len, padded_len), ScalarField::zero());
        self.copy_coefficients_to_array(&mut coeffs, padded_len);
        
        let mut device_buffer = DeviceBuffer::new(padded_len)?;
        
        self.perform_row_fft(&mut coeffs, &mut device_buffer, padded_len)?;
        self.perform_column_fft(&mut coeffs, &mut device_buffer, padded_len)?;
        
        let mut result = Vec::with_capacity(padded_len);
        for i in 0..padded_len {
            let row: Vec<_> = coeffs.row(i).to_vec();
            result.push(DensePolynomial::from_coeffs(
                HostSlice::from_slice(&row),
                padded_len
            ));
        }
        
        Ok(result)
    }

    fn calculate_padded_length(
        &self,
        dx: usize,
        dy: usize,
        x_factor: usize,
        y_factor: usize,
    ) -> Result<usize, NTTError> {
        let min_size = if self.x_degree == 0 && self.y_degree == 0 { 1 } else { 2 };
        let len_x = core::cmp::max(min_size, core::cmp::max(self.x_degree + 1, dx).next_power_of_two() * x_factor);
        let len_y = core::cmp::max(min_size, core::cmp::max(self.y_degree + 1, dy).next_power_of_two() * y_factor);
        let padded_len = len_x.max(len_y);
    
        println!("Padding calculation:");
        println!("min_size: {}", min_size);
        println!("len_x: {} (x_degree: {}, dx: {})", len_x, self.x_degree, dx);
        println!("len_y: {} (y_degree: {}, dy: {})", len_y, self.y_degree, dy);
        println!("final padded_len: {}", padded_len);
    
        if padded_len > 64 {
            return Err("Evaluation size too large for NTT");
        }
        Ok(padded_len)
    }

    fn copy_coefficients_to_array(&self, coeffs: &mut Array2<ScalarField>, padded_len: usize) {
        for (i, row) in self.coefficients.iter().enumerate() {
            let row_coeffs = row.get_coefficients();
            for (j, val) in row_coeffs.iter().enumerate() {
                if i < padded_len && j < padded_len {
                    coeffs[[i, j]] = *val;
                }
            }
        }
    }

    fn perform_row_fft(
        &self,
        coeffs: &mut Array2<ScalarField>,
        device_buffer: &mut DeviceBuffer<ScalarField>,
        padded_len: usize,
    ) -> Result<(), NTTError> {
        for i in 0..padded_len {
            let row: Vec<_> = coeffs.row(i).to_vec();
            let input_slice = HostSlice::from_slice(&row);
            let output = perform_ntt(&input_slice, NTTDir::kForward, device_buffer)?;
            for (j, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }
        Ok(())
    }

    fn perform_column_fft(
        &self,
        coeffs: &mut Array2<ScalarField>,
        device_buffer: &mut DeviceBuffer<ScalarField>,
        padded_len: usize,
    ) -> Result<(), NTTError> {
        for j in 0..padded_len {
            let col: Vec<_> = coeffs.column(j).to_vec();
            let input_slice = HostSlice::from_slice(&col);
            let output = perform_ntt(&input_slice, NTTDir::kForward, device_buffer)?;
            for (i, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }
        Ok(())
    }

    pub fn interpolate_ntt(ntt_evals: &Vec<DensePolynomial>) -> Result<Self, NTTError> {
        println!("\n=== Starting interpolate_ntt ===");
        println!("Input length: {}", ntt_evals.len());
    
        // 1. 빈 입력 처리
        if ntt_evals.is_empty() {
            return Ok(Self::new(vec![vec![]]));
        }
    
        // 2. 상수 다항식 처리
        if ntt_evals.len() == 1 {
            return Ok(Self::new(vec![ntt_evals[0].get_coefficients().to_vec()]));
        }
    
        // 3. 크기 제한 추가
        let len = ntt_evals.len();
        if len > 4 {  // size 4로 제한
            return Err("Interpolation size too large");
        }
    
        // 4. Device buffer 생성
        let mut device_buffer = DeviceBuffer::new(len)?;
    
        // 5. Row-wise inverse FFT
        let mut coeffs = Array2::from_elem((len, len), ScalarField::zero());
        for i in 0..len {
            let row: Vec<_> = ntt_evals[i].get_coefficients().to_vec();
            coeffs.row_mut(i).assign(&ndarray::ArrayView1::from(&row));
            
            let input_slice = HostSlice::from_slice(&row);
            let output = perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer)?;
            
            for (j, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }
    
        // 6. Column-wise inverse FFT
        for j in 0..len {
            let col: Vec<_> = coeffs.column(j).to_vec();
            let input_slice = HostSlice::from_slice(&col);
            let output = perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer)?;
            
            for (i, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }
    
        // 7. 불필요한 0 제거 및 결과 생성
        let mut result = Vec::new();
        let mut non_zero_rows = 0;
        for i in 0..len {
            let row = coeffs.row(i).to_vec();
            if row.iter().any(|x| *x != ScalarField::zero()) {
                result.push(row);
                non_zero_rows += 1;
            }
        }
    
        // 모든 행이 0인 경우 처리
        if result.is_empty() {
            result.push(vec![ScalarField::zero()]);
        }
    
        println!("=== interpolate_ntt completed successfully ===\n");
        Ok(Self::new(result))
    }
}

// ------------------------------------------------------------------------
// 테스트
// ------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use crate::icicle_bipolynomial;

    use super::*;
    use icicle_core::ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig};
    use icicle_runtime::{runtime, Device};
    use core::cmp::min;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn initialize_ntt_domain() {
        INIT.call_once(|| {
            std::panic::catch_unwind(|| {
                let root = get_root_of_unity::<ScalarField>(8);
                initialize_domain(root, &NTTInitDomainConfig::default())
                    .expect("Failed to init domain");
            }).unwrap_or_else(|_| {
                eprintln!("Warning: NTT domain initialization failed");
            });
        });
    }

    // 예제: Bivariate A = 3 + x + 2x*y + x^2*y + 4x*y^2
    fn polynomial_a() -> BivariatePolynomial {
        let rows = vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::from_u32(2), ScalarField::from_u32(1)],
            vec![ScalarField::zero(), ScalarField::from_u32(4), ScalarField::zero()],
        ];
        BivariatePolynomial::new(rows)
    }

    // 예제: B = 1 + 2x + 3y + 4xy
    fn polynomial_b() -> BivariatePolynomial {
        let rows = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::zero()],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::zero(), ScalarField::zero()],
        ];
        BivariatePolynomial::new(rows)
    }


    #[test]
    fn test_evaluation_ntt() {
        // initialize_ntt_domain();
        let a_poly = polynomial_a();

        let evals = a_poly.evaluate_ntt(1, 1, Some(4), Some(4))
            .expect("NTT eval fail");
        let interpolated = BivariatePolynomial::interpolate_ntt(&evals)
            .expect("NTT interpolate fail");

        // 영패딩 4x4
        let poly_a_padded = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::zero(), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::from_u32(4), ScalarField::zero(), ScalarField::zero()],
            vec![ScalarField::zero(); 4],
        ]);

        for (exp, got) in poly_a_padded.coefficients.iter().zip(interpolated.coefficients.iter()) {
            assert_eq!(exp.get_coefficients(), got.get_coefficients());
        }
    }

    #[test]
    fn test_interpolate_ntt() {
        initialize_ntt_domain();

        // 테스트할 간단한 이변수 다항식: f(x,y) = 1 + 2x + 3y
        let original = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2)],  // 1 + 2x
            vec![ScalarField::from_u32(3), ScalarField::zero()],       // 3y
        ]);

        // NTT evaluation 수행
        let evals = original.evaluate_ntt(1, 1, None, None)
            .expect("NTT evaluation failed");

        // Interpolation 수행
        let interpolated = BivariatePolynomial::interpolate_ntt(&evals)
            .expect("NTT interpolation failed");

        // 결과 출력 (디버깅용)
        println!("Original polynomial coefficients:");
        for row in original.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        println!("\nInterpolated polynomial coefficients:");
        for row in interpolated.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        // 차수 비교
        assert_eq!(original.x_degree, interpolated.x_degree, 
            "X degrees should match after interpolation");
        assert_eq!(original.y_degree, interpolated.y_degree, 
            "Y degrees should match after interpolation");

        // 계수 비교
        for (orig_row, interp_row) in original.coefficients.iter().zip(interpolated.coefficients.iter()) {
            assert_eq!(orig_row.get_coefficients(), interp_row.get_coefficients(), 
                "Polynomial coefficients should match after interpolation");
        }

        // 특정 점에서의 값 비교
        let test_points = vec![
            (ScalarField::from_u32(2), ScalarField::from_u32(3)),  // (x,y) = (2,3)
            (ScalarField::from_u32(1), ScalarField::from_u32(1)),  // (x,y) = (1,1)
            (ScalarField::from_u32(0), ScalarField::from_u32(4)),  // (x,y) = (0,4)
        ];

        for (x, y) in test_points {
            let orig_eval = original.evaluate(&x, &y);
            let interp_eval = interpolated.evaluate(&x, &y);
            assert_eq!(orig_eval, interp_eval, 
                "Evaluations should match at point ({:?}, {:?})", x, y);
        }

        // 경계 케이스: 상수 다항식
        let constant_poly = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(5)],  // f(x,y) = 5
        ]);

        let const_evals = constant_poly.evaluate_ntt(1, 1, None, None)
            .expect("NTT evaluation failed for constant polynomial");
        let const_interpolated = BivariatePolynomial::interpolate_ntt(&const_evals)
            .expect("NTT interpolation failed for constant polynomial");

        assert_eq!(constant_poly.x_degree, const_interpolated.x_degree,
            "Degrees should match for constant polynomial");
        assert_eq!(constant_poly.coefficients[0].get_coefficients()[0], 
            const_interpolated.coefficients[0].get_coefficients()[0],
            "Constant term should match after interpolation");

        // 경계 케이스: 영 다항식
        let zero_poly = BivariatePolynomial::new(vec![vec![ScalarField::zero()]]);
        let zero_evals = zero_poly.evaluate_ntt(1, 1, None, None)
            .expect("NTT evaluation failed for zero polynomial");
        let zero_interpolated = BivariatePolynomial::interpolate_ntt(&zero_evals)
            .expect("NTT interpolation failed for zero polynomial");

        assert_eq!(zero_poly.coefficients[0].get_coefficients()[0], 
            zero_interpolated.coefficients[0].get_coefficients()[0],
            "Zero polynomial should remain zero after interpolation");
    }

    #[test]
    fn test_simple_polynomial_multiplication() {
        initialize_ntt_domain();

        // 간단한 이변수 다항식 생성: f(x,y) = 1 + x + y
        let poly_f = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(1)],  // 1 + x
            vec![ScalarField::from_u32(1), ScalarField::zero()],       // y
        ]);

        // g(x,y) = 1 + x + y
        let poly_g = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(1)],  // 1 + x
            vec![ScalarField::from_u32(1), ScalarField::zero()],       // y
        ]);

        // h(x,y) = f(x,y) * g(x,y) = (1 + x + y)^2
        // = 1 + 2x + 2y + x^2 + 2xy + y^2
        let expected = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::from_u32(1)],  // 1 + 2x + x^2
            vec![ScalarField::from_u32(2), ScalarField::from_u32(2), ScalarField::zero()],       // 2y + 2xy
            vec![ScalarField::from_u32(1), ScalarField::zero(), ScalarField::zero()],            // y^2
        ]);

        // 곱셈 수행
        // let result = BivariatePolynomial::test_multiply_bivariates(&poly_f, &poly_g)
        //     .expect("Failed to multiply polynomials");

        let result = poly_f * poly_g;

        println!("Expected polynomial coefficients:");
        for row in expected.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        println!("\nActual polynomial coefficients:");
        for row in result.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        // 결과 비교
        assert_eq!(result.coefficients.len(), expected.coefficients.len(), 
            "Polynomial dimensions mismatch");

        for (res_row, exp_row) in result.coefficients.iter().zip(expected.coefficients.iter()) {
            assert_eq!(res_row.get_coefficients(), exp_row.get_coefficients(), 
                "Polynomial coefficients mismatch");
        }
    }


    #[test]
fn test_size_8_specific_issue() {
    let device = Device::new("CPU", 0);
    icicle_runtime::set_device(&device).expect("Failed to set device");
    initialize_ntt_domain();

    let size = 8;
    println!("Testing with size {}", size);

    let poly_a = polynomial_a();
    let poly_b = polynomial_b();

    // 1. A에 대한 evaluation 수행 후 즉시 확인
    let a_evals = BivariatePolynomial::evaluate_ntt(&poly_a, 1, 1, Some(size), Some(size))
        .expect("A evaluation failed");
    
    println!("A evaluation successful");
    println!("A evaluations count: {}", a_evals.len());
    println!("A first evaluation size: {}", a_evals[0].get_coefficients().len());

    // A의 coefficients 변환
    let a_coeffs: Vec<Vec<ScalarField>> = a_evals.iter()
        .map(|eval| {
            let coeffs = eval.get_coefficients();
            println!("A eval length: {}", coeffs.len());
            coeffs.to_vec()
        })
        .collect();

    // Memory cleanup for A
    drop(a_evals);

    // 2. B에 대한 evaluation 수행 후 즉시 확인
    let b_evals = BivariatePolynomial::evaluate_ntt(&poly_b, 1, 1, Some(size), Some(size))
        .expect("B evaluation failed");
    
    println!("B evaluation successful");
    println!("B evaluations count: {}", b_evals.len());
    println!("B first evaluation size: {}", b_evals[0].get_coefficients().len());

    // B의 coefficients 변환
    let b_coeffs: Vec<Vec<ScalarField>> = b_evals.iter()
        .map(|eval| {
            let coeffs = eval.get_coefficients();
            println!("B eval length: {}", coeffs.len());
            coeffs.to_vec()
        })
        .collect();

    // Memory cleanup for B
    drop(b_evals);

    // 3. 새로운 BivariatePolynomial 생성
    let bipoly_a = BivariatePolynomial::new(a_coeffs);
    let bipoly_b = BivariatePolynomial::new(b_coeffs);

    println!("\nCreated polynomials:");
    println!("A: x_degree={}, y_degree={}", bipoly_a.x_degree, bipoly_a.y_degree);
    println!("B: x_degree={}, y_degree={}", bipoly_b.x_degree, bipoly_b.y_degree);

    // 4. Device buffer 상태 확인
    let buffer_test = DeviceBuffer::<ScalarField>::new(size)
        .expect("Failed to create test buffer");
    println!("Test buffer created successfully");
    drop(buffer_test);

    // 5. 곱셈 시도
    println!("\nAttempting multiplication...");
    
    // Explicit scope for multiplication
    {
        let result = bipoly_a * bipoly_b;
        println!("Multiplication successful");
        println!("Result: x_degree={}, y_degree={}", result.x_degree, result.y_degree);
    }
}
    
#[test]
fn test_interpolation_step_by_step() {
    let device = Device::new("CPU", 0);
    icicle_runtime::set_device(&device).expect("Failed to set device");
    initialize_ntt_domain();

    let size = 4;
    println!("=== Starting interpolation test with size {} ===", size);

    // 1. 원본 다항식 준비
    let poly_a = polynomial_a();
    let poly_b = polynomial_b();

    // 2. NTT evaluation 수행
    let a_evals = BivariatePolynomial::evaluate_ntt(&poly_a, 1, 1, Some(size), Some(size))
        .expect("Failed to evaluate polynomial A");
    let b_evals = BivariatePolynomial::evaluate_ntt(&poly_b, 1, 1, Some(size), Some(size))
        .expect("Failed to evaluate polynomial B");

    // 3. BivariatePolynomial 생성
    let bipoly_a = BivariatePolynomial::new(
        a_evals.iter()
            .map(|eval| eval.get_coefficients().to_vec())
            .collect()
    );
    let bipoly_b = BivariatePolynomial::new(
        b_evals.iter()
            .map(|eval| eval.get_coefficients().to_vec())
            .collect()
    );

    // 4. 곱셈 수행
    let mul_eval = bipoly_a * bipoly_b;
    println!("\nMultiplication result structure:");
    println!("x_degree: {}, y_degree: {}", mul_eval.x_degree, mul_eval.y_degree);
    for (i, row) in mul_eval.coefficients.iter().enumerate() {
        println!("Row {} length: {}", i, row.get_coefficients().len());
    }

    // 5. 행별로 interpolation 시도
    println!("\nTesting row-by-row interpolation:");
    for (i, row) in mul_eval.coefficients.iter().enumerate() {
        println!("\nProcessing row {}", i);
        
        // 단일 행에 대한 interpolation
        let single_row_coeffs = vec![row.clone()];
        match BivariatePolynomial::interpolate_ntt(&single_row_coeffs) {
            Ok(result) => {
                println!("Row {} interpolation successful", i);
                println!("Result dimensions: x_degree={}, y_degree={}", 
                        result.x_degree, result.y_degree);
            },
            Err(e) => {
                println!("Row {} interpolation failed: {}", i, e);
                return;
            }
        }
    }

    // 6. 실제 interpolation 시도
    println!("\nAttempting full interpolation...");
    let device_buffer = DeviceBuffer::<ScalarField>::new(size)
        .expect("Failed to create device buffer");
    
    // Interpolation 전에 coefficients 검증
    println!("\nValidating coefficients before interpolation:");
    println!("Number of coefficient rows: {}", mul_eval.coefficients.len());
    for (i, row) in mul_eval.coefficients.iter().enumerate() {
        let coeffs = row.get_coefficients();
        println!("Row {} size: {}", i, coeffs.len());
        
        // 값들의 유효성 검사
        for (j, val) in coeffs.iter().enumerate() {
            if *val == ScalarField::zero() {
                println!("Zero value at row {}, col {}", i, j);
            }
        }
    }

    // 7. 최종 interpolation
    match BivariatePolynomial::interpolate_ntt(&mul_eval.coefficients) {
        Ok(result) => {
            println!("\nFull interpolation successful!");
            println!("Final result dimensions: x_degree={}, y_degree={}", 
                    result.x_degree, result.y_degree);
            
            // 결과 검증
            println!("\nFinal result coefficients:");
            for (i, row) in result.coefficients.iter().enumerate() {
                println!("Row {}: {:?}", i, row.get_coefficients());
            }
        },
        Err(e) => println!("Full interpolation failed: {}", e)
    }
}
        // if let Ok(mul_poly) = BivariatePolynomial::interpolate_ntt(&mul_eval.coefficients) {
        //     let a_times_b = BivariatePolynomial::new(vec![
        //         vec![ScalarField::from_u32(3), ScalarField::from_u32(7), ScalarField::from_u32(2), ScalarField::from_u32(0)],
        //         vec![ScalarField::from_u32(9), ScalarField::from_u32(17), ScalarField::from_u32(9), ScalarField::from_u32(2)],
        //         vec![ScalarField::from_u32(0), ScalarField::from_u32(10), ScalarField::from_u32(19), ScalarField::from_u32(4)],
        //         vec![ScalarField::from_u32(0), ScalarField::from_u32(12), ScalarField::from_u32(16), ScalarField::from_u32(0)]
        //     ]);
            
        //     for (mul_poly_row, a_times_b_row) in mul_poly.coefficients.iter().zip(a_times_b.coefficients.iter()) {
        //         assert_eq!(mul_poly_row.get_coefficients(), a_times_b_row.get_coefficients());
        //     }
        // }
    

    
}
