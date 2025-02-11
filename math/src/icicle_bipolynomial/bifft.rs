use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::ntt::{
    ntt, NTTConfig, NTTDir, initialize_domain, get_root_of_unity, NTTInitDomainConfig
};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::FieldImpl;
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::Device;
use ndarray::Array2;

use super::dense_ext::DensePolynomialExt;
use super::bipolynomial::BivariatePolynomial;

pub type NTTError = &'static str;

// #[derive(Debug)]
// enum NTTErrorDetail {
//     SizeExceeded {
//         size: usize,
//         max_allowed: usize,
//     },
//     MemoryAllocation {
//         required_size: usize,
//     },
//     InvalidInput {
//         details: String,
//     },
//     TransformFailed {
//         details: String,
//     },
//     DeviceError {
//         details: String,
//     },
// }

// impl NTTErrorDetail {
//     fn to_str(&self) -> &'static str {
//         match self {
//             Self::SizeExceeded { .. } => "Size exceeded maximum allowed",
//             Self::MemoryAllocation { .. } => "Memory allocation failed",
//             Self::InvalidInput { .. } => "Invalid input",
//             Self::TransformFailed { .. } => "Transform failed",
//             Self::DeviceError { .. } => "Device error",
//         }
//     }
// }

struct DeviceBuffer<T> {
    buffer: DeviceVec<T>,
    size: usize,
    max_size: usize,
}

impl DeviceBuffer<ScalarField> {
    fn new(size: usize) -> Result<Self, &'static str> {
        const MAX_BUFFER_SIZE: usize = 64;
        
        if size == 0 {
            return Err("Cannot create buffer with zero size");
        }
        
        if size > MAX_BUFFER_SIZE {
            return Err("Size exceeds maximum allowed");
        }

        match DeviceVec::device_malloc(size) {
            Ok(buffer) => Ok(Self {
                buffer,
                size,
                max_size: MAX_BUFFER_SIZE,  // max_size 필드 추가
            }),
            Err(_) => Err("Failed to allocate device memory"),
        }
    }

    fn copy_from_host(&mut self, src: &HostSlice<ScalarField>) -> Result<(), &'static str> {
        if src.len() != self.size {
            return Err("Source and destination lengths do not match");
        }
        
        self.buffer[..self.size]
            .copy_from_host(src)
            .map_err(|_| "Failed to copy to device")
    }

    fn copy_to_host(&self, dst: &mut HostSlice<ScalarField>) -> Result<(), &'static str> {
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
    ) -> Result<Vec<DensePolynomial>, &'static str> {
        const MAX_EVAL_SIZE: usize = 4;  // size 제한 유지
        
        let dx = domain_x_size.unwrap_or(0);
        let dy = domain_y_size.unwrap_or(0);
        
        // 필요한 최소 크기 계산
        let min_size = core::cmp::max(
            core::cmp::max(self.x_degree + 1, dx).next_power_of_two(),
            core::cmp::max(self.y_degree + 1, dy).next_power_of_two()
        );
    
        // 크기 검증
        if min_size > MAX_EVAL_SIZE {
            return Err("Evaluation size too large");
        }
    
        let padded_len = self.calculate_padded_length(dx, dy, x_blowup_factor, y_blowup_factor)?;
        
        // Device buffer 생성 및 초기화
        let mut device_buffer = DeviceBuffer::new(padded_len)?;
        let mut coeffs = Array2::from_elem((padded_len, padded_len), ScalarField::zero());
        
        // NTT 변환 수행
        self.copy_coefficients_to_array(&mut coeffs, padded_len);
        self.perform_row_fft(&mut coeffs, &mut device_buffer, padded_len)?;
        self.perform_column_fft(&mut coeffs, &mut device_buffer, padded_len)?;
    
        // 결과 변환
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
        const MAX_INTERP_SIZE: usize = 4;  // size 제한 유지
    
        if ntt_evals.is_empty() {
            return Err("Empty evaluation vector");
        }

        let len = ntt_evals.len();
        if len > MAX_INTERP_SIZE {
            return Err("Interpolation size too large");
        }

        // Device buffer 생성
        let mut device_buffer = DeviceBuffer::new(len)?;
        let mut coeffs = Array2::from_elem((len, len), ScalarField::zero());

        // Row-wise inverse FFT
        for i in 0..len {
            let row = ntt_evals[i].get_coefficients();
            if row.len() != len {
                return Err("Inconsistent evaluation lengths");
            }

            let input_slice = HostSlice::from_slice(&row);
            match perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer) {
                Ok(output) => {
                    for (j, val) in output.iter().enumerate() {
                        coeffs[[i, j]] = *val;
                    }
                },
                Err(_) => return Err("Row-wise inverse FFT failed"),
            }
        }
    
        // Column-wise inverse FFT
        for j in 0..len {
            let col: Vec<_> = coeffs.column(j).to_vec();
            let input_slice = HostSlice::from_slice(&col);
            match perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer) {
                Ok(output) => {
                    for (i, val) in output.iter().enumerate() {
                        coeffs[[i, j]] = *val;
                    }
                },
                Err(_) => return Err("Column-wise inverse FFT failed"),
            }
        }
    
        // 결과 생성
        let mut result = Vec::new();
        for i in 0..len {
            let row = coeffs.row(i).to_vec();
            if row.iter().any(|x| *x != ScalarField::zero()) {
                result.push(row);
            }
        }
    
        if result.is_empty() {
            result.push(vec![ScalarField::zero()]);
        }
    
        Ok(Self::new(result))
    }
}


#[cfg(test)]
mod tests {
    

    use super::*;
    use icicle_core::ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig};
    use icicle_runtime::Device;
    
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
    

    
}
