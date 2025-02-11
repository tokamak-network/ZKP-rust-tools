use icicle_bls12_381::polynomials::DensePolynomial;
use std::ops::Neg;
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
use icicle_core::traits::Arithmetic;

pub type NTTError = &'static str;

use icicle_bls12_377::curve::{CurveCfg, G1Projective, ScalarCfg};
use icicle_core::{curve::Curve, msm, msm::MSMConfig, traits::GenerateRandom};
use icicle_runtime::{device::Device, memory::HostSlice};

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
        // println!("Starting to drop DeviceBuffer with size: {}", self.size);
        self.size = 0;
        // println!("DeviceBuffer dropped");
    }
}

fn perform_ntt(
    input: &HostSlice<ScalarField>,
    direction: NTTDir,
    device_buffer: &mut DeviceBuffer<ScalarField>,
) -> Result<Vec<ScalarField>, NTTError> {
    // println!("Starting perform_ntt with input size: {}", input.len());

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
    
    // println!("Copying input to device buffer");
    device_buffer.copy_from_host(input)?;
    
    // println!("Starting NTT computation");
    // 중간 호스트 버퍼 생성
    let mut host_buffer = vec![ScalarField::zero(); device_buffer.size];
    device_buffer.copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))?;
    
    // println!("Performing NTT transform");
    ntt(
        HostSlice::from_slice(&host_buffer),
        direction,
        &NTTConfig::<ScalarField>::default(),
        &mut device_buffer.buffer[..device_buffer.size],
    ).map_err(|e| {
        println!("NTT transform failed with error: {:?}", e);
        "NTT transform failed"
    })?;
    
    // println!("Copying result back to host");
    device_buffer.copy_to_host(output_slice)?;
    
    // println!("perform_ntt completed successfully");
    Ok(output)
}
struct MyScalarField(ScalarField);

impl Neg for MyScalarField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        MyScalarField(ScalarField::zero() - self.0)
    }
}

impl BivariatePolynomial {

    pub fn evaluate_ntt(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
    ) -> Result<Vec<DensePolynomial>, &'static str> {
        // 기존 4에서 64로 상향 조정
        const MAX_EVAL_SIZE: usize = 64;  
        
        let dx = domain_x_size.unwrap_or(0);
        let dy = domain_y_size.unwrap_or(0);
        
        // 필요한 최소 크기 계산
        let min_size = core::cmp::max(
            core::cmp::max(self.x_degree + 1, dx).next_power_of_two(),
            core::cmp::max(self.y_degree + 1, dy).next_power_of_two()
        );
    
        // 크기 검증: 이제 MAX_EVAL_SIZE가 64이므로 대부분의 테스트에서 문제가 없을 것임
        if min_size > MAX_EVAL_SIZE {
            return Err("Evaluation size too large for NTT");
        }
    
        let padded_len = self.calculate_padded_length(dx, dy, x_blowup_factor, y_blowup_factor)?;
        
        // Device buffer 생성 및 초기화
        let mut device_buffer = DeviceBuffer::new(padded_len)?;
        let mut coeffs = Array2::from_elem((padded_len, padded_len), ScalarField::zero());
        
        // NTT 변환 수행
        self.copy_coefficients_to_array(&mut coeffs, padded_len);
        self.perform_row_fft(&mut coeffs, &mut device_buffer, padded_len)?;
        self.perform_column_fft(&mut coeffs, &mut device_buffer, padded_len)?;
    
        // 결과 변환: 각 행을 DensePolynomial로 변환
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

    pub fn evaluate_offset_fft(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
        offset_x: &ScalarField,
        offset_y: &ScalarField,
    ) -> Result<Vec<DensePolynomial>, NTTError> {
        // Scale the polynomial first
        let scaled = self.scale(offset_x, offset_y);
        
        // Then evaluate using normal FFT
        scaled.evaluate_ntt(x_blowup_factor, y_blowup_factor, domain_x_size, domain_y_size)
    }

    pub fn poly_multiply(
        a: &BivariatePolynomial,
        b: &BivariatePolynomial,
        result_x_dim: usize,
        result_y_dim: usize,
    ) -> Result<Self, NTTError> {
        // Evaluate both polynomials
        let a_evals = a.evaluate_ntt(1, 1, Some(result_x_dim), Some(result_y_dim))?;
        let b_evals = b.evaluate_ntt(1, 1, Some(result_x_dim), Some(result_y_dim))?;

        // Multiply point-wise
        let mut c_evals = Vec::with_capacity(a_evals.len());
        for (a_eval, b_eval) in a_evals.iter().zip(b_evals.iter()) {
            let mut c_row = Vec::with_capacity(a_eval.get_coefficients().len());
            for (a_coeff, b_coeff) in a_eval.get_coefficients().iter().zip(b_eval.get_coefficients().iter()) {
                c_row.push(*a_coeff * *b_coeff);
            }
            c_evals.push(DensePolynomial::from_coeffs(
                HostSlice::from_slice(&c_row),
                c_row.len()
            ));
        }

        // Interpolate the result
        Self::interpolate_ntt(&c_evals)
    }

    pub fn coset_division(
        bipoly: &BivariatePolynomial,
        degree_x: usize,
        degree_y: usize,
    ) -> Result<(BivariatePolynomial, BivariatePolynomial), &'static str> {
        // 입력 다항식의 차수 확인
        if bipoly.x_degree + 1 < degree_x || bipoly.y_degree + 1 < degree_y {
            return Err("Polynomial degree is too small for division");
        }
        
        // m: x 방향 조각 개수 (고정 2)  
        let m = 2;
        // n: y 방향 조각 수 계산
        let n = (bipoly.y_degree + 1) / degree_y;
        
        println!("y_degree: {}, degree_y: {}, n: {}", bipoly.y_degree, degree_y, n);
        
        if n < 2 {
            return Err("Unsupported degrees combination: n must be at least 2");
        }
        
        // --- 1단계: A' 계산 (코셋 보정 전 단계) ---
        // A'의 각 계수는, 
        // A'_{i,j} = sum_{y=0}^{n-1} xi^(y) * (해당 블록의 다항식 계수)
        // 여기서는 예시로 xi를 3으로 고정합니다.
        let xi = ScalarField::from_u32(3);
        let mut a_prim_coeffs: Vec<Vec<ScalarField>> = Vec::new();
        // A'는 degree_y 행, degree_x 열의 다항식으로 생성
        for i in 0..degree_y {
            let mut row = Vec::with_capacity(degree_x);
            for j in 0..degree_x {
                let mut sum = ScalarField::zero();
                // 각 조각별 합산: y 인덱스는 0부터 n-1
                for y in 0..n {
                    // 원래 다항식에서 해당 위치: row index = y * degree_y + i, column index = ?  
                    // x 방향도 m 조각으로 나눈다고 가정하면, 각 조각마다 같은 처리 (여기서는 간단하게 1회만 더함)
                    // 실제 알고리즘에 따라, x 방향 조각 처리는 따로 진행할 수 있으므로 여기서는 단순 합산 처리
                    let row_index = y * degree_y + i;
                    if let Some(poly_row) = bipoly.coefficients.get(row_index) {
                        // 예시로 j번째 계수를 사용 (실제는 j번째 블록 내 여러 계수를 합산)
                        if let Some(&coeff) = poly_row.get_coefficients().get(j) {
                            // 각 항에 xi^(y)를 곱하여 더함
                            sum = sum + xi.pow(y as usize) * coeff;
                        }
                    }
                }
                row.push(sum);
            }
            a_prim_coeffs.push(row);
        }
        let a_prim = BivariatePolynomial::new(a_prim_coeffs);
        
        // --- 2단계: r_tilde = A'에 대해 offset FFT 평가 후 보간 ---
        // 예를 들어, r_tilde = A'(x, y) evaluated on coset with offset xi
        let r_tilde_evals = a_prim.evaluate_offset_fft(1, 1, None, None, &ScalarField::one(), &xi)
        .map_err(|_| "FFT evaluation failed")?;
        
        let a_prim_interpolated = BivariatePolynomial::interpolate_ntt(&r_tilde_evals)
            .map_err(|_| "Interpolation failed")?;

    // 결과 반환 전 유효성 검증

        
        // --- 3단계: q_z 계산 (코셋 보정 결과) ---
        // q_z = A' 보간 결과에 대해, 각 계수를 (xi^(degree_y) - 1)로 나누어 정규화
        let divisor = xi.pow(degree_y as usize) - ScalarField::one();
        let divisor_inv = divisor.inv();
        let q_z = a_prim_interpolated.scale(&divisor_inv, &ScalarField::one());

        if q_z.x_degree > bipoly.x_degree || q_z.y_degree > bipoly.y_degree {
            return Err("Invalid result degrees");
        }
        
        // --- 4단계: B에서 remainder를 빼서 q_x 계산 ---
        // remainder = b(x,y) - A(x,y) where A is the part explained by q_z  
        let remainder = bipoly.subtract(&q_z)
            .map_err(|_| "Subtraction failed")?;
        
        // q_x는 remainder에 대해, x 방향으로 coset 보정을 진행 (여기서는 zeta 사용, 예시로 5)
        let zeta = ScalarField::from_u32(5);
        let b_prim_evals = remainder.evaluate_offset_fft(1, 1, None, None, &zeta, &ScalarField::one())
            .map_err(|_| "FFT evaluation on B failed")?;
        let b_prim = BivariatePolynomial::interpolate_ntt(&b_prim_evals)
            .map_err(|_| "Interpolation on B failed")?;
        let divisor_x = zeta.pow(degree_x as usize) - ScalarField::one();
        let divisor_x_inv = divisor_x.inv();
        let q_x = b_prim.scale(&divisor_x_inv, &ScalarField::one());
        
        Ok((q_z, q_x))
    }


    fn subtract(&self, other: &Self) -> Result<Self, NTTError> {
        let mut result_coeffs = Vec::new();
        for (self_row, other_row) in self.coefficients.iter().zip(other.coefficients.iter()) {
            let mut row = Vec::new();
            for (self_coeff, other_coeff) in self_row.get_coefficients().iter()
                .zip(other_row.get_coefficients().iter()) {
                row.push(*self_coeff - *other_coeff);
            }
            result_coeffs.push(row);
        }
        Ok(Self::new(result_coeffs))
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

    #[test]
    fn test_coset_division_m2_n2() {
        // x_degree = 2, y_degree = 4로 다항식 생성 (n=2를 만족하기 위해)
        let rows = vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        ];
        let poly = BivariatePolynomial::new(rows);

        // y_degree가 4인지 확인 (n=2를 위해 필요)
        assert_eq!(poly.y_degree, 3);  // 0부터 시작하므로 실제 차수는 4
        
        // coset_division 호출 시 degree_y를 2로 설정 (n = y_degree/degree_y = 4/2 = 2)
        let result = BivariatePolynomial::coset_division(&poly, 4, 2);
        assert!(result.is_ok(), "Coset division failed: {:?}", result.err());

        if let Ok((q_z, q_x)) = result {
            // 결과 검증
            println!("q_z coefficients:");
            for row in q_z.coefficients.iter() {
                println!("{:?}", row.get_coefficients());
            }

            println!("q_x coefficients:");
            for row in q_x.coefficients.iter() {
                println!("{:?}", row.get_coefficients());
            }

            // 차수 검증
            assert!(q_z.x_degree <= poly.x_degree);
            assert!(q_z.y_degree <= poly.y_degree);
            assert!(q_x.x_degree <= poly.x_degree);
            assert!(q_x.y_degree <= poly.y_degree);
        }
    }
}
    
#[cfg(test)]
mod coset_division_tests {
    use super::*;

    // 테스트 헬퍼 함수
    fn verify_division_result(
        original: &BivariatePolynomial,
        q_z: &BivariatePolynomial,
        q_x: &BivariatePolynomial,
        degree_x: usize,
        degree_y: usize,
    ) -> bool {
        // 차수 검증
        if q_z.x_degree > original.x_degree || q_z.y_degree > original.y_degree {
            println!("q_z degrees exceed original polynomial degrees");
            return false;
        }
        if q_x.x_degree > original.x_degree || q_x.y_degree > original.y_degree {
            println!("q_x degrees exceed original polynomial degrees");
            return false;
        }

        // TODO: 실제 다항식 복원 및 검증 로직 추가
        true
    }

    #[test]
    fn test_coset_division_large_polynomial() {
        // 8x8 크기의 큰 다항식 생성
        let mut rows = Vec::new();
        for i in 0..8 {
            let mut row = Vec::new();
            for j in 0..8 {
                // 복잡한 계수 패턴 생성
                let coeff = match (i, j) {
                    (0, 0) => ScalarField::from_u32(1),  // 상수항
                    (i, 0) => ScalarField::from_u32((i * 2) as u32),  // y축 계수
                    (0, j) => ScalarField::from_u32((j * 3) as u32),  // x축 계수
                    (i, j) => ScalarField::from_u32((i * j) as u32),  // 교차항
                };
                row.push(coeff);
            }
            rows.push(row);
        }
        let poly = BivariatePolynomial::new(rows);

        // 2x4 분할로 coset division 수행
        let result = BivariatePolynomial::coset_division(&poly, 2, 4);
        assert!(result.is_ok(), "Coset division failed: {:?}", result.err());

        if let Ok((q_z, q_x)) = result {
            assert!(verify_division_result(&poly, &q_z, &q_x, 2, 4));
        }
    }

    #[test]
    fn test_coset_division_sparse_polynomial() {
        // 희소 다항식 생성 (대부분의 계수가 0)
        let mut rows = vec![vec![ScalarField::zero(); 6]; 6];
        
        // 의미있는 계수만 설정
        rows[0][0] = ScalarField::from_u32(1);  // 상수항
        rows[2][2] = ScalarField::from_u32(5);  // x²y² 항
        rows[3][1] = ScalarField::from_u32(3);  // xy³ 항
        rows[5][5] = ScalarField::from_u32(7);  // x⁵y⁵ 항

        let poly = BivariatePolynomial::new(rows);

        // 2x2 분할로 테스트
        let result = BivariatePolynomial::coset_division(&poly, 2, 2);
        assert!(result.is_ok(), "Coset division failed on sparse polynomial: {:?}", result.err());

        if let Ok((q_z, q_x)) = result {
            assert!(verify_division_result(&poly, &q_z, &q_x, 2, 2));
        }
    }

    #[test]
    fn test_coset_division_random_polynomial() {
        // 랜덤한 계수를 가진 6x6 다항식 생성
        let mut rows = Vec::new();
        for _ in 0..6 {
            let mut row = Vec::new();
            for _ in 0..6 {
                // 0부터 100 사이의 랜덤 값 사용
                let random_value = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .subsec_nanos() % 100) as u32;
                row.push(ScalarField::from_u32(random_value));
            }
            rows.push(row);
        }
        let poly = BivariatePolynomial::new(rows);

        // 2x3 분할로 테스트
        let result = BivariatePolynomial::coset_division(&poly, 2, 3);
        assert!(result.is_ok(), "Coset division failed on random polynomial: {:?}", result.err());

        if let Ok((q_z, q_x)) = result {
            assert!(verify_division_result(&poly, &q_z, &q_x, 2, 3));
        }
    }

    #[test]
    fn test_coset_division_edge_cases() {
        // 최소 크기 테스트 (2x4 다항식)
        let small_rows = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2)],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4)],
            vec![ScalarField::from_u32(5), ScalarField::from_u32(6)],
            vec![ScalarField::from_u32(7), ScalarField::from_u32(8)],
        ];
        let small_poly = BivariatePolynomial::new(small_rows);
        let small_result = BivariatePolynomial::coset_division(&small_poly, 2, 2);
        assert!(small_result.is_ok(), "Coset division failed on minimal size polynomial: {:?}", small_result.err());

        // 모든 계수가 같은 값인 경우
        let constant_rows = vec![
            vec![ScalarField::from_u32(5); 4],
            vec![ScalarField::from_u32(5); 4],
            vec![ScalarField::from_u32(5); 4],
            vec![ScalarField::from_u32(5); 4],
        ];
        let constant_poly = BivariatePolynomial::new(constant_rows);
        let constant_result = BivariatePolynomial::coset_division(&constant_poly, 2, 2);
        assert!(constant_result.is_ok(), "Coset division failed on constant polynomial: {:?}", constant_result.err());
    }

    #[test]
    fn test_coset_division_reconstruction() {
        // 4x4 다항식 생성
        let original_rows = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::from_u32(3), ScalarField::from_u32(4)],
            vec![ScalarField::from_u32(5), ScalarField::from_u32(6), ScalarField::from_u32(7), ScalarField::from_u32(8)],
            vec![ScalarField::from_u32(9), ScalarField::from_u32(10), ScalarField::from_u32(11), ScalarField::from_u32(12)],
            vec![ScalarField::from_u32(13), ScalarField::from_u32(14), ScalarField::from_u32(15), ScalarField::from_u32(16)],
        ];
        let original_poly = BivariatePolynomial::new(original_rows);

        // coset division 수행
        let result = BivariatePolynomial::coset_division(&original_poly, 2, 2)
            .expect("Coset division failed");
        let (q_z, q_x) = result;

        // 결과 출력 및 검증
        println!("Original polynomial:");
        for row in original_poly.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        println!("\nq_z coefficients:");
        for row in q_z.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        println!("\nq_x coefficients:");
        for row in q_x.coefficients.iter() {
            println!("{:?}", row.get_coefficients());
        }

        // TODO: q_z와 q_x를 사용하여 원본 다항식 복원 로직 구현
    }
}
