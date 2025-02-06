use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::ntt::{
    ntt, NTTConfig, NTTDir, initialize_domain, get_root_of_unity, NTTInitDomainConfig
};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{Arithmetic, FieldImpl};
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use ndarray::{Array2, Axis};

use super::dense_ext::DensePolynomialExt;
use super::bipolynomial::BivariatePolynomial;

pub type NTTError = &'static str;

/// DeviceBuffer: GPU 메모리 관리를 위한 래퍼
struct DeviceBuffer<T> {
    buffer: DeviceVec<T>,
    size: usize,
}

impl<T> DeviceBuffer<T> {
    fn new(size: usize) -> Result<Self, NTTError> {
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

/// FFT 방향에 따른 NTT 수행
fn perform_ntt(
    input: &HostSlice<ScalarField>,
    direction: NTTDir,
    device_buffer: &mut DeviceBuffer<ScalarField>,
) -> Result<Vec<ScalarField>, NTTError> {
    let mut output = vec![ScalarField::zero(); input.len()];
    let output_slice = HostSlice::from_mut_slice(&mut output);
    
    device_buffer.copy_from_host(input)?;
    ntt(
        input,
        direction,
        &NTTConfig::<ScalarField>::default(),
        &mut device_buffer.buffer[..device_buffer.size],
    ).map_err(|_| "NTT transform failed")?;
    device_buffer.copy_to_host(output_slice)?;
    
    Ok(output)
}

impl BivariatePolynomial {
    pub fn test_multiply_bivariates(a: &Self, b: &Self) -> Result<Self, NTTError> {
        // 최종 결과의 차수 계산
        let result_x_degree = a.x_degree + b.x_degree;
        let result_y_degree = a.y_degree + b.y_degree;
        let max_degree = core::cmp::max(result_x_degree, result_y_degree);
        
        // FFT를 위한 패딩된 크기 계산
        let padded_size = (max_degree + 1).next_power_of_two();

        // Evaluate
        let a_evals = a.evaluate_ntt(1, 1, Some(padded_size), Some(padded_size))?;
        let b_evals = b.evaluate_ntt(1, 1, Some(padded_size), Some(padded_size))?;

        assert_eq!(a_evals.len(), b_evals.len(), "NTT row count mismatch");
        let n = a_evals.len();

        // Pointwise multiplication with scaling
        let mut mul_evals = Vec::with_capacity(n);
        for i in 0..n {
            let mut prod = a_evals[i].mul(&b_evals[i]);
            // Scale by 1/n
            let scale = ScalarField::from_u32(n as u32).inv();
            prod = prod.scale(&scale);
            mul_evals.push(prod);
        }

        // Interpolate
        let mut result = Self::interpolate_ntt(&mul_evals)?;

        // 최종 결과의 크기를 실제 차수에 맞게 조정
        let target_rows = result_y_degree + 1;
        let target_cols = result_x_degree + 1;

        let mut final_coeffs = Vec::with_capacity(target_rows);
        for row in result.coefficients.iter().take(target_rows) {
            let mut new_row = row.get_coefficients();
            new_row.truncate(target_cols);
            final_coeffs.push(new_row);
        }

        Ok(Self::new(final_coeffs))
    }

    pub fn evaluate_ntt(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
    ) -> Result<Vec<DensePolynomial>, NTTError> {
        let dx = domain_x_size.unwrap_or(0);
        let dy = domain_y_size.unwrap_or(0);

        let len_x = core::cmp::max(self.x_degree + 1, dx).next_power_of_two() * x_blowup_factor;
        let len_y = core::cmp::max(self.y_degree + 1, dy).next_power_of_two() * y_blowup_factor;
        let padded_len = len_x.max(len_y);

        // Initialize NTT domain
        initialize_domain(
            get_root_of_unity::<ScalarField>(padded_len as u64),
            &NTTInitDomainConfig::default()
        ).map_err(|_| "Failed to initialize NTT domain")?;

        // Initialize coefficient matrix
        let mut coeffs = Array2::from_elem((padded_len, padded_len), ScalarField::zero());
        
        // Copy input data
        for (i, row) in self.coefficients.iter().enumerate() {
            let row_coeffs = row.get_coefficients();
            for (j, val) in row_coeffs.iter().enumerate() {
                if i < padded_len && j < padded_len {
                    coeffs[[i, j]] = *val;
                }
            }
        }

        let mut device_buffer = DeviceBuffer::new(padded_len)?;

        // Row-wise FFT
        for i in 0..padded_len {
            let row: Vec<_> = coeffs.row(i).to_vec();
            let input_slice = HostSlice::from_slice(&row);
            let output = perform_ntt(&input_slice, NTTDir::kForward, &mut device_buffer)?;
            for (j, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }

        // Column-wise FFT
        for j in 0..padded_len {
            let col: Vec<_> = coeffs.column(j).to_vec();
            let input_slice = HostSlice::from_slice(&col);
            let output = perform_ntt(&input_slice, NTTDir::kForward, &mut device_buffer)?;
            for (i, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }

        // Convert to DensePolynomial
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

    pub fn interpolate_ntt(
        ntt_evals: &Vec<DensePolynomial>
    ) -> Result<Self, NTTError> {
        let len = ntt_evals.len();
        
        let mut coeffs = Array2::from_elem((len, len), ScalarField::zero());
        
        // Copy input data
        for (i, poly) in ntt_evals.iter().enumerate() {
            let row = poly.get_coefficients();
            for (j, val) in row.iter().enumerate() {
                if j < len {
                    coeffs[[i, j]] = *val;
                }
            }
        }

        let mut device_buffer = DeviceBuffer::new(len)?;

        // Row-wise inverse FFT
        for i in 0..len {
            let row: Vec<_> = coeffs.row(i).to_vec();
            let input_slice = HostSlice::from_slice(&row);
            let output = perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer)?;
            for (j, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }

        // Column-wise inverse FFT
        for j in 0..len {
            let col: Vec<_> = coeffs.column(j).to_vec();
            let input_slice = HostSlice::from_slice(&col);
            let output = perform_ntt(&input_slice, NTTDir::kInverse, &mut device_buffer)?;
            for (i, val) in output.iter().enumerate() {
                coeffs[[i, j]] = *val;
            }
        }

        // Convert back to 2D vector
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let row: Vec<_> = coeffs.row(i).to_vec();
            result.push(row);
        }

        Ok(BivariatePolynomial::new(result))
    }
}

// ------------------------------------------------------------------------
// 테스트
// ------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use icicle_core::ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig};
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn initialize_ntt_domain() {
        INIT.call_once(|| {
            let root = get_root_of_unity::<ScalarField>(8);
            initialize_domain(root, &NTTInitDomainConfig::default())
                .expect("Failed to init domain");
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
        initialize_ntt_domain();
        let a_poly = polynomial_a();

        let evals = a_poly.evaluate_ntt(1, 1, None, None)
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
        let result = BivariatePolynomial::test_multiply_bivariates(&poly_f, &poly_g)
            .expect("Failed to multiply polynomials");

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
    fn test_multiply_bivariates() {
        initialize_ntt_domain();
        let a_times_b = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(7), ScalarField::from_u32(2), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(9), ScalarField::from_u32(17), ScalarField::from_u32(9), ScalarField::from_u32(2)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(10), ScalarField::from_u32(19), ScalarField::from_u32(4)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(12), ScalarField::from_u32(16), ScalarField::from_u32(0)]
        ]); 

        let a = polynomial_a();
        let b = polynomial_b();

        let mul_eval = BivariatePolynomial::test_multiply_bivariates(&a, &b)
            .expect("NTT multiply fail");
        // let a_evals =  BivariatePolynomial::evaluate_ntt(&polynomial_a(), 1, 1, Some(4), Some(4)).unwrap();
        // let b_evals = BivariatePolynomial::evaluate_ntt(&polynomial_b(), 1, 1,  Some(4), Some(4)).unwrap();

        let mul_poly = BivariatePolynomial::interpolate_ntt(&mul_eval.coefficients).unwrap();
        for (mul_poly_row, a_times_b_row) in mul_poly.coefficients.iter().zip(a_times_b.coefficients.iter()) {
            assert_eq!(mul_poly_row.get_coefficients(), a_times_b_row.get_coefficients());
        }

        println!("test_multiply_bivariates finished successfully!");
    }
}
