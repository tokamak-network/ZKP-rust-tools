use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::ntt::{ntt, NTTConfig, NTTDir, initialize_domain, get_root_of_unity, NTTInitDomainConfig};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{FieldImpl, Arithmetic};
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};

use super::bipolynomial::BivariatePolynomial;
use super::dense_ext::DensePolynomialExt;

pub type NTTError = &'static str;

struct DeviceBuffer<T> {
    buffer: DeviceVec<T>,
    size: usize,
}

impl<T> DeviceBuffer<T> {
    fn new(size: usize) -> Result<Self, NTTError> {
        println!("Creating DeviceBuffer with size: {}", size);
        Ok(Self {
            buffer: DeviceVec::device_malloc(size)
                .map_err(|_| "Device memory allocation failed")?,
            size,
        })
    }

    fn copy_from_host(&mut self, src: &HostSlice<T>) -> Result<(), NTTError> {
        let src_len = src.len();
        let dst_len = self.size;
        
        if src_len != dst_len {
            return Err("Source and destination lengths do not match");
        }
        println!("Copy from host: src_len={}, dst_len={}", src_len, dst_len);
        
        self.buffer[..self.size]
            .copy_from_host(src)
            .map_err(|_| "Failed to copy to device")
    }

    fn copy_to_host(&self, dst: &mut HostSlice<T>) -> Result<(), NTTError> {
        let src_len = self.size;
        let dst_len = dst.len();
        
        if src_len != dst_len {
            return Err("Source and destination lengths do not match");
        }
        println!("Copy to host: src_len={}, dst_len={}", src_len, dst_len);
        
        self.buffer[..self.size]
            .copy_to_host(dst)
            .map_err(|_| "Failed to copy from device")
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // DeviceVec이 자동으로 해제되도록 함
    }
}

fn ntt_forward(
    input: &HostSlice<ScalarField>,
    device_buffer: &mut DeviceBuffer<ScalarField>,
) -> Result<(), NTTError> {
    ntt(
        input,
        NTTDir::kForward,
        &NTTConfig::<ScalarField>::default(),
        &mut device_buffer.buffer[..device_buffer.size],
    ).map_err(|_| "NTT forward transform failed")
}

fn ntt_inverse(
    input: &HostSlice<ScalarField>,
    device_buffer: &mut DeviceBuffer<ScalarField>,
) -> Result<(), NTTError> {
    ntt(
        input,
        NTTDir::kInverse,
        &NTTConfig::<ScalarField>::default(),
        &mut device_buffer.buffer[..device_buffer.size],
    ).map_err(|_| "NTT inverse transform failed")
}

impl BivariatePolynomial {
    pub fn evaluate_ntt(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
    ) -> Result<Vec<DensePolynomial>, NTTError> {
        let domain_x_size = domain_x_size.unwrap_or(0);
        let domain_y_size = domain_y_size.unwrap_or(0);

        println!("Initial parameters:");
        println!("x_blowup_factor: {}", x_blowup_factor);
        println!("y_blowup_factor: {}", y_blowup_factor);
        println!("domain_x_size: {}", domain_x_size);
        println!("domain_y_size: {}", domain_y_size);
        println!("self.x_degree: {}", self.x_degree);
        println!("self.y_degree: {}", self.y_degree);

        let len_x = core::cmp::max(self.x_degree, domain_x_size)
            .next_power_of_two() * x_blowup_factor;
        let len_y = core::cmp::max(self.y_degree, domain_y_size)
            .next_power_of_two() * y_blowup_factor;
        let padded_len = len_x.max(len_y);

        println!("Calculated sizes:");
        println!("len_x: {}", len_x);
        println!("len_y: {}", len_y);
        println!("padded_len: {}", padded_len);

        // NTT 도메인 초기화
        initialize_domain(
            get_root_of_unity::<ScalarField>(padded_len.try_into().unwrap()),
            &NTTInitDomainConfig::default()
        ).map_err(|_| "Failed to initialize NTT domain")?;

        // 초기 계수 준비
        let mut coeffs: Vec<DensePolynomial> = Vec::with_capacity(padded_len);
        for i in 0..padded_len {
            let row = if i < self.coefficients.len() {
                let mut row = self.coefficients[i].get_coefficients().to_vec();
                row.resize(padded_len, ScalarField::zero());
                row
            } else {
                vec![ScalarField::zero(); padded_len]
            };
            coeffs.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&row), row.len()));
            println!("Created polynomial row {}, length: {}", i, row.len());
        }

        let mut device_buffer = DeviceBuffer::new(padded_len)?;

        // 행 방향 NTT
        for i in 0..padded_len {
            println!("Processing row {}", i);
            let row = coeffs[i].get_coefficients().to_vec();
            println!("Row length: {}", row.len());
            
            let host_slice = HostSlice::from_slice(&row);
            device_buffer.copy_from_host(&host_slice)?;

            ntt_forward(&host_slice, &mut device_buffer)?;

            let mut host_result = vec![ScalarField::zero(); padded_len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            coeffs[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }

        // 전치 행렬 계산
        let mut transposed: Vec<DensePolynomial> = Vec::with_capacity(padded_len);
        for i in 0..padded_len {
            let mut col = Vec::with_capacity(padded_len);
            for j in 0..padded_len {
                col.push(coeffs[j].get_coefficients()[i].clone());
            }
            transposed.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&col), col.len()));
        }

        // 열 방향 NTT
        for i in 0..padded_len {
            println!("Processing column {}", i);
            let col = transposed[i].get_coefficients().to_vec();
            println!("Column length: {}", col.len());
            
            let host_slice = HostSlice::from_slice(&col);
            device_buffer.copy_from_host(&host_slice)?;

            ntt_forward(&host_slice, &mut device_buffer)?;

            let mut host_result = vec![ScalarField::zero(); padded_len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            transposed[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }

        // 재전치 및 결과 반환
        let mut result = Vec::with_capacity(padded_len);
        for i in 0..padded_len {
            let mut row = Vec::with_capacity(padded_len);
            for j in 0..padded_len {
                row.push(transposed[j].get_coefficients()[i].clone());
            }
            result.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&row), row.len()));
        }

        Ok(result)
    }

    pub fn interpolate_ntt(
        ntt_evals: &Vec<DensePolynomial>
    ) -> Result<Self, NTTError> {
        let len = ntt_evals.len();
        let mut coeffs = ntt_evals.clone();
        let mut device_buffer = DeviceBuffer::new(len)?;

        // 행 방향 역NTT
        for i in 0..len {
            println!("Interpolating row {}", i);
            let row = coeffs[i].get_coefficients().to_vec();
            println!("Row length: {}", row.len());
            
            let host_slice = HostSlice::from_slice(&row);
            device_buffer.copy_from_host(&host_slice)?;

            ntt_inverse(&host_slice, &mut device_buffer)?;

            let mut host_result = vec![ScalarField::zero(); len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            coeffs[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }

        // 전치
        let mut transposed = Vec::with_capacity(len);
        for i in 0..len {
            let mut col = Vec::with_capacity(len);
            for j in 0..len {
                col.push(coeffs[j].get_coefficients()[i].clone());
            }
            transposed.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&col), col.len()));
        }

        // 열 방향 역NTT
        for i in 0..len {
            println!("Interpolating column {}", i);
            let col = transposed[i].get_coefficients().to_vec();
            println!("Column length: {}", col.len());
            
            let host_slice = HostSlice::from_slice(&col);
            device_buffer.copy_from_host(&host_slice)?;

            ntt_inverse(&host_slice, &mut device_buffer)?;

            let mut host_result = vec![ScalarField::zero(); len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            transposed[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }

        // 재전치
        let mut final_rows = Vec::with_capacity(len);
        for i in 0..len {
            let mut row = Vec::with_capacity(len);
            for j in 0..len {
                row.push(transposed[j].get_coefficients()[i].clone());
            }
            final_rows.push(row);
        }

        Ok(BivariatePolynomial::new(final_rows))
    }


    pub fn evaluate_offset_ntt(
        &self,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
        offset_x: &ScalarField,
        offset_y: &ScalarField,
    ) -> Result<Vec<DensePolynomial>, NTTError> {
        let scaled = self.scale(offset_x, offset_y);
        scaled.evaluate_ntt(x_blowup_factor, y_blowup_factor, domain_x_size, domain_y_size)
    }

    pub fn interpolate_offset_ntt(
        ntt_evals: &Vec<DensePolynomial>,
        offset_x: &ScalarField,
        offset_y: &ScalarField,
    ) -> Result<Self, NTTError> {
        let poly = Self::interpolate_ntt(ntt_evals)?;
        Ok(poly.scale(&offset_x.inv(), &offset_y.inv()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn initialize_ntt_domain() {
        INIT.call_once(|| {
            let root = get_root_of_unity::<ScalarField>(8);
            initialize_domain(root, &NTTInitDomainConfig::default())
                .expect("Failed to initialize NTT domain");
        });
    }

    // Helper function to create test polynomial A
    fn polynomial_a() -> BivariatePolynomial {
        // 3 + x + 2x*y + x^2*y + 4x*y^2
        let rows = vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0)],
        ];
        BivariatePolynomial::new(rows)
    }

    // Helper function to create test polynomial B
    fn polynomial_b() -> BivariatePolynomial {
        // 1 + 2x + 3y + 4xy 
        let rows = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        ];
        BivariatePolynomial::new(rows)
    }

    #[test]
    fn test_evaluation_ntt() {
        initialize_ntt_domain();
        
        let a_poly = polynomial_a();
        let evals = a_poly.evaluate_ntt(1, 1, None, None)
            .expect("Failed to evaluate NTT");
        let interpolated = BivariatePolynomial::interpolate_ntt(&evals)
            .expect("Failed to interpolate NTT");

        let poly_a_zero_pad = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)]
        ]);

        for (poly_a_zero_pad_row, interpolated_row) in poly_a_zero_pad.coefficients.iter().zip(interpolated.coefficients.iter()) {
            assert_eq!(poly_a_zero_pad_row.get_coefficients(), interpolated_row.get_coefficients());
        }
    }

    #[test]
    fn test_multiply_bivariates() {
        initialize_ntt_domain();

        // Expected result after multiplication
        let expected = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(7), ScalarField::from_u32(2), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(9), ScalarField::from_u32(17), ScalarField::from_u32(9), ScalarField::from_u32(2)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(10), ScalarField::from_u32(19), ScalarField::from_u32(4)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(12), ScalarField::from_u32(16), ScalarField::from_u32(0)]
        ]);

        // Evaluate polynomials with explicit domain sizes and proper padding
        let domain_size = 4; // 다항식의 차수를 고려한 적절한 크기
        let a_poly = polynomial_a();
        let b_poly = polynomial_b();

        let a_evals = a_poly.evaluate_ntt(1, 1, Some(domain_size), Some(domain_size))
            .expect("Failed to evaluate polynomial A");
        let b_evals = b_poly.evaluate_ntt(1, 1, Some(domain_size), Some(domain_size))
            .expect("Failed to evaluate polynomial B");

        assert_eq!(a_evals.len(), b_evals.len(), "Evaluation lengths must match");
        
        // Pointwise multiplication
        let mut mul_evals = Vec::with_capacity(a_evals.len());
        for i in 0..a_evals.len() {
            let eval_a = &a_evals[i];
            let eval_b = &b_evals[i];
            assert_eq!(
                eval_a.get_nof_coeffs(),
                eval_b.get_nof_coeffs(),
                "Coefficient counts must match at index {}",
                i
            );
            mul_evals.push(eval_a.mul(eval_b));
        }

        // Interpolate back
        let result = BivariatePolynomial::interpolate_ntt(&mul_evals)
            .expect("Failed to interpolate product");

        // Verify results
        assert_eq!(
            result.coefficients.len(),
            expected.coefficients.len(),
            "Result and expected polynomial heights must match"
        );

        for (result_row, expected_row) in result.coefficients.iter().zip(expected.coefficients.iter()) {
            assert_eq!(
                result_row.get_coefficients(),
                expected_row.get_coefficients(),
                "Polynomial coefficients must match"
            );
        }

        // Clean up
        drop(a_evals);
        drop(b_evals);
        drop(mul_evals);
    }
}