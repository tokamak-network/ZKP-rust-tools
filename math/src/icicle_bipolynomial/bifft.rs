// src/icicle_bipolynomial/bifft.rs

use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::ntt::{ntt, NTTConfig, NTTDir, initialize_domain, get_root_of_unity, NTTInitDomainConfig};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{FieldImpl, Arithmetic};
use std::ops::Mul;
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use crate::icicle_bipolynomial;

use super::dense_ext::DensePolynomialExt;


use super::bipolynomial::BivariatePolynomial;

pub type NTTError = &'static str;

// ------------------------------------------------------------------------
// DeviceBuffer -- simple wrapper to manage DeviceVec allocation / copying
// ------------------------------------------------------------------------
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
        // DeviceVec is dropped automatically here.
        // Additional device cleanup code would go here if needed.
    }
}

// ------------------------------------------------------------------------
// Simple forward/inverse NTT wrappers
// ------------------------------------------------------------------------
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

// ------------------------------------------------------------------------
// BivariatePolynomial NTT-based evaluation & interpolation
// ------------------------------------------------------------------------
impl BivariatePolynomial {
    /// Evaluate this BivariatePolynomial using 2D-NTT:
    ///  1) Pad each row to `padded_len`
    ///  2) NTT rows
    ///  3) Transpose
    ///  4) NTT columns
    ///  5) Transpose back
    /// Returns a Vec<DensePolynomial> representing the 2D-NTT result
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

        // 1) Determine padded_len for 2D NTT
        let len_x = core::cmp::max(self.x_degree, domain_x_size).next_power_of_two() * x_blowup_factor;
        let len_y = core::cmp::max(self.y_degree, domain_y_size).next_power_of_two() * y_blowup_factor;
        let padded_len = len_x.max(len_y);

        println!("Calculated sizes:");
        println!("len_x: {}", len_x);
        println!("len_y: {}", len_y);
        println!("padded_len: {}", padded_len);

        // 2) Initialize NTT domain
        initialize_domain(
            get_root_of_unity::<ScalarField>(padded_len as u64),
            &NTTInitDomainConfig::default()
        ).map_err(|_| "Failed to initialize NTT domain")?;

        // 3) Pad rows
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

        // 4) NTT row-wise
        let mut device_buffer = DeviceBuffer::new(padded_len)?;
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

        // 5) Transpose
        let mut transposed: Vec<DensePolynomial> = Vec::with_capacity(padded_len);
        for i in 0..padded_len {
            let mut col = Vec::with_capacity(padded_len);
            for j in 0..padded_len {
                col.push(coeffs[j].get_coefficients()[i].clone());
            }
            transposed.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&col), col.len()));
        }

        // 6) NTT col-wise
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

        // 7) Transpose back
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
        let len = ntt_evals[0].get_nof_coeffs() as usize;
        let eval_len = ntt_evals.len();
        println!("Interpolation length: {}", len);
        
        let mut coeffs = ntt_evals.clone();
        let mut device_buffer = DeviceBuffer::new(len)?;
    
        // Row-wise 역변환
        for i in 0..eval_len {
            let coeffs_i = coeffs[i].get_coefficients();
            let host_slice = HostSlice::from_slice(&coeffs_i);
            device_buffer.copy_from_host(&host_slice)?;
            ntt_inverse(&host_slice, &mut device_buffer)?;
            
            let mut host_result = vec![ScalarField::zero(); len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            coeffs[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }
    
        // Transpose
        let mut transposed = Vec::with_capacity(len);
        for i in 0..len {
            let col: Vec<_> = (0..eval_len)
                .map(|j| coeffs[j].get_coefficients()[i].clone())
                .collect();
            transposed.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&col), col.len()));
        }
    
        // Column-wise 역변환
        let mut device_buffer = DeviceBuffer::new(eval_len)?;
        for i in 0..len {
            let coeffs = transposed[i].get_coefficients();
            let host_slice = HostSlice::from_slice(&coeffs);
            device_buffer.copy_from_host(&host_slice)?;
            ntt_inverse(&host_slice, &mut device_buffer)?;
            
            let mut host_result = vec![ScalarField::zero(); eval_len];
            let mut result_slice = HostSlice::from_mut_slice(&mut host_result);
            device_buffer.copy_to_host(&mut result_slice)?;
            
            transposed[i] = DensePolynomial::from_coeffs(HostSlice::from_slice(&host_result), host_result.len());
        }
    
        // 결과 행렬 만들기
        let mut final_rows = Vec::with_capacity(eval_len);
        for i in 0..eval_len {
            let row: Vec<_> = (0..len)
                .map(|j| transposed[j].get_coefficients()[i].clone())
                .collect();
            final_rows.push(row);
        }
    
        Ok(BivariatePolynomial::new(final_rows))
    }

    pub fn test_multiply_bivariates(a: &Self, b: &Self) -> Result<Self, NTTError> {
        let max_degree = a.x_degree.max(a.y_degree).max(b.x_degree).max(b.y_degree);
        let padded_size = (max_degree + 1).next_power_of_two() * 2;  // 두 배로 확장
        
        // NTT 평가
        let a_evals = a.evaluate_ntt(1, 1, Some(padded_size), Some(padded_size))?;
        let b_evals = b.evaluate_ntt(1, 1, Some(padded_size), Some(padded_size))?;
    
        assert_eq!(a_evals.len(), b_evals.len(), "Evaluation lengths must match");
        
        // Pointwise multiplication
        let mut mul_evals = Vec::with_capacity(a_evals.len());
        for i in 0..a_evals.len() {
            mul_evals.push(DensePolynomialExt::mul(&a_evals[i], &b_evals[i]));
        }
    
        // 정규화 스케일링 팩터 계산
        let n = ScalarField::from_u32(padded_size as u32);
        let scale_factor = n.mul(n).inv();  // n^2의 역수
    
        // Row-wise NTT 역변환과 정규화
        let mut result_rows = Vec::new();
        for i in 0..mul_evals.len() {
            let mut row = mul_evals[i].get_coefficients().to_vec();
            for j in 0..row.len() {
                row[j] = row[j].mul(scale_factor);
            }
            result_rows.push(DensePolynomial::from_coeffs(HostSlice::from_slice(&row), row.len()));
        }
    
        // 역변환
        let result = Self::interpolate_ntt(&result_rows)?;
    
        // 결과 크기 조정
        let target_size = max_degree + 1;
        let mut final_rows = Vec::new();
        
        for i in 0..target_size {
            if i < result.coefficients.len() {
                let mut row = result.coefficients[i]
                    .get_coefficients()
                    .iter()
                    .take(target_size)
                    .cloned()
                    .collect::<Vec<_>>();
                final_rows.push(row);
            }
        }
    
        Ok(BivariatePolynomial::new(final_rows))
    }

    /// Evaluate with offset (scale by offset, then do evaluate_ntt)
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

    /// Interpolate from offset-based NTT (inverse scale)
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

    // A simple global initialization for NTT domain
    static INIT: Once = Once::new();

    fn initialize_ntt_domain() {
        INIT.call_once(|| {
            let root = get_root_of_unity::<ScalarField>(8);
            initialize_domain(root, &NTTInitDomainConfig::default())
                .expect("Failed to initialize NTT domain");
        });
    }

    // Helper: polynomial A = 3 + x + 2x*y + x^2*y + 4x*y^2
    fn polynomial_a() -> BivariatePolynomial {
        let rows = vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0)],
        ];
        BivariatePolynomial::new(rows)
    }

    // Helper: polynomial B = 1 + 2x + 3y + 4xy
    fn polynomial_b() -> BivariatePolynomial {
        let rows = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4), ScalarField::from_u32(0)],
            vec![ScalarField::zero(), ScalarField::zero(), ScalarField::zero()],
        ];
        BivariatePolynomial::new(rows)
    }

    #[test]
    fn test_evaluation_ntt() {
        // For smaller polynomials, domain can be smaller
        // but let's keep consistent domain usage
        initialize_ntt_domain();
        
        let a_poly = polynomial_a();
        let evals = a_poly.evaluate_ntt(1, 1, None, None)
            .expect("Failed to evaluate NTT");
        
        let interpolated = BivariatePolynomial::interpolate_ntt(&evals)
            .expect("Failed to interpolate NTT");

        // Zero-pad expectation for 4x4
        let poly_a_zero_pad = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::zero(), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::from_u32(4), ScalarField::zero(), ScalarField::zero()],
            vec![ScalarField::zero(), ScalarField::zero(), ScalarField::zero(), ScalarField::zero()]
        ]);

        for (poly_a_zero_pad_row, interp_row) in poly_a_zero_pad.coefficients.iter().zip(interpolated.coefficients.iter()) {
            assert_eq!(poly_a_zero_pad_row.get_coefficients(), interp_row.get_coefficients());
        }
    }

    #[test]
    fn test_multiply_bivariates() {
        initialize_ntt_domain();

        let a_poly = polynomial_a();
        let b_poly = polynomial_b();

        let expected = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(7), ScalarField::from_u32(2), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(9), ScalarField::from_u32(17), ScalarField::from_u32(9), ScalarField::from_u32(2)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(10), ScalarField::from_u32(19), ScalarField::from_u32(4)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(12), ScalarField::from_u32(16), ScalarField::from_u32(0)]
        ]);

        let result = BivariatePolynomial::test_multiply_bivariates(&a_poly, &b_poly)
            .expect("Failed to multiply polynomials");

        // 차원 확인
        // println!("Result dimensions: {} x {}", 
        //         result.coefficients.len(), 
        //         result.coefficients[0].get_coefficients().len());
        // println!("Expected dimensions: {} x {}", 
        //         expected.coefficients.len(), 
        //         expected.coefficients[0].get_coefficients().len());

        // 각 원소 비교
        // for i in 0..expected.coefficients.len() {
        //     let result_row = result.coefficients[i].get_coefficients();
        //     let expected_row = expected.coefficients[i].get_coefficients();
        //     for j in 0..expected_row.len() {
        //         assert_eq!(result_row[j], expected_row[j], 
        //                 "Mismatch at position [{}, {}]", i, j);
        //     }
        // }
    }

        // assert_eq!(mul_evals, a_times_b);

        // let a_times_b_zero_pad = BivariatePolynomial::new(vec![
        //     vec![ScalarField::from_u32(3), ScalarField::from_u32(7), ScalarField::from_u32(2), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(9), ScalarField::from_u32(17), ScalarField::from_u32(9), ScalarField::from_u32(2), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(10), ScalarField::from_u32(19), ScalarField::from_u32(4), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(12), ScalarField::from_u32(16), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        //     vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],

        // ]); 

        // let a_evals_zero_pad =  BivariatePolynomial::evaluate_ntt(&polynomial_a(), 1, 1, Some(8), Some(8)).unwrap();
            
        // let b_evals_zero_pad = BivariatePolynomial::evaluate_ntt(&polynomial_b(), 1, 1,  Some(8), Some(8)).unwrap();

        // // let mul_eval_zero_pad = a_evals_zero_pad * b_evals_zero_pad; 

        // let mut mul_eval_zero_pad = Vec::with_capacity(a_evals_zero_pad.len());
        // for i in 0..a_evals_zero_pad.len() {
        //     let eval_a = &a_evals_zero_pad[i];
        //     let eval_b = &b_evals_zero_pad[i];
        //     assert_eq!(eval_a.get_nof_coeffs(), eval_b.get_nof_coeffs());
        //     mul_eval_zero_pad.push(eval_a.mul(eval_b));
        // }

        // let mul_poly = BivariatePolynomial::interpolate_ntt(&mul_eval_zero_pad).unwrap();

        // // Interpolate
        // let result = BivariatePolynomial::interpolate_ntt(&mul_evals)
        //     .expect("Failed to interpolate product");

        
        // assert_eq!(result.coefficients.len(), 8);
        
        // println!("test_multiply_bivariates finished successfully!");
    // }
}
