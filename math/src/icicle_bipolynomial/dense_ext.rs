use icicle_bls12_381::polynomials::DensePolynomial;
use std::vec::Vec;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::traits::FieldImpl;

pub trait DensePolynomialExt {
    fn zero() -> Self;
    fn get_coefficients(&self) -> Vec<ScalarField>;
    fn scale(&self, scalar: &ScalarField) -> Self;
    fn sub_constant(&mut self, constant: ScalarField);
    fn get_constant(&self) -> ScalarField;
    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str>;
    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial;
    fn mul(&self, other: &DensePolynomial) -> DensePolynomial;
}

impl DensePolynomialExt for DensePolynomial {
    fn zero() -> Self {
        DensePolynomial::from_coeffs(HostSlice::from_slice(&[ScalarField::zero()]), 1)
    }

    fn get_coefficients(&self) -> Vec<ScalarField> {
        let n = self.get_nof_coeffs();
        let mut coeffs = vec![ScalarField::zero(); n as usize];
        self.copy_coeffs(0, HostSlice::from_mut_slice(&mut coeffs));
        coeffs
    }

    fn scale(&self, scalar: &ScalarField) -> Self {
        let new_coeffs: Vec<ScalarField> = self
            .get_coefficients()
            .iter()
            .map(|c| *c * *scalar)
            .collect();

        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&new_coeffs),
            new_coeffs.len()
        )
    }

    fn sub_constant(&mut self, constant: ScalarField) {
        let mut coeffs = self.get_coefficients();
        if coeffs.is_empty() {
            panic!("Polynomial has no coefficients.");
        }
        coeffs[0] = coeffs[0] - constant; 
        *self = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&coeffs),
            coeffs.len()
        );
    }

    fn get_constant(&self) -> ScalarField {
        self.get_coeff(0)
    }

    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str> {
        let n = self.get_nof_coeffs();
        if n == 0 {
            return Err("Polynomial has no coefficients.");
        }

        let coeffs = self.get_coefficients();
        if n == 1 {
            return Ok((
                DensePolynomial::from_coeffs(HostSlice::from_slice(&[]), 0),
                coeffs[0]
            ));
        }

        let mut temp = coeffs[n as usize - 1];
        let mut q = Vec::with_capacity(n as usize - 1);
        q.push(temp);

        for i in (0..n-1).rev() {
            temp = temp * *b + coeffs[i as usize];
            if i > 0 {
                q.insert(0, temp);
            }
        }

        let remainder = temp;

        Ok((
            DensePolynomial::from_coeffs(HostSlice::from_slice(&q), q.len()),
            remainder
        ))
    }

    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial {
        let a = self.get_coefficients();
        let b = other.get_coefficients();
        let max_len = a.len().max(b.len());
        let mut sum = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let aa = if i < a.len() { a[i] } else { ScalarField::zero() };
            let bb = if i < b.len() { b[i] } else { ScalarField::zero() };
            sum.push(aa + bb);
        }

        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&sum),
            sum.len()
        )
    }

    fn mul(&self, other: &DensePolynomial) -> DensePolynomial {
        let self_coeffs = self.get_coefficients();
        let other_coeffs = other.get_coefficients();
        let n = self_coeffs.len();
        let m = other_coeffs.len();
        let result_len = n + m - 1;
        
        let mut result = vec![ScalarField::zero(); result_len];
        
        for i in 0..n {
            for j in 0..m {
                result[i + j] = result[i + j] + (self_coeffs[i] * other_coeffs[j]);
            }
        }
        
        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&result),
            result.len()
        )
    }
}
