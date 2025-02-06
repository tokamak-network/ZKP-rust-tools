// src/icicle_bipolynomial/dense_ext.rs

use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_bls12_381::curve::ScalarField;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::FieldImpl;
use icicle_runtime::memory::HostSlice;

pub trait DensePolynomialExt {
    fn get_coefficients(&self) -> Vec<ScalarField>;
    fn scale(&self, scalar: &ScalarField) -> Self;
    fn sub_constant(&mut self, constant: ScalarField);
    fn get_constant(&self) -> ScalarField;
    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str>;
    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial;
    fn zero() -> DensePolynomial;
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
        let new_cfs: Vec<ScalarField> = self.get_coefficients()
            .iter()
            .map(|c| *c * *scalar)
            .collect();

        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&new_cfs),
            new_cfs.len()
        )
    }

    fn sub_constant(&mut self, constant: ScalarField) {
        let mut cfs = self.get_coefficients();
        if cfs.is_empty() {
            panic!("Polynomial has no coefficients");
        }
        cfs[0] = cfs[0] - constant;
        *self = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&cfs),
            cfs.len()
        );
    }

    fn get_constant(&self) -> ScalarField {
        self.get_coeff(0)
    }

    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str> {
        let n = self.get_nof_coeffs();
        if n == 0 {
            return Err("Polynomial has no coefficients");
        }
        let cfs = self.get_coefficients();
        if n == 1 {
            // 상수항만 있음
            return Ok((DensePolynomial::from_coeffs(HostSlice::from_slice(&[]), 0), cfs[0]));
        }

        let mut result = vec![cfs[(n-1) as usize]];
        let mut temp = cfs[(n-1) as usize];
        for i in (0..(n-1)).rev() {
            temp = temp * *b + cfs[i as usize];
            result.push(temp);
        }

        let remainder = result.pop().unwrap();
        result.reverse();
        let quotient = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&result),
            result.len()
        );
        Ok((quotient, remainder))
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
        let a = self.get_coefficients();
        let b = other.get_coefficients();
        let mut product = vec![ScalarField::zero(); a.len() + b.len() - 1];

        for (i, &aa) in a.iter().enumerate() {
            for (j, &bb) in b.iter().enumerate() {
                product[i + j] = product[i + j] + (aa * bb);
            }
        }

        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&product),
            product.len()
        )
    }
}
