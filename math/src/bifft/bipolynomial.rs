
use lambdaworks_math::fft::errors::FFTError;

use lambdaworks_math::fft::polynomial::interpolate_fft_cpu;
use lambdaworks_math::field::traits::{IsField, IsSubFieldOf};

use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },

    fft::polynomial::evaluate_fft_cpu,
};
use alloc::{vec, vec::Vec};

use crate::bipolynomial::BivariatePolynomial;


impl<E: IsField> BivariatePolynomial<FieldElement<E>> {
    /// Returns `N*M` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_x_size).next_power_of_two() * x_blowup_factor`.
    /// If `domain_x_size` or `domain_y_size` is `None`, it defaults to 0.
    pub fn evaluate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        bipoly: &BivariatePolynomial<FieldElement<E>>,
        x_blowup_factor: usize, // 
        y_blowup_factor: usize,
        domain_x_size: Option<usize>, // 
        domain_y_size: Option<usize>,
    )  -> Result<Vec<Vec<FieldElement<E>>>, FFTError>  {
        let domain_x_size = domain_x_size.unwrap_or(0);
        let domain_y_size = domain_y_size.unwrap_or(0);

        let len_x = core::cmp::max(bipoly.x_degree, domain_x_size).next_power_of_two() * x_blowup_factor;
        let len_y = core::cmp::max(bipoly.y_degree, domain_y_size).next_power_of_two() * y_blowup_factor;

        // todo :: check vector of coefficients is not zero

        let mut coeffs = bipoly.coefficients.clone();
        
        let iter = coeffs.iter_mut();
        for val in iter {
            val.resize(len_x, FieldElement::zero());
            let result = evaluate_fft_cpu::<F, E>(&val);
            match result {
                Ok(fft_result) => (*val = fft_result),
                Err(e) => return Err(e),
            }
        }
        // Transpose coeffs vector of vector
        let mut transposed_coeffs: Vec<Vec<FieldElement<E>>> = Vec::new();
        for i in 0..len_x {
            let mut row: Vec<FieldElement<E>> = Vec::new();
            for j in 0..len_y {
                row.push(coeffs[j][i].clone());
            }
            transposed_coeffs.push(row);
        }

        let iter = transposed_coeffs.iter_mut();
        for val in iter {
            val.resize(len_y, FieldElement::zero());
            let result = evaluate_fft_cpu::<F, E>(&val);
            match result {
                Ok(fft_result) => *val = fft_result,
                Err(e) => return Err(e),
            }   
        }

                // Transpose coeffs vector of vector
        let mut final_transposed_coeffs: Vec<Vec<FieldElement<E>>> = Vec::new();
        for i in 0..len_y {
            let mut row: Vec<FieldElement<E>> = Vec::new();
            for j in 0..len_x {
                row.push(transposed_coeffs[j][i].clone());
            }
            final_transposed_coeffs.push(row);
        }

        Ok((final_transposed_coeffs))
    }

    // TODO :: if we import bipoly as mutable we can call scale_in_place which is more efficient that this one. // k.w^0 , .. . 
    pub fn evaluate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        bipoly: &BivariatePolynomial<FieldElement<E>>,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
        offset: &FieldElement<F>,
    ) -> Result<Vec<Vec<FieldElement<E>>>, FFTError> {
        let scaled = bipoly.scale(offset);
        BivariatePolynomial::evaluate_fft::<F>(&scaled, x_blowup_factor, y_blowup_factor, domain_x_size, domain_y_size)
    }


    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[&[FieldElement<E>]],
    ) -> Result<Self, FFTError> {

        let len_y =  fft_evals.len();
        let len_x = fft_evals[0].len();

        let mut x_ifft: Vec<Vec<FieldElement<E>>> = Vec::new();

        let iter = fft_evals.iter();
        for val in iter {
            if val.len() != len_x {
                return Err(FFTError::InputError(len_x));
            }
            let result_poly = interpolate_fft_cpu::<F, E>(&val);
            match result_poly {
                Ok(fft_result) => x_ifft.push(fft_result.coefficients),
                Err(e) => return Err(e),
            }
        }
        
        let mut transposed_x_fft: Vec<Vec<FieldElement<E>>> = Vec::new();
        for i in 0..len_x {
            let mut row: Vec<FieldElement<E>> = Vec::new();
            for j in 0..len_y {
                row.push(x_ifft[j][i].clone());
            }
            transposed_x_fft.push(row);
        }

        let mut y_ifft: Vec<Vec<FieldElement<E>>> = Vec::new();

        let iter = transposed_x_fft.iter_mut();
        for val in iter {
            if val.len() != len_y {
                return Err(FFTError::InputError(len_y));
            }
            let result_poly = interpolate_fft_cpu::<F, E>(&val);
            match result_poly {
                Ok(fft_result) => y_ifft.push(fft_result.coefficients),
                Err(e) => return Err(e),
            }
        }

        let mut transposed_y_fft: Vec<Vec<FieldElement<E>>> = Vec::new();
        for i in 0..len_y {
            let mut row: Vec<FieldElement<E>> = Vec::new();
            for j in 0..len_x {
                row.push(y_ifft[j][i].clone());
            }
            transposed_y_fft.push(row);
        }
        // let dd = transposed_y_fft as &[&[FieldElement<E>]];
        Ok(BivariatePolynomial::from_vec(transposed_x_fft))
    }
    // 
    pub fn interpolate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[&[FieldElement<E>]],
        offset: &FieldElement<F>,
    )  -> Result<Self, FFTError> {
        let scaled = BivariatePolynomial::interpolate_fft::<F>(fft_evals)?;
        Ok(scaled.scale(&offset.inv().unwrap()))
    } 
}



