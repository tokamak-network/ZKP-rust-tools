use lambdaworks_math::fft::errors::FFTError;
use lambdaworks_math::fft::polynomial::{evaluate_fft_cpu, interpolate_fft_cpu};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use ndarray::{Array2, Axis};

use crate::bipolynomial::BivariatePolynomial;

impl<E: IsField> BivariatePolynomial<FieldElement<E>> {
    /// Returns `N*M` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_x_size).next_power_of_two() * x_blowup_factor`.
    /// If `domain_x_size` or `domain_y_size` is `None`, it defaults to 0.
    pub fn evaluate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        bipoly: &BivariatePolynomial<FieldElement<E>>,
        x_blowup_factor: usize,
        y_blowup_factor: usize,
        domain_x_size: Option<usize>,
        domain_y_size: Option<usize>,
    ) -> Result<Array2<FieldElement<E>>, FFTError> {
        let domain_x_size = domain_x_size.unwrap_or(0);
        let domain_y_size = domain_y_size.unwrap_or(0);

        let len_x =
            core::cmp::max(bipoly.x_degree, domain_x_size).next_power_of_two() * x_blowup_factor;
        let len_y =
            core::cmp::max(bipoly.y_degree, domain_y_size).next_power_of_two() * y_blowup_factor;

        // Initialize the coefficients array with zeros and copy the polynomial's coefficients
        let mut coeffs = Array2::<FieldElement<E>>::from_elem((len_y, len_x), FieldElement::zero());
        for (i, row) in bipoly.coefficients.axis_iter(Axis(0)).enumerate() {
            for (j, coeff) in row.iter().enumerate() {
                coeffs[(i, j)] = coeff.clone();
            }
        }

        // Perform FFT row-wise
        for mut row in coeffs.axis_iter_mut(Axis(0)) {
            let fft_result = evaluate_fft_cpu::<F, E>(&row.to_vec())?;
            row.assign(&ndarray::Array1::from(fft_result));
        }

        // Transpose the array to perform FFT column-wise
        let mut transposed_coeffs = coeffs.reversed_axes();

        // Perform FFT column-wise
        for mut col in transposed_coeffs.axis_iter_mut(Axis(0)) {
            let fft_result = evaluate_fft_cpu::<F, E>(&col.to_vec())?;
            col.assign(&ndarray::Array1::from(fft_result));
        }

        // Transpose back to the original orientation
        let final_coeffs = transposed_coeffs.reversed_axes();

        Ok(final_coeffs)
    }

    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &Array2<FieldElement<E>>,
    ) -> Result<Self, FFTError> {
        let len_y = fft_evals.nrows();
        let len_x = fft_evals.ncols();

        // Create an Array2 for the x inverse FFT results
        let mut x_ifft = Array2::<FieldElement<E>>::default((len_y, len_x));

        // Perform IFFT for each row
        for (i, val) in fft_evals.axis_iter(Axis(0)).enumerate() {
            let result_poly = interpolate_fft_cpu::<F, E>(&val.to_vec())?;
            x_ifft
                .row_mut(i)
                .assign(&ndarray::Array1::from(result_poly.coefficients));
        }

        // Transpose x_ifft to prepare for the y IFFT
        let transposed_x_fft = x_ifft.reversed_axes();

        // Create an Array2 for the y inverse FFT results
        let mut y_ifft = Array2::<FieldElement<E>>::default((len_x, len_y));

        // Perform IFFT for each transposed row
        for (i, val) in transposed_x_fft.axis_iter(Axis(0)).enumerate() {
            let result_poly = interpolate_fft_cpu::<F, E>(&val.to_vec())?;
            y_ifft
                .row_mut(i)
                .assign(&ndarray::Array1::from(result_poly.coefficients));
        }

        // Transpose back to the original orientation
        let final_coeffs = y_ifft.reversed_axes();

        Ok(BivariatePolynomial::new(final_coeffs))
    }
}