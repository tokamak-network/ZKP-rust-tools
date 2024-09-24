
// use lambdaworks_math::fft::errors::FFTError;

// use lambdaworks_math::field::traits::{IsField, IsSubFieldOf};
// use lambdaworks_math::{
//     field::{
//         element::FieldElement,
//         traits::{IsFFTField, RootsConfig},
//     },
//     polynomial::Polynomial,
// };
// use lambdaworks_math::fft::cpu::{ops, roots_of_unity};


// pub fn interpolate_fft_cpu<F, E>(
//     fft_evals: &[FieldElement<E>],
// ) -> Result<Vec<FieldElement<E>>, FFTError>
// where
//     F: IsFFTField + IsSubFieldOf<E>,
//     E: IsField,
// {
//     let order = fft_evals.len().trailing_zeros();
//     let twiddles =
//         roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverseInversed)?;

//     let coeffs = ops::fft(fft_evals, &twiddles)?;

//     let scale_factor = FieldElement::from(fft_evals.len() as u64).inv().unwrap();
//     Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
// }