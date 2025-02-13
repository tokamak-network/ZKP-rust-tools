use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::{FieldImpl, FieldConfig, GenerateRandom};
use icicle_core::{ntt, ntt::NTTInitDomainConfig};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice, DeviceSlice};
use std::{
    clone, cmp,
    ops::{Add, AddAssign, Div, Mul, Rem, Sub},
    ptr, slice,
};

pub struct DensePolynomialExt {
    pub poly: DensePolynomial,
    pub x_degree: usize,
    pub y_degree: usize,
    pub x_size: usize,
    pub y_size: usize,
    ntt_rou: ScalarField,
    ntt_dom_config: NTTInitDomainConfig
}

impl DensePolynomialExt {
    // Inherit DensePolynomial
    pub fn print(&self) {
        unsafe {
            &self.poly.print();
        }
    }
    // Inherit DensePolynomial
    pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<ScalarField> {
        unsafe {
            &mut self.coeffs_mut_slice()
        }
    }

    // Method to get the degree of the polynomial.
    pub fn degree(&self) -> Vec<usize> {
        vec![self.x_degree, self.y_degree]
    }
}

// impl Drop for DensePolynomialExt {
//     fn drop(&mut self) {
//         unsafe {
//             delete(self.poly);
//             delete(self.x_degree);
//             delete(self.y_degree);
//         }
//     }
// }

// impl Clone for DensePolynomialExt {
//     fn clone(&self) -> Self {
//         unsafe {
//             DensePolynomialExt {
//                 poly: clone(self.poly),
//                 x_degree: clone(self.x_degree),
//                 y_degree: clone(self.y_degree)
//             }
//         }
//     }
// }

pub trait BivariatePolynomial
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field: FieldImpl;
    type FieldConfig: FieldConfig;

    // Methods to create polynomials from coefficients or roots-of-unity evaluations.
    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, x_size: usize, y_size: usize) -> Self;
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, x_size: usize, y_size: usize) -> Self;

    // // Method to divide this polynomial by another, returning quotient and remainder.
    // fn divide_x(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    // // Method to divide this polynomial by another, returning quotient and remainder.
    // fn divide_y(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    // // Method to divide this polynomial by the vanishing polynomial 'X^N-1'.
    // fn div_by_vanishing_x(&self, degree: u64) -> Self;

    // // Method to divide this polynomial by the vanishing polynomial 'X^N-1'.
    // fn div_by_vanishing_y(&self, degree: u64) -> Self;

    // // Methods to add or subtract a monomial in-place.
    // fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    // fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);

    // // Method to slice the polynomial, creating a sub-polynomial.
    // fn slice(&self, offset: u64, stride: u64, size: u64) -> Self;

    // // Methods to return new polynomials containing only the even or odd terms.
    // fn even_x(&self) -> Self;
    // fn even_y(&self) -> Self;
    // fn odd_y(&self) -> Self;
    // fn odd_y(&self) -> Self;

    // Method to evaluate the polynomial at a given domain point.
    fn eval_x(&self, x: &Self::Field) -> Self;

    // Method to evaluate the polynomial at a given domain point.
    fn eval_y(&self, y: &Self::Field) -> Self;

    fn eval(&self, x: &Self::Field, y: &Self::Field) -> Self::Field;

    // // Method to evaluate the polynomial over a domain and store the results.
    // fn eval_on_domain<D_x: HostOrDeviceSlice<Self::Field> + ?Sized, D_y: HostOrDeviceSlice<Self::Field> + ?Sized, E: HostOrDeviceSlice<Self::Field> + ?Sized>(
    //     &self,
    //     domain_x: &D_x,
    //     domain_y: &D_y,
    //     evals: &mut E,
    // );

    // // Method to evaluate the polynomial over the roots-of-unity domain for power-of-two sized domain
    // fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, domain_log_size: u64, evals: &mut E);

    // Method to retrieve a coefficient at a specific index.
    fn get_coeff(&self, idx_x: u64, idx_y: u64) -> Self::Field;
    // fn get_nof_coeffs_x(&self) -> u64;
    // fn get_nof_coeffs_y(&self) -> u64;

    // Method to copy coefficients into a provided slice.
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S);

}

impl BivariatePolynomial for DensePolynomialExt {
    type Field = ScalarField;
    type FieldConfig = ScalarCfg;

    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, x_size: usize, y_size: usize) -> Self {
        unsafe{
            let _poly = DensePolynomial::from_coeffs(coeffs, x_size * y_size);
            let mut _x_degree: usize = 0;
            let mut _y_degree: usize = 0;

            for x_offset in (0 .. x_size).rev() {
                let sub_poly_y = _poly.slice(x_offset, x_size, y_size);
                _y_degree = sub_poly_y.degree() as usize;
                if _y_degree > 0 {
                    _x_degree = x_offset;
                    break;
                }
            }

            Self{
                poly: _poly,
                x_degree: _x_degree,
                y_degree: _y_degree,
                x_size,
                y_size,
                ntt_rou: ntt::get_root_of_unity::<Field>(
                    (x_size * y_size).try_into()
                        .unwrap(),
                ),
                ntt_dom_config: NTTInitDomainConfig::default()
            }
        }
    }

    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, x_size: usize, y_size: usize) -> Self {
        unsafe{
            let _ntt_rou = ntt::get_root_of_unity::<Field>(
                (x_size * y_size).try_into()
                    .unwrap(),
            );
            let _ntt_dom_config = NTTInitDomainConfig::default();

            ntt::initialize_domain::<Field>(_ntt_rou, &_ntt_dom_config).unwrap();

            let mut ntt_result = evals.clone();
            // FFT along X
            let mut cfg = ntt::NTTConfig::<Field>::default();
            cfg.batch_size = y_size;
            cfg.columns_batch = false;
            ntt::ntt_inplace(&mut ntt_result, ntt::NTTDir::kForward, &cfg).unwrap();
            cfg.batch_size = x_size;
            cfg.columns_batch = true;
            ntt::ntt_inplace(&mut ntt_result, ntt::NTTDir::kForward, &cfg).unwrap();

            Self{
                poly: DensePolynomial::from_coeffs(ntt_result, x_size, y_size),
                x_degree: _x_degree,
                y_degree: _y_degree,
                x_size,
                y_size,
                ntt_rou: _ntt_rou,
                ntt_dom_config: _ntt_dom_config
            }
        }
    }

    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S) {
        self.poly.copy_coeffs(start_idx, coeffs);
    }

    fn eval_x(&self, x: &Self::Field) -> Self {
        let mut coef_slice = vec![Field::zero(), self.x_size * self.y_size];
        let mut coeffs = HostSlice::from_mut_slice(&mut coef_slice);
        self.copy_coeffs(0, &mut coeffs);

        let x_size = self.x_degree + 1;
        let y_size = self.y_degree + 1;
        let mut result_slice = vec![Field::zero(), y_size];
        let mut result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..y_size {
            let sub_xpoly_coef_slice = coef_slice[offset*x_size .. (offset+1)*x_size];
            let sub_xpoly = DensePolynomial::from_coeffs(HostSlice::from_slice(&sub_xpoly_coef_slice), x_size); 
            result_slice[offset] = sub_xpoly.eval(x);
        }

        Self {
            poly: DensePolynomial::from_coeffs(result),
            x_degree: 0,
            y_degree: self.y_degree.clone(),
            x_size: 1,
            y_size,
            ntt_rou: self.ntt_rou.clone(),
            ntt_dom_config: self.ntt_dom_config.clone(),
        }
    }

    fn eval_y(&self, y: &Self::Field) -> Self {
        let mut coef_slice = vec![Field::zero(), self.x_size * self.y_size];
        let mut coeffs = HostSlice::from_mut_slice(&mut coef_slice);
        self.copy_coeffs(0, &mut coeffs);

        let x_size = self.x_degree + 1;
        let y_size = self.y_degree + 1;
        let mut result_slice = vec![Field::zero(), x_size];
        let mut result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..x_size {
            let sub_ypoly_coef_slice = coef_slice.slice(offset, x_size, y_size);
            let sub_ypoly = DensePolynomial::from_coeffs(HostSlice::from_slice(&sub_ypoly_coef_slice), y_size); 
            result_slice[offset] = sub_ypoly.eval(y);
        }

        Self {
            poly: DensePolynomial::from_coeffs(result),
            x_degree: self.x_degree.clone(),
            y_degree: 0,
            x_size,
            y_size: 1,
            ntt_rou: self.ntt_rou.clone(),
            ntt_dom_config: self.ntt_dom_config.clone(),
        }
    }

    fn eval(&self, x: &Self::Field, y: &Self::Field) -> Self::Field {
        let res1 = self.eval_x(x);
        let res2 = res1.eval_y(y);
        if !(res2.x_degree == 0 && res2.y_degree == 0) {
            panic!("Evaluation result is not a constant.");
        } else {
            res2.get_coeff(0,0)
        }
    }

    fn get_coeff(&self, idx_x: u64, idx_y: u64) -> Self::Field {
        if !(idx_x <= self.x_size && idx_y <= self.y_size){
            panic!("The index at which to get a coefficient exceeds the coefficient size.");
        }
        let idx = idx_x + idx_y * self.x_size;
        self.poly.get_coeff(idx)
    }

}

fn main() {
    let x_size = 3;
    let y_size = 2;
    let size = x_size * y_size;
    let coeffs = HostSlice::from_slice(&ScalarCfg::generate_random(size));
    let mut evals = coeffs.clone();

    ntt::initialize_domain::<ScalarField>(
        ntt::get_root_of_unity::<ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    // Using default config
    let mut cfg = ntt::NTTConfig::<ScalarField>::default();
    cfg.batch_size = y_size;
    cfg.columns_batch = false;

    // Computing NTT columns batch
    ntt::ntt(
        coeffs,
        ntt::NTTDir::kForward,
        &cfg,
        &mut evals,
    )
    .unwrap();

    // Using default config
    let mut cfg = ntt::NTTConfig::<ScalarField>::default();
    cfg.batch_size = x_size;
    cfg.columns_batch = true;

    // Computing NTT columns batch
    ntt::ntt(
        evals.clone(),
        ntt::NTTDir::kForward,
        &cfg,
        &mut evals,
    )
    .unwrap();
    
    let poly1 = DensePolynomialExt::from_coeffs(coeffs, x_size, y_size);
    let poly2 = DensePolynomialExt::from_rou_evals(evals, x_size, y_size);

    let x = ScalarCfg::generate_random(1)[0];
    let y = ScalarCfg::generate_random(1)[0];

    let eval1 = poly1.eval(&x, &y);
    let eval2 = poly2.eval(&x, &y);
    
    println!("eval result = {:?}", ScalarField::eq(&eval1, &eval2))
}