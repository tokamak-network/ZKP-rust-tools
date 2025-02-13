extern crate icicle_bls12_381;
extern crate icicle_core;
extern crate icicle_runtime;
use icicle_bls12_381::curve::{ScalarField, ScalarCfg};
use icicle_core::traits::{FieldImpl, FieldConfig, GenerateRandom};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::{ntt, ntt::NTTInitDomainConfig};
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice, DeviceSlice, DeviceVec};
use std::ops::Deref;
use std::{
    clone, cmp,
    ops::{Add, AddAssign, Div, Mul, Rem, Sub},
    ptr, slice,
};

pub struct DensePolynomialExt {
    pub poly: DensePolynomial,
    pub x_degree: i64,
    pub y_degree: i64,
    pub x_size: u64,
    pub y_size: u64,
    ntt_rou: ScalarField,
    ntt_dom_config: NTTInitDomainConfig
}

impl DensePolynomialExt {
    // Inherit DensePolynomial
    pub fn print(&self) {
        unsafe {
            self.poly.print()
        }
    }
    // Inherit DensePolynomial
    pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<ScalarField> {
        unsafe {
            self.coeffs_mut_slice()          
        }
    }

    // Method to get the degree of the polynomial.
    pub fn degree(&self) -> Vec<i64> {
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

    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, _x_size: usize, _y_size: usize) -> Self {
        unsafe{
            let x_size = _x_size as u64;
            let y_size = _y_size as u64;

            let poly = DensePolynomial::from_coeffs(coeffs, x_size as usize * y_size as usize);
            let mut _x_degree: u64 = 0;
            let mut _y_degree: u64 = 0;

            for x_offset in (0 .. x_size).rev() {
                let sub_poly_y = poly.slice(x_offset, x_size, y_size);
                _y_degree = sub_poly_y.degree() as u64;
                if _y_degree > 0 {
                    _x_degree = x_offset;
                    break;
                }
            }

            Self{
                poly,
                x_degree: _x_degree as i64,
                y_degree: _y_degree as i64,
                x_size,
                y_size,
                ntt_rou: ntt::get_root_of_unity::<Self::Field>( x_size * y_size ),
                ntt_dom_config: NTTInitDomainConfig::default()
            }
        }
    }

    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, _x_size: usize, _y_size: usize) -> Self {
        unsafe{
            let x_size = _x_size as u64;
            let y_size = _y_size as u64;
            let _ntt_rou = ntt::get_root_of_unity::<Self::Field>( x_size * y_size );
            let _ntt_dom_config = NTTInitDomainConfig::default();

            ntt::initialize_domain::<Self::Field>(_ntt_rou, &_ntt_dom_config).unwrap();

            let mut ntt_result = DeviceVec::device_malloc(_x_size * _y_size).unwrap();
            
            // FFT along X
            let mut cfg = ntt::NTTConfig::<Self::Field>::default();
            cfg.batch_size = y_size as i32;
            cfg.columns_batch = false;
            ntt::ntt(evals, ntt::NTTDir::kForward, &cfg, &mut ntt_result).unwrap();
            cfg.batch_size = x_size as i32;
            cfg.columns_batch = true;
            ntt::ntt_inplace(&mut ntt_result, ntt::NTTDir::kForward, &cfg).unwrap();

            let poly = DensePolynomial::from_coeffs(&ntt_result, x_size as usize * y_size as usize);

            let mut _x_degree: u64 = 0;
            let mut _y_degree: u64 = 0;

            for x_offset in (0 .. x_size).rev() {
                let sub_poly_y = poly.slice(x_offset, x_size, y_size);
                _y_degree = sub_poly_y.degree() as u64;
                if _y_degree > 0 {
                    _x_degree = x_offset;
                    break;
                }
            }

            Self{
                poly,
                x_degree: _x_degree as i64,
                y_degree: _y_degree as i64,
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
        let mut coef_slice = vec![Self::Field::zero(); self.x_size as usize * self.y_size as usize];
        let mut coeffs = HostSlice::from_mut_slice(&mut coef_slice);
        self.copy_coeffs(0, coeffs);

        let x_size = self.x_degree as usize + 1;
        let y_size = self.y_degree as usize + 1;
        let mut result_slice = vec![Self::Field::zero(); y_size];
        let mut result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..y_size {
            let sub_xpoly_coef_slice = &coef_slice[offset*x_size .. (offset+1)*x_size];
            let sub_xpoly = DensePolynomial::from_coeffs(HostSlice::from_slice(&sub_xpoly_coef_slice), x_size); 
            result[offset] = sub_xpoly.eval(x);
        }

        Self {
            poly: DensePolynomial::from_coeffs(result, y_size),
            x_degree: 0,
            y_degree: self.y_degree.clone(),
            x_size: 1,
            y_size: y_size as u64,
            ntt_rou: self.ntt_rou.clone(),
            ntt_dom_config: self.ntt_dom_config.clone(),
        }
    }

    fn eval_y(&self, y: &Self::Field) -> Self {
        let mut coef_slice = vec![Self::Field::zero(); self.x_size as usize * self.y_size as usize];
        let mut coeffs = HostSlice::from_mut_slice(&mut coef_slice);
        self.copy_coeffs(0, coeffs);

        let x_size = self.x_degree as usize + 1;
        let y_size = self.y_degree as usize + 1;
        let mut result_slice = vec![Self::Field::zero(); x_size];
        let mut result = HostSlice::from_mut_slice(&mut result_slice);

        for offset in 0..x_size {
            let sub_ypoly_coef_slice: Vec<_> = coef_slice
                .chunks_exact(x_size)
                .map(|chunk| chunk[offset]) 
                .collect();
            let sub_ypoly = DensePolynomial::from_coeffs(HostSlice::from_slice(&sub_ypoly_coef_slice), y_size); 
            result[offset] = sub_ypoly.eval(y);
        }

        Self {
            poly: DensePolynomial::from_coeffs(result, x_size),
            x_degree: self.x_degree.clone(),
            y_degree: 0,
            x_size: x_size as u64,
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
    let x_size = 4;
    let y_size = 2;
    let size = x_size * y_size;
    let mut coeffs_vec = vec![ScalarField::one(); size];
    let coeffs = HostSlice::from_slice(&coeffs_vec);
    let mut evals = DeviceVec::<ScalarField>::device_malloc(size).unwrap();

    ntt::initialize_domain::<ScalarField>(
        ntt::get_root_of_unity::<ScalarField>(size as u64),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    // Using default config
    let mut cfg = ntt::NTTConfig::<ScalarField>::default();
    cfg.batch_size = y_size as i32;
    cfg.columns_batch = false;

    // Computing NTT columns batch
    ntt::ntt(
        coeffs,
        ntt::NTTDir::kInverse,
        &cfg,
        &mut evals,
    )
    .unwrap();

    // Using default config
    let mut cfg = ntt::NTTConfig::<ScalarField>::default();
    cfg.batch_size = x_size as i32;
    cfg.columns_batch = true;

    // Computing NTT columns batch
    let mut evals2 = DeviceVec::<ScalarField>::device_malloc(size).unwrap();
    ntt::ntt(
        &evals,
        ntt::NTTDir::kInverse,
        &cfg,
        &mut evals2,
    )
    .unwrap();

    let poly1 = DensePolynomialExt::from_coeffs(coeffs, x_size, y_size);
    let poly2 = DensePolynomialExt::from_rou_evals(&evals2, x_size, y_size);

    let mut coeff1_vec = vec![ScalarField::zero(); size];
    let mut coeff2_vec = vec![ScalarField::zero(); size];
    let coeff1 = HostSlice::from_mut_slice(&mut coeff1_vec);
    let coeff2 = HostSlice::from_mut_slice(&mut coeff2_vec);
    poly1.copy_coeffs(0, coeff1);
    poly2.copy_coeffs(0, coeff2);
    println!("coeffs = {:?}", coeff1_vec);
    println!("evals2 = {:?}", coeff2_vec);
    
    let x = ScalarCfg::generate_random(1)[0];
    let y = ScalarCfg::generate_random(1)[0];

    let eval1 = poly1.eval(&x, &y);
    let eval2 = poly2.eval(&x, &y);
    
    println!("eval result = {:?}", ScalarField::eq(&eval1, &eval2));
}