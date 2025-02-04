use icicle_bls12_381::polynomials::DensePolynomial;
use alloc::vec::Vec;

use core::ops::{Add, Sub};
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::{polynomials::UnivariatePolynomial, traits::FieldImpl};

use super::dense_ext::DensePolynomialExt;


pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>,
    pub x_degree: usize,
    pub y_degree: usize,
}

impl Add for BivariatePolynomial {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result_coeffs = Vec::with_capacity(self.coefficients.len().max(other.coefficients.len()));

        for i in 0..result_coeffs.capacity() {
            let zero_poly = DensePolynomial::zero();
            let self_poly = self.coefficients.get(i).unwrap_or(&zero_poly);
            let other_poly = other.coefficients.get(i).unwrap_or(&zero_poly);
            result_coeffs.push(self_poly.add_polynomial(other_poly));
        }

        BivariatePolynomial {
            coefficients: result_coeffs,
            x_degree: self.x_degree.max(other.x_degree),
            y_degree: self.y_degree.max(other.y_degree),
        }
    }
}

impl Sub for BivariatePolynomial {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_y_degree = self.y_degree.max(other.y_degree);
        let max_x_degree = self.x_degree.max(other.x_degree);

        let mut result_coeffs = Vec::with_capacity(max_y_degree + 1);

        for i in 0..=max_y_degree {
            let self_coeffs = if i < self.coefficients.len() {
                self.coefficients[i].get_coefficients()
            } else {
                vec![ScalarField::zero(); max_x_degree + 1]
            };

            let other_coeffs = if i < other.coefficients.len() {
                other.coefficients[i].get_coefficients()
            } else {
                vec![ScalarField::zero(); max_x_degree + 1]
            };

            let mut row_coeffs = vec![];
            for j in 0..=max_x_degree {
                let a = if j < self_coeffs.len() { self_coeffs[j] } else { ScalarField::zero() };
                let b = if j < other_coeffs.len() { other_coeffs[j] } else { ScalarField::zero() };
                row_coeffs.push(a - b);
            }

            result_coeffs.push(DensePolynomial::from_coeffs(
                HostSlice::from_slice(&row_coeffs),
                row_coeffs.len(),
            ));
        }

        Self {
            coefficients: result_coeffs,
            x_degree: max_x_degree,
            y_degree: max_y_degree,
        }
    }
}


impl BivariatePolynomial {
    pub fn new(rows: Vec<Vec<ScalarField>>) -> Self {
        // row(행) 개수를 그대로 y_degree 로
        let y_degree = rows.len(); // 예: row가 4개면 y_degree = 4
    
        // 각 행(row) 중에서 가장 긴 길이를 x_degree 로
        let x_degree = rows
            .iter()
            .map(|r| r.len())
            .max()
            .unwrap_or(0); // 아무것도 없으면 0
    
        // 이제 각 행을 x_degree 길이로 0-패딩
        let polys = rows.into_iter().map(|mut row| {
            if row.len() < x_degree {
                row.resize(x_degree, ScalarField::zero());
            }
            DensePolynomial::from_coeffs(HostSlice::from_slice(&row), row.len())
        }).collect();
    
        BivariatePolynomial {
            coefficients: polys,
            x_degree,
            y_degree,
        }
    }

    pub fn zero() -> Self {
        let z = vec![ScalarField::zero()];
        let slice = HostSlice::from_slice(&z);
        BivariatePolynomial {
            coefficients: vec![DensePolynomial::from_coeffs(slice, z.len())],
            x_degree: 0,
            y_degree: 0,
        }
    }

    fn dense_poly_new_monomial(c: ScalarField, deg: usize) -> DensePolynomial {
        let mut cf = vec![ScalarField::zero(); deg+1];
        cf[deg] = c;
        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&cf),
            cf.len()
        )
    }

    fn dense_poly_zero() -> DensePolynomial {
        let cf = vec![ScalarField::zero()];
        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&cf),
            cf.len()
        )
    }

    pub fn sub_by_field_element(&self, element: ScalarField) -> Self {
        // 새로운 Vec<Vec<ScalarField>> 생성
        let mut new_rows = Vec::new();
    
        for (i, poly) in self.coefficients.iter().enumerate() {
            let mut row = poly.get_coefficients();
    
            // 첫 번째 row의 첫 번째 계수만 업데이트
            if i == 0 {
                row[0] = row[0] - element;
            }
    
            new_rows.push(row);
        }
    
        // BivariatePolynomial::new를 사용하여 새로운 인스턴스 생성
        BivariatePolynomial::new(new_rows)
    }

    pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField {
        let mut acc = ScalarField::zero();
        for (i, poly) in self.coefficients.iter().enumerate() {
            let mut ypow = ScalarField::one();
            for _ in 0..i {
                ypow = ypow * *y;
            }
            let px = poly.eval(x);
            acc = acc + (px * ypow);
        }
        acc
    }

    pub fn flatten_out(&self) -> Vec<ScalarField> {
        let mut flattened = Vec::new();
    
        for row_poly in &self.coefficients {
            let row_coeffs = row_poly.get_coefficients();
            flattened.extend(row_coeffs.iter());
        }
    
        flattened
    }

    pub fn ruffini_division(
        &self,
        a: &ScalarField,  // (x-a)
        b: &ScalarField,  // (y-b)
    ) -> Result<(BivariatePolynomial, DensePolynomial), &'static str> {
        let mut q_xy_rows = Vec::new();
        let mut remainders = Vec::new();
    
        // 각 row에 대해 x로 나누기
        for poly in &self.coefficients {
            let (q, r) = poly.ruffini_division(a)?;
            q_xy_rows.push(q);
            remainders.push(r);
        }
    
        // remainder_y 다항식 구성
        let mut remainder_y = Self::dense_poly_zero();
        for (i, &rem) in remainders.iter().enumerate() {
            let monomial = Self::dense_poly_new_monomial(rem, i);
            remainder_y = remainder_y.add_polynomial(&monomial);
        }
    
        // y로 나누기
        let (q_y, _) = remainder_y.ruffini_division(b)?;
        let x_degree = self.x_degree.saturating_sub(1);
        let y_degree = self.y_degree;
    
        // 각 row 정규화
        let normalized_rows: Vec<_> = q_xy_rows
            .into_iter()
            .map(|row| {
                let mut coeffs = row.get_coefficients();
                coeffs.resize(x_degree + 1, ScalarField::zero());
                DensePolynomial::from_coeffs(
                    HostSlice::from_slice(&coeffs),
                    coeffs.len()
                )
            })
            .collect();
    
        Ok((
            Self {
                coefficients: normalized_rows,
                x_degree,
                y_degree,
            },
            q_y
        ))
    }

    pub fn scale(&self, x_factor: &ScalarField, y_factor: &ScalarField) -> Self {
        let mut scaled_coefficients = Vec::with_capacity(self.coefficients.len());
    
        for (i, row_poly) in self.coefficients.iter().enumerate() {
            // y^i 항의 계수에 (y_factor)^i를 곱함
            let mut y_power = ScalarField::one();
            for _ in 0..i {
                y_power = y_power * *y_factor;
            }
    
            let row_coeffs = row_poly.get_coefficients();
            let mut scaled_row = Vec::with_capacity(self.x_degree + 1);
    
            for (j, coeff) in row_coeffs.iter().enumerate() {
                // x^j 항의 계수에 (x_factor)^j를 곱함
                let mut x_power = ScalarField::one();
                for _ in 0..j {
                    x_power = x_power * *x_factor;
                }
                scaled_row.push(*coeff * y_power * x_power);
            }
    
            while scaled_row.len() <= self.x_degree {
                scaled_row.push(ScalarField::zero());
            }
    
            scaled_coefficients.push(
                DensePolynomial::from_coeffs(
                    HostSlice::from_slice(&scaled_row),
                    scaled_row.len()
                )
            );
        }
    
        Self {
            coefficients: scaled_coefficients,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }
}