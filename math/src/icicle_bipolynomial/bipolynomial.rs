// src/icicle_bipolynomial/bipolynomial.rs

use icicle_bls12_381::curve::ScalarField;
use icicle_core::{polynomials::UnivariatePolynomial, traits::FieldImpl};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_runtime::memory::HostSlice;
use std::ops::{Add, Sub};

use super::dense_ext::DensePolynomialExt;

/// 이변수 다항식(2D). 내부적으로 여러 개의 `DensePolynomial`(단변수 다항식)을
/// y방향으로 쌓아둠. 즉, coefficients[i] = (단변수 in x) * y^i
pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>,
    pub x_degree: usize,
    pub y_degree: usize,
}

impl BivariatePolynomial {
    /// 새 이차 다항식 생성
    /// rows: 각 y^i 행에 대해 x방향의 계수를 담은 벡터
    /// 예: rows[i][j] = (coefficient for x^j * y^i)
    pub fn new(rows: Vec<Vec<ScalarField>>) -> Self {
        let y_degree = rows.len().saturating_sub(1);
        let x_degree = rows
            .iter()
            .map(|r| r.len())
            .max()
            .map(|len| len.saturating_sub(1))
            .unwrap_or(0);

        let polys = rows.into_iter().map(|row| {
            let mut row_vec = row;
            while row_vec.len() <= x_degree {
                row_vec.push(ScalarField::zero());
            }
            DensePolynomial::from_coeffs(
                HostSlice::from_slice(&row_vec),
                row_vec.len()
            )
        }).collect();

        BivariatePolynomial {
            coefficients: polys,
            x_degree,
            y_degree,
        }
    }

    /// 0 이차 다항식
    pub fn zero() -> Self {
        let z = vec![ScalarField::zero()];
        BivariatePolynomial {
            coefficients: vec![
                DensePolynomial::from_coeffs(HostSlice::from_slice(&z), z.len())
            ],
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

    /// p(x,y) = ∑_{i=0..y_degree} coefficients[i].eval(x) * (y^i)
    pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField {
        let mut acc = ScalarField::zero();
        for (i, poly) in self.coefficients.iter().enumerate() {
            let mut y_pow = ScalarField::one();
            for _ in 0..i {
                y_pow = y_pow * *y;
            }
            let px = poly.eval(x);
            acc = acc + (px * y_pow);
        }
        acc
    }

    /// (x-축 차수+1) * (y-축 차수+1) 길이로 계수를 1D 벡터로 펼침
    pub fn flatten_out(&self) -> Vec<ScalarField> {
        let total_len = (self.x_degree + 1) * (self.y_degree + 1);
        let mut flattened = Vec::with_capacity(total_len);

        for row_i in 0..=self.y_degree {
            let row_poly = &self.coefficients[row_i];
            let row_coeffs = row_poly.get_coefficients();
            for x_i in 0..=self.x_degree {
                let val = if x_i < row_coeffs.len() {
                    row_coeffs[x_i]
                } else {
                    ScalarField::zero()
                };
                flattened.push(val);
            }
        }

        flattened
    }

    /// Ruffini division 예시: (x - a), (y - b)로 나누는 시나리오
    ///  - 각 row = p_i(x)에 대해 (x-a)로 나눈 몫·나머지 -> 나머지를 y방향에 합성
    pub fn ruffini_division(
        &self,
        a: &ScalarField,
        b: &ScalarField,
    ) -> Result<(BivariatePolynomial, DensePolynomial), &'static str> {
        let mut q_xy_rows = Vec::new();
        let mut remainders = Vec::new();
    
        // (x - a)로 나눠서 row별로 몫·나머지 구함
        for poly in &self.coefficients {
            let (q, r) = poly.ruffini_division(a)?;
            q_xy_rows.push(q);
            remainders.push(r);
        }
    
        // remainders: r_i -> remainder_y( x^?, y^i ) 형태
        let mut remainder_y = DensePolynomial::zero();
        for (i, &rem) in remainders.iter().enumerate() {
            let mut cf = vec![ScalarField::zero(); i+1];
            cf[i] = rem;
            let monomial = DensePolynomial::from_coeffs(HostSlice::from_slice(&cf), cf.len());
            remainder_y = remainder_y.add_polynomial(&monomial);
        }
        // remainder_y를 (y - b)로 나누기
        let (q_y, _) = remainder_y.ruffini_division(b)?;
    
        // x_degree 하나 줄인 몫 bivariate
        let x_degree = self.x_degree.saturating_sub(1);
        let mut normalized_rows = Vec::new();
        for mut row in q_xy_rows {
            let mut cfs = row.get_coefficients();
            while cfs.len() <= x_degree {
                cfs.push(ScalarField::zero());
            }
            cfs.truncate(x_degree + 1);
            normalized_rows.push(DensePolynomial::from_coeffs(
                HostSlice::from_slice(&cfs),
                cfs.len()
            ));
        }
    
        Ok((
            BivariatePolynomial {
                coefficients: normalized_rows,
                x_degree,
                y_degree: self.y_degree,
            },
            q_y
        ))
    }

    /// (x_factor, y_factor)로 스케일링
    ///  = Σ_i [ (RowPoly in x) * (x_factor^j) * (y_factor^i) ]
    pub fn scale(&self, x_factor: &ScalarField, y_factor: &ScalarField) -> Self {
        let mut scaled_coeffs = Vec::with_capacity(self.coefficients.len());

        for (i, row_poly) in self.coefficients.iter().enumerate() {
            let mut y_pow = ScalarField::one();
            for _ in 0..i {
                y_pow = y_pow * *y_factor;
            }

            let row = row_poly.get_coefficients();
            let mut new_row = Vec::with_capacity(row.len());
            for (j, &coeff) in row.iter().enumerate() {
                let mut x_pow = ScalarField::one();
                for _ in 0..j {
                    x_pow = x_pow * *x_factor;
                }
                new_row.push(coeff * y_pow * x_pow);
            }

            scaled_coeffs.push(
                DensePolynomial::from_coeffs(
                    HostSlice::from_slice(&new_row),
                    new_row.len()
                )
            );
        }

        BivariatePolynomial {
            coefficients: scaled_coeffs,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }
}

// ------------------------------------------------------------------------
// 연산자 오버로드(Add, Sub)를 통해 이차 다항식 덧셈/뺄셈
// ------------------------------------------------------------------------
impl Add for BivariatePolynomial {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result_coeffs = Vec::with_capacity(self.coefficients.len().max(other.coefficients.len()));

        for i in 0..result_coeffs.capacity() {
            let zero_poly = DensePolynomial::zero();
            let spoly = self.coefficients.get(i).unwrap_or(&zero_poly);
            let opoly = other.coefficients.get(i).unwrap_or(&zero_poly);
            result_coeffs.push(spoly.add_polynomial(opoly));
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
        let mut result_coeffs = Vec::with_capacity(self.coefficients.len().max(other.coefficients.len()));

        for i in 0..result_coeffs.capacity() {
            let zero_poly = DensePolynomial::zero();
            let spoly = self.coefficients.get(i).unwrap_or(&zero_poly);
            let opoly = other.coefficients.get(i).unwrap_or(&zero_poly);

            // opoly에 -1을 곱해서 spoly와 add
            let neg_opoly = opoly.scale(&ScalarField::zero().sub(ScalarField::one())); // -1
            result_coeffs.push(spoly.add_polynomial(&neg_opoly));
        }

        BivariatePolynomial {
            coefficients: result_coeffs,
            x_degree: self.x_degree.max(other.x_degree),
            y_degree: self.y_degree.max(other.y_degree),
        }
    }
}
