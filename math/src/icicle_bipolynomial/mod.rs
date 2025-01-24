use icicle_core::polynomials::UnivariatePolynomial;

use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::traits::FieldImpl;
use alloc::vec::Vec;
use alloc::vec;
use core::ops::{Add, Sub};

/// Trait definition for DensePolynomialExt
pub trait DensePolynomialExt {
    fn zero() -> Self;
    fn get_coefficients(&self) -> Vec<ScalarField>;
    fn scale(&self, scalar: &ScalarField) -> Self;
    fn sub_constant(&mut self, constant: ScalarField);
    fn get_constant(&self) -> ScalarField;
    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str>;
    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial;
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
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_381::curve::ScalarField;

    fn create_field_elements() -> (ScalarField, ScalarField, ScalarField, ScalarField) {
        let zero = ScalarField::zero();
        let one = ScalarField::one();
        let two = one + one;
        let three = two + one;
        (zero, one, two, three)
    }

    // 3 + x + 2x*y + x^2*y + 4x*y^2
    fn polynomial_a() -> BivariatePolynomial {
        BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(1),
                ScalarField::zero()
            ],
            vec![
                ScalarField::zero(),
                ScalarField::from_u32(2),
                ScalarField::from_u32(1)
            ],
            vec![
                ScalarField::zero(),
                ScalarField::from_u32(4),
                ScalarField::zero()
            ]
        ])
    }

    // 1 + 2x + 3y + 4xy
    fn polynomial_b() -> BivariatePolynomial {
        BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::zero()
            ],
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(4),
                ScalarField::zero()
            ],
            vec![
                ScalarField::zero(),
                ScalarField::zero(),
                ScalarField::zero()
            ]
        ])
    }

    fn polynomial_one() -> BivariatePolynomial {
        BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1)
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1)
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1)
            ]
        ])
    }

    #[test]
    fn test_bp_new() {
        let poly = polynomial_b();
        assert_eq!(poly.x_degree, 3);
        assert_eq!(poly.y_degree, 3);
    }

    #[test]
    fn test_bivariate_polynomial_new() {
        // Example: 3 + x + 2xy + x^2y + 4xy^2
        let poly = polynomial_a();

        assert_eq!(poly.x_degree, 3);
        assert_eq!(poly.y_degree, 3);

        let expected_coeffs = polynomial_a().coefficients;
        for (actual, expected) in poly.coefficients.iter().zip(expected_coeffs.iter()) {
            assert_eq!(actual.get_coefficients(), expected.get_coefficients());
        }
    }

    #[test]
    fn test_evaluate() {
        let poly = polynomial_a();
        let x = ScalarField::from_u32(2);
        let y = ScalarField::from_u32(3);

        let result = poly.evaluate(&x, &y);

        // 3 + x + 2xy + x^2y + 4xy^2
        // = 3 + 2 + 2*2*3 + 2^2*3 + 4*2*3^2
        // = 3 + 2 + 12 + 12 + 72
        // = 101 mod ORDER
        let expected = ScalarField::from_u32(101);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero() {
        let zero_poly = BivariatePolynomial::zero();
        assert!(zero_poly.coefficients[0].get_coefficients().iter().all(|c| *c == ScalarField::zero()));
    }

    #[test]
    fn test_flatten_out() {
        let coeffs = vec![
            vec![ScalarField::from_u32(1), ScalarField::from_u32(2)], 
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4)],
        ];

        let poly = BivariatePolynomial::new(coeffs);
        let flattened = poly.flatten_out();

        let expected = vec![ScalarField::from_u32(1), ScalarField::from_u32(2), ScalarField::from_u32(3), ScalarField::from_u32(4)];

        assert_eq!(flattened, expected);
    }

    #[test]
    fn test_bivariate_polynomial_ruffini_division() {
        let (zero, one, _, _) = create_field_elements();
        
        let poly = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(14),
                ScalarField::from_u32(2),
                ScalarField::from_u32(1),
                zero
            ],
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(21),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1)
            ],
            vec![
                ScalarField::from_u32(21),
                ScalarField::from_u32(19),
                ScalarField::from_u32(4),
                ScalarField::zero()
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::zero(),
                ScalarField::zero(),
                ScalarField::zero()
            ]
        ]);

        assert_eq!(ScalarField::from_u32(253), poly.evaluate(&one, &ScalarField::from_u32(2)));

        let (q_xy, q_y) = poly.ruffini_division(&one, &ScalarField::from_u32(2))
            .expect("Ruffini division failed");

        let expected_q_xy = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(1),
                ScalarField::zero(),
                ScalarField::zero()
            ],
            vec![
                ScalarField::from_u32(23),
                ScalarField::from_u32(2),
                ScalarField::from_u32(1),
                ScalarField::zero()
            ],
            vec![
                ScalarField::from_u32(23),
                ScalarField::from_u32(4),
                ScalarField::zero(),
                ScalarField::zero()
            ],
            vec![
                ScalarField::zero(),
                ScalarField::zero(),
                ScalarField::zero(),
                ScalarField::zero()
            ]
        ]);

        let remainder_coeffs = vec![
            ScalarField::from_u32(118),
            ScalarField::from_u32(46),
            ScalarField::from_u32(1),
        ];
        
        let expected_q_y = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&remainder_coeffs),
            remainder_coeffs.len()
        );

        q_xy.coefficients.iter().zip(expected_q_xy.coefficients.iter()).for_each(|(q, expected_q)| {
            assert_eq!(q.get_coefficients(), expected_q.get_coefficients());
        });
        println!("{:?}", expected_q_y.get_coefficients());
        assert_eq!(q_xy.coefficients.len(), expected_q_xy.coefficients.len());
        assert_eq!(q_y.get_coefficients(), expected_q_y.get_coefficients());
    }

    #[test]
    fn test_polynomial_addition() {
        let p1 = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::from_u32(3)
            ],
            vec![
                ScalarField::from_u32(4),
                ScalarField::from_u32(5),
                ScalarField::from_u32(6) 
            ],
            vec![
                ScalarField::from_u32(4),
                ScalarField::from_u32(5),
                ScalarField::from_u32(6) 
            ],
        ]);

        let p2 = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(6),
                ScalarField::from_u32(5),
                ScalarField::from_u32(4)
            ],
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(2),
                ScalarField::from_u32(1) 
            ],
        ]);

        let expected = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(7), 
                ScalarField::from_u32(7), 
                ScalarField::from_u32(7)
            ],
            vec![
                ScalarField::from_u32(7), 
                ScalarField::from_u32(7), 
                ScalarField::from_u32(7)  
            ],
            vec![
                ScalarField::from_u32(4), 
                ScalarField::from_u32(5), 
                ScalarField::from_u32(6)  
            ],
        ]);
        let result = p1 + p2;

        assert_eq!(expected.coefficients.len(), result.coefficients.len());
        for (exp_row, res_row) in expected.coefficients.iter().zip(result.coefficients.iter()) {
            assert_eq!(exp_row.get_coefficients(), res_row.get_coefficients());
        }
    }

    #[test]
    fn test_polynomial_subtraction() {
        let p1 = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(2),
                ScalarField::from_u32(2)
            ],
            vec![
                ScalarField::from_u32(4),
                ScalarField::from_u32(5),
                ScalarField::from_u32(6) 
            ],
            vec![
                ScalarField::from_u32(7),
                ScalarField::from_u32(8),
                ScalarField::from_u32(9) 
            ],
        ]);

        let p2 = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
                ScalarField::from_u32(2)
            ],
            vec![
                ScalarField::from_u32(2),
                ScalarField::from_u32(3),
                ScalarField::from_u32(4) 
            ],
            vec![
                ScalarField::from_u32(5),
                ScalarField::from_u32(6),
                ScalarField::from_u32(7) 
            ],
        ]);

        let expected = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(2), 
                ScalarField::from_u32(1), 
                ScalarField::from_u32(0)
            ],
            vec![
                ScalarField::from_u32(2), 
                ScalarField::from_u32(2), 
                ScalarField::from_u32(2)  
            ],
            vec![
                ScalarField::from_u32(2), 
                ScalarField::from_u32(2), 
                ScalarField::from_u32(2)  
            ],
        ]);
        let result = p1 - p2;

        // assert_eq!(expected.coefficients.len(), result.coefficients.len());
        for (exp_row, res_row) in expected.coefficients.iter().zip(result.coefficients.iter()) {
            assert_eq!(exp_row.get_coefficients(), res_row.get_coefficients());
        }
    }

    #[test]
    fn test_sub_by_field_element() { // test case 추가
        let coeffs = vec![
            vec![ScalarField::from_u32(5), ScalarField::from_u32(2)],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(4)]
        ];
        let poly = BivariatePolynomial::new(coeffs);

        let element_to_subtract = ScalarField::from_u32(3);

        let result = poly.sub_by_field_element(element_to_subtract);

        let expected = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(2), 
                ScalarField::from_u32(2),
            ],
            vec![
                ScalarField::from_u32(3),
                ScalarField::from_u32(4),
              
            ],
        ]);

        // assert_eq!(expected.coefficients.len(), result.coefficients.len());
        for (exp_row, res_row) in expected.coefficients.iter().zip(result.coefficients.iter()) {
            assert_eq!(exp_row.get_coefficients(), res_row.get_coefficients());
        }
    }

    #[test]
    fn test_scale() {
        let poly = polynomial_one();
        
        // x 방향 스케일링
        let x_scaled = poly.scale(&ScalarField::from_u32(2), &ScalarField::from_u32(1));
        let expected_x = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::from_u32(4)
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::from_u32(4)
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::from_u32(4)
            ]
        ]);
        
        assert_eq!(expected_x.coefficients.len(), x_scaled.coefficients.len());
        for (exp_row, scaled_row) in expected_x.coefficients.iter().zip(x_scaled.coefficients.iter()) {
            assert_eq!(exp_row.get_coefficients(), scaled_row.get_coefficients());
        }

        // y 방향 스케일링
        let y_scaled = poly.scale(&ScalarField::from_u32(1), &ScalarField::from_u32(2));
        let expected_y = BivariatePolynomial::new(vec![
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1)
            ],
            vec![
                ScalarField::from_u32(2),
                ScalarField::from_u32(2),
                ScalarField::from_u32(2)
            ],
            vec![
                ScalarField::from_u32(4),
                ScalarField::from_u32(4),
                ScalarField::from_u32(4)
            ]
        ]);

        assert_eq!(expected_y.coefficients.len(), y_scaled.coefficients.len());
        for (exp_row, scaled_row) in expected_y.coefficients.iter().zip(y_scaled.coefficients.iter()) {
            assert_eq!(exp_row.get_coefficients(), scaled_row.get_coefficients());
        }
    }
}