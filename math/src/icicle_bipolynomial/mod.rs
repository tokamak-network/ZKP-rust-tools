use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_bls12_381::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::traits::FieldImpl;
use alloc::vec::Vec;
use alloc::vec;

/// DensePolynomial에 대한 확장 트레이트
pub trait DensePolynomialExt {
    fn get_coefficients(&self) -> Vec<ScalarField>;
    fn scale(&self, scalar: &ScalarField) -> Self;
    fn sub_constant(&mut self, constant: ScalarField);
    fn get_constant(&self) -> ScalarField;
    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str>;
    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial;
}

impl DensePolynomialExt for DensePolynomial {
    fn get_coefficients(&self) -> Vec<ScalarField> {
        let n = self.get_nof_coeffs();
        let mut coeffs = vec![ScalarField::zero(); n as usize];
        self.copy_coeffs(0, HostSlice::from_mut_slice(&mut coeffs));
        coeffs
    }

    fn scale(&self, scalar: &ScalarField) -> Self {
        // 각 계수에 scalar 곱
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
        // 맨 앞 coeffs[0] -= constant
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
        // 맨 앞이 상수항
        self.get_coeff(0)
    }

    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str> {
        // (x - b) 꼴에서 b가 루트 역할
        let n = self.get_nof_coeffs();
        if n == 0 {
            return Err("Polynomial has no coefficients.");
        }

        let coeffs = self.get_coefficients(); 
        // 뒤에서부터 몫을 채움 (synthetic division)
        let mut iter = coeffs.iter().rev();
        let mut temp = *iter.next().unwrap(); 
        let mut q = Vec::with_capacity(n as usize - 1);

        for &c in iter {
            q.push(temp);
            temp = temp * *b + c;  // temp = temp*b + c
        }
        q.reverse(); 

        // 몫에서 trailing zero 제거
        let mut q_poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&q),
            q.len()
        );
        let mut q_cf = q_poly.get_coefficients();
        while q_cf.len() > 1 && *q_cf.last().unwrap() == ScalarField::zero() {
            q_cf.pop();
        }
        q_poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&q_cf),
            q_cf.len()
        );

        // temp 가 나머지
        Ok((q_poly, temp))
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

/// 이변량 다항식
pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>, // row i => y^i
    pub x_degree: usize,
    pub y_degree: usize,
}

impl BivariatePolynomial {
    pub fn new(rows: Vec<Vec<ScalarField>>) -> Self {
        // 행 개수 = y_degree+1 임을 고려
        let y_degree = rows.len().saturating_sub(1);
    
        let x_degree = rows
            .iter()
            .map(|r| r.len())
            .max()
            .map(|len| len.saturating_sub(1))
            .unwrap_or(0);
    
        let polys = rows.into_iter().map(|row| {
            let slice = HostSlice::from_slice(&row);
            DensePolynomial::from_coeffs(slice, row.len())
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
        // degree+1 길이, 해당 차수에 c
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

    pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField {
        // row i => y^i
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
        let total_len = (self.x_degree ) * (self.y_degree);
        let mut flattened = Vec::with_capacity(total_len);

        for row_index in 0..=self.y_degree {
            let row_poly = &self.coefficients[row_index];
            let row_coeffs = row_poly.get_coefficients(); 
            // univariate poly의 계수 (길이 <= x_degree+1 일 수 있음)

            for x_index in 0..=self.x_degree {
                let val = if x_index < row_coeffs.len() {
                    row_coeffs[x_index]
                } else {
                    ScalarField::zero()
                };
                flattened.push(val);
            }
        }

        flattened
    }

    pub fn ruffini_division(
        &self,
        a: &ScalarField,  // (x-a)
        b: &ScalarField,  // (y-b)
    ) -> Result<(BivariatePolynomial, DensePolynomial), &'static str> {
        // (1) (x - a) 로 y=0..y_degree-1 순서대로 나누기
        //     bipolynomial/mod.rs 쪽은 axis_iter(Axis(0)) 윗행부터 y_index=0 -> y^0 → y1, ...
        //     여기서도 i=0 => row0 (y^0), i=1 => row1 (y^1) 순으로.
        let mut q_xy_rows = Vec::with_capacity(self.y_degree);
        let mut remainders = Vec::with_capacity(self.y_degree);
    
        for (_y_index, poly) in self.coefficients.iter().enumerate() {
            // poly = row(y_index)
            let (mut q, r) = poly.ruffini_division(a)?;
    
            // Univariate 몫에서 trailing zero 제거
            let mut q_cf = q.get_coefficients();
            while q_cf.len() > 1 && q_cf.last().unwrap() == &ScalarField::zero() {
                q_cf.pop();
            }
            q = DensePolynomial::from_coeffs(
                HostSlice::from_slice(&q_cf),
                q_cf.len()
            );
    
            // row(y_index)의 몫
            q_xy_rows.push(q);
            // row(y_index)의 remainder
            remainders.push(r);
        }
    
        // (2) remainder_y = ∑_{i=0..} remainders[i]*y^i
        let mut remainder_y = Self::dense_poly_zero();
        for (i, &rem) in remainders.iter().enumerate() {
            // monomial = rem*x^0 with degree=i in y
            let monomial = Self::dense_poly_new_monomial(rem, i);
            remainder_y = remainder_y.add_polynomial(&monomial);
        }
    
        // (3) x_degree: bipolynomial/mod.rs 처럼 굳이 줄이지 않고 그대로
        //     혹은 필요 시 saturating_sub(1).
        let new_x_degree = self.x_degree;
    
        // (4) q_xy_rows[i] 의 coeff 길이를 (new_x_degree+1)로 맞추기
        for q_poly in q_xy_rows.iter_mut() {
            let mut cfs = q_poly.get_coefficients();
            while cfs.len() < new_x_degree + 1 {
                cfs.push(ScalarField::zero());
            }
            cfs.truncate(new_x_degree + 1);
            *q_poly = DensePolynomial::from_coeffs(
                HostSlice::from_slice(&cfs),
                cfs.len()
            );
        }
    
        let q_xy = BivariatePolynomial {
            coefficients: q_xy_rows,
            x_degree: new_x_degree,
            y_degree: self.y_degree,
        };
    
        // (5) remainder_y를 (y - b) 로 Ruffini
        let (mut q_y, _final_rem) = remainder_y.ruffini_division(b)?;
    
        // Univariate 몫 q_y 에서 trailing zero 제거
        let mut qy_cf = q_y.get_coefficients();
        while qy_cf.len() > 1 && qy_cf.last().unwrap() == &ScalarField::zero() {
            qy_cf.pop();
        }
        q_y = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&qy_cf),
            qy_cf.len()
        );
    
    
    
        Ok((q_xy, q_y))
    }
}

// fn pow_field(base: &ScalarField, exp: usize) -> ScalarField {
//     let mut r = ScalarField::one();
//     let mut cur = *base;
//     let mut e = exp;
//     while e > 0 {
//         if e & 1 == 1 {
//             r = r * cur;
//         }
//         cur = cur * cur;
//         e >>= 1;
//     }
//     r
// }

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_381::curve::ScalarField;

    /// 간단한 필드 원소 생성
    fn create_field_elements() -> (ScalarField, ScalarField, ScalarField, ScalarField) {
        let zero = ScalarField::zero();
        let one = ScalarField::one();
        let two = one + one;
        let three = two + one;
        (zero, one, two, three)
    }

    #[test]
    fn test_get_coefficients() {
        let (_, _, _, _) = create_field_elements();
        let v_coeffs = [ScalarField::from_u32(1), ScalarField::from_u32(2)];
        // 길이=2
        let poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&v_coeffs),
            v_coeffs.len()
        );

        let coefficients = poly.get_coefficients();
        assert_eq!(coefficients, vec![ScalarField::from_u32(1), ScalarField::from_u32(2)]);
    }

    #[test]
    fn test_scale() {
        let (_, _, two, _) = create_field_elements();
        let poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[ScalarField::from_u32(1), ScalarField::from_u32(2)]),
            2 // 계수 개수
        );
        let scaled_poly = poly.scale(&two);
        assert_eq!(
            scaled_poly.get_coefficients(),
            vec![ScalarField::from_u32(2), ScalarField::from_u32(4)]
        );
    }

    #[test]
    fn test_sub_constant() {
        let (_, _, _, three) = create_field_elements();
        let v_coeffs = [ScalarField::from_u32(5)];
        // 길이=1
        let mut poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&v_coeffs),
            v_coeffs.len()
        );
        poly.sub_constant(three);
        assert_eq!(poly.get_coefficients(), vec![ScalarField::from_u32(2)]);
    }

    #[test]
    fn test_get_constant() {
        let (_, _, _, three) = create_field_elements();
        // 길이=1
        let poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[three]),
            1
        );
        assert_eq!(poly.get_constant(), three);
    }

    #[test]
    fn test_ruffini_division_dense() {
        let (_, one, _, _) = create_field_elements();
        // 계수 = [2, 1, 1] → 길이=3
        let poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[
                ScalarField::from_u32(2),
                ScalarField::from_u32(1),
                ScalarField::from_u32(1),
            ]),
            3
        );

        let (quotient, remainder) = poly.ruffini_division(&one).expect("Ruffini division failed");

        // 기대 몫 = [2, 1], 기대 나머지 = 4
        let expected_quotient = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[ScalarField::from_u32(2), ScalarField::from_u32(1)]),
            2
        );
        let expected_remainder = ScalarField::from_u32(4);

        assert_eq!(quotient.get_coefficients(), expected_quotient.get_coefficients());
        assert_eq!(remainder, expected_remainder);
    }

    #[test]
    fn test_add_polynomial() {
        let (_, _, _, _) = create_field_elements();
        let poly1 = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[ScalarField::from_u32(1), ScalarField::from_u32(2)]),
            2
        );
        let poly2 = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&[ScalarField::from_u32(3), ScalarField::from_u32(4)]),
            2
        );

        let sum_poly = poly1.add_polynomial(&poly2);
        assert_eq!(
            sum_poly.get_coefficients(),
            vec![ScalarField::from_u32(4), ScalarField::from_u32(6)]
        );
    }

    #[test]
    fn test_bivariate_polynomial_evaluate() {
        let (_, one, two, three) = create_field_elements();
        // y^0: 1 + 2x, y^1: 3 + x
        let poly = BivariatePolynomial::new(vec![
            vec![one, two],   // (상수항=1, x계수=2)
            vec![three, one], // (상수항=3, x계수=1)
        ]);

     
        let result = poly.evaluate(&two, &one);

      
        assert_eq!(result, ScalarField::from_u32(10));
    }

    #[test]
    fn test_bivariate_polynomial_ruffini_division() {
        let (zero, one, two, _) = create_field_elements();
        // y=0 행: [14, 2, 1], y=1 행: [3, 21, 1]
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
                ScalarField::from_u32(1),
            ],
            vec![
                ScalarField::from_u32(21),
                ScalarField::from_u32(19),
                ScalarField::from_u32(4),
                ScalarField::from_u32(0),
            ],
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(0),
                ScalarField::from_u32(0),
                ScalarField::from_u32(0),
            ],
        ]);

        let (q_xy, q_y) = poly.ruffini_division(&one, &two).expect("Ruffini division failed");

        // 테스트에서 기대하는 몫 & 나머지
        let expected_q_xy = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0), ScalarField::from_u32(0)],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0)],
        ]);
        
        let vector = vec![
            ScalarField::from_u32(3), 
            ScalarField::from_u32(0),
            ScalarField::from_u32(1),
        ];

        let expected_q_y = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&vector[..]), 
            vector.len()
        );

        q_xy.coefficients.iter().zip(expected_q_xy.coefficients.iter()).for_each(|(q, expected_q)| {
            assert_eq!(q.get_coefficients(), expected_q.get_coefficients());
        });
        // 단순 체크
        assert_eq!(q_xy.coefficients.len(), expected_q_xy.coefficients.len());
        assert_eq!(q_y.get_coefficients(), expected_q_y.get_coefficients());
    }

    #[test]
    fn test_bivariate_poly_flatten_out() {
        let row0 = vec![
            ScalarField::from_u32(1),
            ScalarField::from_u32(2),
        ];
        let row1 = vec![
            ScalarField::from_u32(3),
            ScalarField::from_u32(4),
        ];
        let poly = BivariatePolynomial::new(vec![row0.clone(), row1.clone()]);

        let flat = poly.flatten_out();
        assert_eq!(flat.len(), 4);

        // 순서: row_index=0 (y=0), x=0..1 → [1, 2], row_index=1 (y=1), x=0..1 → [3, 4]
        // 따라서 [1, 2, 3, 4]
        assert_eq!(
            flat,
            vec![
                ScalarField::from_u32(1),
                ScalarField::from_u32(2),
                ScalarField::from_u32(3),
                ScalarField::from_u32(4)
            ]
        );
    }
}
