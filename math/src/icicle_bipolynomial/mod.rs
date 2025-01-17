use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_377::polynomials::DensePolynomial;
use icicle_bls12_377::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::traits::FieldImpl;

/// `DensePolynomial`에 추가 기능을 제공하는 확장 트레이트
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
        let nof_coeffs = self.get_nof_coeffs();
        
        let mut coeffs = vec![ScalarField::zero(); nof_coeffs as usize];
        self.copy_coeffs(0, HostSlice::from_mut_slice(&mut coeffs));
        coeffs
    }

    fn scale(&self, scalar: &ScalarField) -> Self {
        let scaled_coeffs: Vec<ScalarField> = self
            .get_coefficients()
            .iter()
            .map(|c| *c * *scalar)
            .collect();

        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&scaled_coeffs[..]), 
            scaled_coeffs.len()
        )
    }

    fn sub_constant(&mut self, constant: ScalarField) {
        let mut coeffs = self.get_coefficients();
        if coeffs.is_empty() {
            panic!("Polynomial has no coefficients.");
        }
        coeffs[0] = coeffs[0] - constant;
        *self = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&coeffs[..]),
            coeffs.len()
        );
    }

    fn get_constant(&self) -> ScalarField {
        self.get_coeff(0)
    }

    fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str> {
        if self.get_nof_coeffs() == 0 {
            return Err("Polynomial has no coefficients.");
        }

        // 1) 뒤에서부터 temp를 업데이트하며 몫 계수 생성
        let binding = self.get_coefficients();
        let mut coeffs_rev = binding.iter().rev();

        let mut temp = coeffs_rev.next().unwrap().clone();
        let mut quotient_coeffs = Vec::with_capacity(binding.len() - 1);

        for &coeff in coeffs_rev {
            quotient_coeffs.push(temp);
            temp = temp * *b + coeff;
        }
        quotient_coeffs.reverse();

        // 2) 몫 다항식 생성 후, **끝에 0 계수 있으면 제거**
        let mut q_poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&quotient_coeffs),
            quotient_coeffs.len(),
        );
        // trailing zero trim
        let mut q_coeffs_trim = q_poly.get_coefficients();
        while q_coeffs_trim.len() > 1 && q_coeffs_trim.last().unwrap() == &ScalarField::zero() {
            q_coeffs_trim.pop();
        }
        q_poly = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&q_coeffs_trim),
            q_coeffs_trim.len(),
        );

        // temp 가 나머지
        Ok((q_poly, temp))
    }

    fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial {
        let a_coeffs = self.get_coefficients();
        let b_coeffs = other.get_coefficients();
        let max_len = a_coeffs.len().max(b_coeffs.len());
        let mut result_coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = a_coeffs.get(i).cloned().unwrap_or_else(ScalarField::zero);
            let b = b_coeffs.get(i).cloned().unwrap_or_else(ScalarField::zero);
            result_coeffs.push(a + b);
        }
        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&result_coeffs[..]), 
            result_coeffs.len()
        )
    }
}

/// 이변량 다항식 구조체
pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>, // 각 y별 DensePolynomial (x에 대한)
    pub x_degree: usize,                   
    pub y_degree: usize,                   
}

impl BivariatePolynomial {
    pub fn new(coefficients: Vec<Vec<ScalarField>>) -> Self {
        let y_degree = coefficients.len();
        let x_degree = coefficients
            .iter()
            .map(|row| row.len())
            .max()
            .map(|len| len.saturating_sub(1))
            .unwrap_or(0);

        let dense_polys = coefficients
            .into_iter()
            .map(|row| {
                let host_slice = HostSlice::from_slice(&row[..]);
                DensePolynomial::from_coeffs(host_slice, row.len())
            })
            .collect();

        BivariatePolynomial {
            coefficients: dense_polys,
            x_degree,
            y_degree,
        }
    }

    pub fn zero() -> Self {
        let zero_coeffs = vec![ScalarField::zero()];
        let host_slice = HostSlice::from_slice(&zero_coeffs[..]);
        BivariatePolynomial {
            coefficients: vec![DensePolynomial::from_coeffs(host_slice, zero_coeffs.len())],
            x_degree: 0,
            y_degree: 0,
        }
    }

    pub fn flatten_out(&self) -> Vec<ScalarField> {
        self.coefficients
            .iter()
            .flat_map(|poly| poly.get_coefficients())
            .collect()
    }

    pub fn scale(&self, x_factor: &ScalarField, y_factor: &ScalarField) -> Self {
        let scaled_coefficients: Vec<DensePolynomial> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(y, poly)| {
                let y_power = pow_field(y_factor, y as usize);
                let scaled_poly = poly.scale(x_factor);
                let scaled_coeffs: Vec<ScalarField> = scaled_poly
                    .get_coefficients()
                    .iter()
                    .map(|c| *c * y_power)
                    .collect();

                DensePolynomial::from_coeffs(
                    HostSlice::from_slice(&scaled_coeffs[..]), 
                    scaled_coeffs.len()
                )
            })
            .collect();

        BivariatePolynomial {
            coefficients: scaled_coefficients,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }

    pub fn sub_by_field_element(&mut self, element: &ScalarField) {
        if let Some(first_poly) = self.coefficients.get_mut(0) {
            first_poly.sub_constant(*element);
        } else {
            let zero = ScalarField::zero();
            let neg_element = zero - *element;
            let neg_coeffs = vec![neg_element];
            let host_slice = HostSlice::from_slice(&neg_coeffs[..]);
            self.coefficients.push(DensePolynomial::from_coeffs(host_slice, neg_coeffs.len()));
        }
    }

    pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField {
        let mut result = ScalarField::zero();
        // 주의: 여기서는 coefficients[0]이 y^0, coefficients[1]이 y^1, ... 로 보고 있음
        for (i, poly) in self.coefficients.iter().enumerate() {
            let y_power = pow_field(y, i as usize);
            let poly_eval = poly.eval(x);
            let term = poly_eval * y_power;
            result = result + term;
        }
        result
    }

    fn dense_poly_new_monomial(coeff: ScalarField, degree: usize) -> DensePolynomial {
        let mut coeffs = vec![ScalarField::zero(); degree + 1];
        coeffs[degree] = coeff;
        DensePolynomial::from_coeffs(
            HostSlice::from_slice(&coeffs[..]), 
            coeffs.len()
        )
    }

    fn dense_poly_zero() -> DensePolynomial {
        let coeffs = vec![ScalarField::zero()];
        DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs[..]), coeffs.len())
    }

    pub fn ruffini_division(
        &self, 
        a: &ScalarField, 
        b: &ScalarField
    ) -> Result<(BivariatePolynomial, DensePolynomial), &'static str> {
        println!("Starting Ruffini Division with a = {:?}, b = {:?}", a, b);
    
        let mut q_xy_coeffs: Vec<DensePolynomial> = vec![Self::dense_poly_zero(); self.y_degree]; 
        // y_degree개 자리 미리 확보 (row i 몫이 들어갈 자리)
    
        let mut remainders = Vec::with_capacity(self.y_degree);
    
        // --- 변경점: 아래서부터 위로. 
        //     i = (y_degree - 1) down to 0
        for rev_i in 0..self.y_degree {
            let i = self.y_degree - 1 - rev_i;
            let poly = &self.coefficients[i];
    
            let (mut q, r) = poly.ruffini_division(a)?;
    
            // univariate 몫에서 trailing zeros trim
            let mut q_coeffs = q.get_coefficients();
            while q_coeffs.len() > 1 && q_coeffs.last().unwrap() == &ScalarField::zero() {
                q_coeffs.pop();
            }
            q = DensePolynomial::from_coeffs(
                HostSlice::from_slice(&q_coeffs),
                q_coeffs.len()
            );
    
            // q를 q_xy_coeffs[i] 자리에 넣는다. (기존 row와 동일한 i)
            q_xy_coeffs[i] = q;
            remainders.push(r);
        }
        println!("remainders (bottom->top): {:?}", remainders);
    
        // remainders는 현재 "아랫행부터" 들어가 있으므로, 뒤집어서 remainders[0]이 y^0
        remainders.reverse();
        // => 이제 remainders[0] = row0의 나머지, remainders[1] = row1 나머지 ...
    
        // (2) R(y) = sum_{i=0..} [ remainders[i] * y^i ]
        let mut remainder_y = Self::dense_poly_zero();
        for (i, &r) in remainders.iter().enumerate() {
            let remainder_poly = Self::dense_poly_new_monomial(r, i);
            remainder_y = remainder_y.add_polynomial(&remainder_poly);
        }
    
        // (3) x차수 1 감소 (테스트가 이를 기대한다면 유지)
        let new_x_degree = if self.x_degree > 0 {
            self.x_degree - 1
        } else {
            0
        };
    
        // (4) q_xy_coeffs 각 DensePolynomial을 new_x_degree+1 길이로 맞추기
        for q in q_xy_coeffs.iter_mut() {
            let mut coeffs = q.get_coefficients();
            while coeffs.len() < new_x_degree + 1 {
                coeffs.push(ScalarField::zero());
            }
            coeffs.truncate(new_x_degree + 1);
            *q = DensePolynomial::from_coeffs(
                HostSlice::from_slice(&coeffs),
                coeffs.len()
            );
        }
    
        let q_xy = BivariatePolynomial {
            coefficients: q_xy_coeffs,
            x_degree: new_x_degree,
            y_degree: self.y_degree,
        };
    
        // (5) remainder_y를 (y - b)로 Ruffini
        let (mut q_y, final_remainder) = remainder_y.ruffini_division(b)?;
    
        // univariate 몫 trimming
        let mut qy_coeffs = q_y.get_coefficients();
        while qy_coeffs.len() > 1 && qy_coeffs.last().unwrap() == &ScalarField::zero() {
            qy_coeffs.pop();
        }
        q_y = DensePolynomial::from_coeffs(
            HostSlice::from_slice(&qy_coeffs),
            qy_coeffs.len()
        );
    
        println!("final_remainder: {:?}", final_remainder);
    
        Ok((q_xy, q_y))
    }
}

/// 거듭제곱 계산
fn pow_field(base: &ScalarField, exponent: usize) -> ScalarField {
    let mut result = ScalarField::one();
    let mut cur = *base;
    let mut e = exponent;
    while e > 0 {
        if e % 2 == 1 {
            result = result * cur;
        }
        cur = cur * cur;
        e /= 2;
    }
    result
}


#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_377::curve::ScalarField;

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
}
