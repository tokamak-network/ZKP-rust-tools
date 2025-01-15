use icicle_core::polynomials::UnivariatePolynomial;
use icicle_bls12_377::polynomials::DensePolynomial;
use icicle_bls12_377::curve::ScalarField;
use icicle_runtime::memory::HostSlice;
use icicle_core::traits::FieldImpl;
use std::ops::Add;

// #[derive(Debug, Clone, PartialEq, Eq)]
pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>, // 고정된 DensePolynomial 타입
    pub x_degree: usize,                    // 최대 x 차수
    pub y_degree: usize,                    // 최대 y 차수
}

impl BivariatePolynomial {
    /// 새로운 이변량 다항식을 생성합니다.
    pub fn new(coefficients: Vec<Vec<ScalarField>>) -> Self {
        let y_degree = coefficients.len();
        let x_degree = coefficients
            .iter()
            .map(|row| row.len())
            .max()
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

    /// 영 이변량 다항식을 생성합니다.
    pub fn zero() -> Self {
        let zero_coeffs = vec![ScalarField::zero()];
        let host_slice = HostSlice::from_slice(&zero_coeffs[..]);
        BivariatePolynomial {
            coefficients: vec![DensePolynomial::from_coeffs(host_slice, zero_coeffs.len())],
            x_degree: 0,
            y_degree: 0,
        }
    }

    /// 계수를 단일 벡터로 평탄화합니다.
    pub fn flatten_out(&self) -> Vec<ScalarField> {
        self.coefficients.iter().flat_map(|poly| poly.get_coefficients()).collect()
    }

    /// 다항식을 x_factor 및 y_factor로 스케일링합니다.
    pub fn scale(&self, x_factor: &ScalarField, y_factor: &ScalarField) -> Self {
        let scaled_coefficients: Vec<DensePolynomial> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(y, poly)| {
                let y_power = pow_field(y_factor, y as usize);
                let scaled_poly = poly.scale(x_factor);
                // 각 계수를 y_power로 스케일링
                let scaled_coeffs: Vec<ScalarField> = scaled_poly.get_coefficients()
                    .iter()
                    .map(|c| *c * y_power) // 연산자 오버로딩 사용
                    .collect();
                let host_slice = HostSlice::from_slice(&scaled_coeffs[..]);
                DensePolynomial::from_coeffs(host_slice, scaled_coeffs.len())
            })
            .collect();

        BivariatePolynomial {
            coefficients: scaled_coefficients,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }

    /// 상수 항에서 필드 원소를 뺍니다.
    pub fn sub_by_field_element(&mut self, element: &ScalarField) {
        if let Some(first_poly) = self.coefficients.get_mut(0) {
            first_poly.sub_constant(*element);
        } else {
            // 계수가 없는 경우, 음수 원소로 초기화
            let zero = ScalarField::zero();
            let neg_element = zero - *element; // 연산자 오버로딩 사용
            let neg_coeffs = vec![neg_element];
            let host_slice = HostSlice::from_slice(&neg_coeffs[..]);
            self.coefficients.push(DensePolynomial::from_coeffs(host_slice, neg_coeffs.len()));
        }
    }

    /// 다항식을 주어진 x, y 값에서 평가합니다.
    pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField {
        let mut result = ScalarField::zero();
        for (i, poly) in self.coefficients.iter().enumerate().rev() {
            let y_power = pow_field(y, i as usize);
            let poly_eval = poly.eval(x);
            let term = poly_eval * y_power; // 연산자 오버로딩 사용
            result = result + term;           // 연산자 오버로딩 사용
        }
        result
    }

    fn dense_poly_new_monomial(coeff: ScalarField, degree: usize) -> DensePolynomial {
        let mut coeffs = vec![ScalarField::zero(); degree + 1];
        coeffs[degree] = coeff;
        DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs), degree + 1)
    }

    pub fn ruffini_division(
        &self, 
        a: &ScalarField, 
        b: &ScalarField
    ) -> (
        BivariatePolynomial,
        DensePolynomial
    ) {
        println!("a, b: {:?}, {:?}", a, b);
        
        // Step 1: (x - a) 다항식 생성
        let zero = ScalarField::zero();
        let neg_a = zero - *a; // (x - a)에서 a의 음수
        let divisor_coeffs = vec![neg_a, ScalarField::one()]; // (x - a)의 계수: -a + 1*x
        let host_divisor = HostSlice::from_slice(&divisor_coeffs[..]);
        let divisor = DensePolynomial::from_coeffs(host_divisor, divisor_coeffs.len());

        let mut q_xy_coeffs = Vec::with_capacity(self.y_degree);
        let mut remainders = Vec::with_capacity(self.y_degree);

        // Step 2: 각 y-고차항에 대해 (x - a)로 나눕니다.
        for poly in &self.coefficients {
            // 다항식 나눗셈 수행: poly / (x - a)
            let (q, r) = poly.divide(&divisor);
            q_xy_coeffs.push(q);
            remainders.push(r.get_constant()); // 나머지의 상수항만 수집
        }
        println!("remainders: {:?}", remainders);

        // Step 3: 잔여항을 R(y)로 조합
        // R(y) = sum (remainders[i] * y^i)
        let mut remainder_y = DensePolynomial::from_coeffs(HostSlice::from_slice(&[]), self.y_degree);
        for (i, r) in remainders.iter().enumerate() {
            let remainder_poly = Self::dense_poly_new_monomial(r.clone(), i);
            remainder_y = remainder_y.add(&remainder_poly);
        }

        // println!("R(y): {:?}", remainder_y);

        // Step 4: q_xy의 x_degree 업데이트 (1 감소)
        let new_x_degree = if self.x_degree > 0 { self.x_degree - 1 } else { 0 };

        // Step 5: 모든 q_xy_coeffs의 계수 길이를 동일하게 맞추기 (0으로 패딩)
        let expected_coeff_len = new_x_degree + 1; // x_degree=4이면 계수 길이=5
        for q in q_xy_coeffs.iter_mut() {
            let mut coeffs = q.get_coefficients();
            while coeffs.len() < expected_coeff_len {
                coeffs.push(ScalarField::zero());
            }
            // 계수 길이를 초과하지 않도록 잘라냄
            coeffs.truncate(expected_coeff_len);
            *q = DensePolynomial::from_coeffs(HostSlice::from_slice(&coeffs[..]), coeffs.len());
        }

        // 모든 q_xy_coeffs 출력 (디버깅 용도)
        q_xy_coeffs.iter().for_each(|poly: &DensePolynomial| {
            poly.print()
        });

        // Step 6: 새로운 BivariatePolynomial 생성
        let q_xy = BivariatePolynomial {
            coefficients: q_xy_coeffs,
            x_degree: new_x_degree,
            y_degree: self.y_degree,
        };
        
        // println!("R(y): {:?}", remainder_y);
        
        // Step 7: (y - b)로 나누기 위한 나머지 계산
        let (q_y, final_remainder) = remainder_y.ruffini_division(b)?;
        // println!("final_remainder: {:?}", final_remainder);
        
        (q_xy, final_remainder)
    }
}

/// `DensePolynomial`에 추가 기능을 제공하는 확장 트레이트
pub trait DensePolynomialExt {
    /// 모든 계수를 벡터로 반환합니다.
    fn get_coefficients(&self) -> Vec<ScalarField>;

    /// 다항식을 스칼라 값으로 곱합니다.
    fn scale(&self, scalar: &ScalarField) -> Self;

    /// 다항식의 상수 항에서 주어진 값을 뺍니다.
    fn sub_constant(&mut self, constant: ScalarField);

    /// 다항식의 상수 항을 반환합니다.
    fn get_constant(&self) -> ScalarField;
}

impl DensePolynomialExt for DensePolynomial {
    fn get_coefficients(&self) -> Vec<ScalarField> {
        let nof_coeffs = self.get_nof_coeffs();
        let mut coeffs = vec![ScalarField::zero(); nof_coeffs as usize];
        self.copy_coeffs(0, HostSlice::from_mut_slice(&mut coeffs));
        coeffs
    }

    fn scale(&self, scalar: &ScalarField) -> Self {
        let scaled_coeffs: Vec<ScalarField> = self.get_coefficients().iter()
            .map(|c| *c * *scalar)
            .collect();
        let host_slice = HostSlice::from_slice(&scaled_coeffs[..]);
        DensePolynomial::from_coeffs(host_slice, scaled_coeffs.len())
    }

    fn sub_constant(&mut self, constant: ScalarField) {
        let coeff = self.get_coeff(0);
        let new_coeff = coeff - constant; // 연산자 오버로딩 사용
        self.copy_coeffs(0, HostSlice::from_mut_slice(&mut [new_coeff]));
    }

    fn get_constant(&self) -> ScalarField {
        self.get_coeff(0)
    }
}

/// 필드 연산을 위한 헬퍼 함수 (이진 지수법)
fn pow_field(element: &ScalarField, exponent: usize) -> ScalarField {
    let mut result = ScalarField::one();
    let mut base = element.clone();
    let mut exp = exponent;

    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base.clone(); // 연산자 오버로딩 사용
        }
        base = base.clone() * base.clone(); // 연산자 오버로딩 사용
        exp /= 2;
    }

    result
}

/// 단위 테스트
#[cfg(test)]
mod tests {
    use super::*;
    use icicle_bls12_377::curve::ScalarField;

    /// 간단한 필드 원소 생성
    fn create_field_elements() -> (ScalarField, ScalarField, ScalarField, ScalarField) {
        let zero = ScalarField::zero();
        let one = ScalarField::one();
        let two = one.clone() + one.clone();
        let three = two.clone() + one.clone();
        (zero, one, two, three)
    }

    #[test]
    fn ruffini_test() {
        let (_, one, two, _) = create_field_elements();
        let poly = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(14), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::from_u32(0),],
            vec![ScalarField::from_u32(3), ScalarField::from_u32(21), ScalarField::from_u32(1), ScalarField::from_u32(1),],
            vec![ScalarField::from_u32(21), ScalarField::from_u32(19), ScalarField::from_u32(4), ScalarField::from_u32(0),],
            vec![ScalarField::from_u32(1), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0),],
        ]);

        let (q_xy, q_y) = poly.ruffini_division(&two, &one);
        q_xy.coefficients.iter().for_each(|poly: &DensePolynomial| {
            poly.print()
        });
        println!("q_xy: {:?}", q_xy.x_degree);
        let expected_q_xy = BivariatePolynomial::new(vec![
            vec![ScalarField::from_u32(3), ScalarField::from_u32(1), ScalarField::from_u32(0), ScalarField::from_u32(0),],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(2), ScalarField::from_u32(1), ScalarField::from_u32(0),],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(4), ScalarField::from_u32(0), ScalarField::from_u32(0),],
            vec![ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0), ScalarField::from_u32(0),],
        ]);
        let expected_q_y = vec![ScalarField::from_u32(3), ScalarField::from_u32(0), ScalarField::from_u32(1)];
        // println!("expected_q_y: {:?}", expected_q_y);
        assert_eq!(q_xy.x_degree, expected_q_xy.x_degree);
        assert_eq!(q_xy.y_degree, expected_q_xy.y_degree);
        // assert_eq!(q_y, expected_q_y, "The remainder polynomial is incorrect.");        
    }
}