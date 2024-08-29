use core::ops::{Add,Sub};
use core::ops;

use crate::alloc::borrow::ToOwned;



use alloc::sync::Arc;
use lambdaworks_math::field::element::FieldElement; 
use lambdaworks_math::field::traits::{IsField,IsSubFieldOf};

use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;




/// Represents the polynomial (c_00 + c_01 * X + c_02 * X^2 + ... + c_0n * X^n) * Y^0 + 
///                           (c_10 + c_11 * X + c_12 * X^2 + ... + c_1n * X^n) * Y^1 + ... + 
///                           (c_n0 + c_n1 * X + c_n2 * X^2 + ... + c_nn * X^n) * Y^n 
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BivariatePolynomial<FE> {
    pub coefficients: alloc::vec::Vec<alloc::vec::Vec<FE>>, // ndarray , 1D 
    pub x_degree: usize,
    pub y_degree: usize, 
}


impl<F: IsField> BivariatePolynomial<FieldElement<F>> {
    /// Creates a new polynomial with the given coefficients
    pub fn new(coefficients: &[&[FieldElement<F>]]) -> Self {
        let mut max_x_length = 0;
        for y_row in coefficients.iter() {
            max_x_length = max_x_length.max(y_row.len());
        }
        // check for storing more efficiently 
        BivariatePolynomial {
            coefficients: coefficients.iter()
                .map(|inner| inner.to_vec())        
                .collect(),
            y_degree: coefficients.len(),
            x_degree: max_x_length,

        }
    }

    pub fn from_vec(coeffs: alloc::vec::Vec<alloc::vec::Vec<FieldElement<F>>>) -> Self {
        BivariatePolynomial {
            y_degree: coeffs.len(),
            x_degree: coeffs[0].len(),
            coefficients: coeffs, 
        }
    }

    pub fn flatten_out(&self) -> alloc::vec::Vec<FieldElement<F>> {
        self.coefficients.iter().flat_map(|row| row.clone()).collect()
    }

    pub fn scale<S: IsSubFieldOf<F>>(&self, factor: &FieldElement<S>) -> Self {
        let scaled_coefficient: alloc::vec::Vec<alloc::vec::Vec<FieldElement<F>>> = self
            .coefficients
            .iter()
            .zip(core::iter::successors(Some(FieldElement::one()), |x| {
                Some(x * factor)
            }))
            .map(|(row,y_power)| {
                row
                    .iter()
                    .zip(core::iter::successors(Some(FieldElement::one()), |x| {
                        Some(x * factor)
                    }))
                    .map(|(coeff, power)| y_power.clone() * power * coeff)
                    .collect()
            })
            .collect();
        
        Self{
            coefficients: scaled_coefficient, 
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }

    // TODO ::  check on return type , I decided to scale the values in place
    pub fn scale_in_place<S: IsSubFieldOf<F>>(&mut self , factor: &FieldElement<S>) { 
        let iter = self.coefficients.iter_mut();
        let mut x = FieldElement::one();
        for dd in iter {
            dd
            .iter_mut()
            .zip(core::iter::successors(Some(FieldElement::one()), |y| Some(y * factor)))
            .for_each(|(coef, power)| *coef = coef.clone() * power.to_extension() * x.clone());


            x = x * factor.clone().to_extension();
        }

    }

    //TODO write ops overloading for it 
    pub fn sub_by_field_element(&self, element: &FieldElement<F>) -> BivariatePolynomial<FieldElement<F>> {
        let mut new_coefficients = self.coefficients.clone();

        new_coefficients[0][0] =  new_coefficients[0][0].clone() - element;
        
        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }


    pub fn zero() -> Self {
        Self::new(&[])
    }
    
    //O(log k) for x^k-1 and it is for only vanishing polynomial TODO.
    
    // check Horners method implementation.
    pub fn evaluate<E>(&self, x: &FieldElement<E>, y: &FieldElement<E>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {   

        self.coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |y_acc: FieldElement<E>, y_row | {
                let y_coeff = y_row
                    .iter()
                    .rev()
                    .fold(FieldElement::zero(), |x_acc: FieldElement<E>, x_coeff| {
                        x_coeff + x_acc * x.to_owned()
                    }); 
                y_coeff+ y_acc * y.to_owned()
                })
    }

    // P(x,y) = (x-a) Q(x,y) + (y-b)Z(y)
    /// Computes quotients with `x - a` and then `y - b`  in place.
    pub fn ruffini_division<L>(&self, a: &FieldElement<L>, b: &FieldElement<L>) -> (BivariatePolynomial<FieldElement<L>>, UnivariatePolynomial<FieldElement<L>>)
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        let (q_xy, remainder_y) = self.coefficients
                    .iter()
                    .fold((BivariatePolynomial::zero(), UnivariatePolynomial::zero()), |mut poly_acc_tuple, y_row| {
                        if let Some(c) = y_row.last() {
                            let mut c = c.clone().to_extension();
                            let mut coefficients: alloc::vec::Vec<FieldElement<L>> = alloc::vec::Vec::with_capacity(self.x_degree);
                            for coeff in y_row.iter().rev().skip(1) {
                                coefficients.push(c.clone());
                                c = coeff + c * a;
                            }// after the loop c is reminder 
                            // Add Q(x,y) to the coefficients of accumulator poly
                            poly_acc_tuple.0.x_degree = poly_acc_tuple.0.x_degree.max(coefficients.len());
                            poly_acc_tuple.0.y_degree+=1 ;

                            coefficients = coefficients.into_iter().rev().collect();
                            poly_acc_tuple.0.coefficients.push(coefficients);
                            
                            // add reminder poly to Qy(y) 
                            let remainder_poly = UnivariatePolynomial::new_monomial(c, poly_acc_tuple.0.coefficients.len()-1);
                            poly_acc_tuple.1 = poly_acc_tuple.1.add(remainder_poly);

                            poly_acc_tuple


                        } else {
                            // if there was not sth in the row continue. maybe I have to handle error here 
                            poly_acc_tuple
                        }  
                    });

        let q_y = remainder_y.ruffini_division(b);

        (q_xy,q_y)
    }
    

}
 


// /* Substraction field element at right */
// impl<F, L> ops::std::Sub<&BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
// where
//     L: IsField,
//     F: IsSubFieldOf<L>,
// {
//     type Output = BivariatePolynomial<FieldElement<L>>;

//     fn sub(self, other: &BivariatePolynomial<FieldElement<L>>) -> BivariatePolynomial<FieldElement<L>> {
//         BivariatePolynomial::new_monomial(self.clone(), 0) - other
//     }
// }







//Owned FieldElement minus Borrowed BivariatePolynomial
impl<F, L> ops::Sub<&BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, poly: &BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Create a new vector for the result coefficients
        let mut new_coefficients = poly.coefficients.clone();

        // Subtract the scalar from the constant term of the polynomial
        if !new_coefficients.is_empty() && !new_coefficients[0].is_empty() {
            new_coefficients[0][0] = self.to_extension() - poly.coefficients[0][0].clone();
        } else {
            new_coefficients.push(vec![self.to_extension()]);
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: poly.x_degree,
            y_degree: poly.y_degree,
        }
    }
}





// Owned FieldElement minus Owned BivariatePolynomial
impl<F, L> ops::Sub<BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Subtract the scalar from the constant term of the polynomial
        if !poly.coefficients.is_empty() && !poly.coefficients[0].is_empty() {
            poly.coefficients[0][0] = self.to_extension() - poly.coefficients[0][0].clone();
        } else {
            poly.coefficients.push(vec![self.to_extension()]);
        }

        poly
    }
}






//Borrowed FieldElement minus Owned BivariatePolynomial
impl<F, L> ops::Sub<BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Subtract the scalar from the constant term of the polynomial
        if !poly.coefficients.is_empty() && !poly.coefficients[0].is_empty() {
            poly.coefficients[0][0] = self.clone().to_extension() - poly.coefficients[0][0].clone();
        } else {
            poly.coefficients.push(vec![self.clone().to_extension()]);
        }

        poly
    }
}


// Borrowed FieldElement plus Borrowed BivariatePolynomial
impl<F, L> Add<&BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn add(self, other: &BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        let mut new_coefficients = other.coefficients.clone();

        // Add the FieldElement to the constant term
        if !new_coefficients.is_empty() && !new_coefficients[0].is_empty() {
            new_coefficients[0][0] = new_coefficients[0][0].clone() + self.clone().to_extension();
        } else {
            // If the polynomial has no constant term, we effectively add the FieldElement as the constant term
            new_coefficients = vec![vec![self.clone().to_extension()]];
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: other.x_degree,
            y_degree: other.y_degree,
        }
    }
}

// Owned FieldElement plus Borrowed BivariatePolynomial
impl<F, L> Add<&BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn add(self, poly: &BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Create a new vector for the result coefficients
        let mut new_coefficients = poly.coefficients.clone();

        // Add the scalar to the constant term of the polynomial
        if !new_coefficients.is_empty() && !new_coefficients[0].is_empty() {
            new_coefficients[0][0] = new_coefficients[0][0].clone() + self.to_extension();
        } else {
            new_coefficients.push(vec![self.to_extension()]);
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: poly.x_degree,
            y_degree: poly.y_degree,
        }
    }
}

// Owned FieldElement plus Owned BivariatePolynomial
impl<F, L> Add<BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn add(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Add the scalar to the constant term of the polynomial
        if !poly.coefficients.is_empty() && !poly.coefficients[0].is_empty() {
            poly.coefficients[0][0] = poly.coefficients[0][0].clone() + self.to_extension();
        } else {
            poly.coefficients.push(vec![self.to_extension()]);
        }

        poly
    }
}

// Borrowed FieldElement plus Owned BivariatePolynomial
impl<F, L> Add<BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn add(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Add the scalar to the constant term of the polynomial
        if !poly.coefficients.is_empty() && !poly.coefficients[0].is_empty() {
            poly.coefficients[0][0] = poly.coefficients[0][0].clone() + self.clone().to_extension();
        } else {
            poly.coefficients.push(vec![self.clone().to_extension()]);
        }

        poly
    }
}












#[cfg(test)]
mod tests {
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

        // Some of these tests work when the finite field has order greater than 2.
    use super::*;

    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    // 3 + x + 2x*y + x^2*y + 4x*y^2 
    // because we lexicography order is based on y and x the vector should represent like this
    // ( 3 + 1 + 0 ) , ( 0 , 2 , 1) , (0 , 4 , 0) 
    fn polynomial_a() -> BivariatePolynomial<FE> {
        BivariatePolynomial::new(&[
            &[FE::new(3), FE::new(1), FE::new(0)],
            &[FE::new(0), FE::new(2), FE::new(1)],
            &[FE::new(0), FE::new(4), FE::new(0)],
        ])
    }
    #[test]
    fn evaluation_a_2_3(){
        // 2*2*3 + 2^2*3 + 4*3^2*2 + 3 + 2 = 12+12+72+3+2
        // 5 + 8*3 + 8*9 = 5+24+72
        let eval = polynomial_a().evaluate(&FE::new(2), &FE::new(3));
        assert_eq!(FE::new(9),eval);
    }

    // test ruffini implementation 
    // Q = (x-1)[3 + x + 2xy + x^2*y + 4xy^2] + (y-2)[ y^2 + 3 ]
    // Q.ruffinit(1,2) => [3+x+2xy+x^2y+4xy^2] , [y^2+3]
    // bear in mind we test it in Z_23 => -9 => 14 
    #[test]
    fn ruffini_test(){
        let p = BivariatePolynomial::new(&[
            &[FE::new(14), FE::new(2),FE::new(1),FE::zero()],
            &[FE::new(3), FE::new(21),FE::new(1),FE::new(1)],
            &[FE::new(21), FE::new(19),FE::new(4),FE::new(0)],
            &[FE::new(1), FE::zero(),FE::zero(),FE::new(0)],
        ]);
        let dd = p.evaluate(&FE::new(1), &FE::new(2));
        assert_eq!(FE::zero(),p.evaluate(&FE::one(), &FE::new(2)));
        let (q_xy, q_y) = p.ruffini_division(&FE::new(1),&FE::new(2));
        
        let p_xy = BivariatePolynomial::new(&[
            &[FE::new(3), FE::new(1),FE::new(0)],
            &[FE::new(0), FE::new(2),FE::new(1)],
            &[FE::new(0), FE::new(4),FE::new(0)],
            &[FE::new(0), FE::zero(),FE::zero()],
        ]);

        assert_eq!(p_xy,q_xy);
        
        

    }
}