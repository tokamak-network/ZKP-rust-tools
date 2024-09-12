use crate::alloc::borrow::ToOwned;
use core::ops::{Add, Sub};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;
use ndarray::{s, Array, Array2, Axis, Ix2};

/// Represents the polynomial (c_00 + c_01 * X + c_02 * X^2 + ... + c_0n * X^n) * Y^0 +
///                           (c_10 + c_11 * X + c_12 * X^2 + ... + c_1n * X^n) * Y^1 + ... +
///                           (c_n0 + c_n1 * X + c_n2 * X^2 + ... + c_nn * X^n) * Y^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BivariatePolynomial<FE> {
    pub coefficients: Array2<FE>,
    pub x_degree: usize,
    pub y_degree: usize,
}

impl<F: IsField> BivariatePolynomial<FieldElement<F>> {
    /// Creates a new polynomial with the given coefficients
    /// Creates a new polynomial with the given coefficients in the form of an ndarray.
    pub fn new(coefficients: Array2<FieldElement<F>>) -> Self {
        let y_degree = coefficients.nrows();
        let x_degree = coefficients.ncols();

        BivariatePolynomial {
            coefficients,
            x_degree,
            y_degree,
        }
    }

    pub fn flatten_out(&self) -> alloc::vec::Vec<FieldElement<F>> {
        self.coefficients.iter().cloned().collect()
    }

    //TODO write ops overloading for it
    pub fn sub_by_field_element(
        &self,
        element: &FieldElement<F>,
    ) -> BivariatePolynomial<FieldElement<F>> {
        // Clone the coefficients array to create a new one for the result
        let mut new_coefficients = self.coefficients.clone();

        // Subtract the given element from the (0, 0) coefficient
        new_coefficients[(0, 0)] = new_coefficients[(0, 0)].clone() - element.clone();

        // Return the new BivariatePolynomial with the updated coefficients
        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: self.x_degree,
            y_degree: self.y_degree,
        }
    }

    ///Creates a zero polynomial with no coefficients.
    pub fn zero() -> Array2<FieldElement<F>> {
        Array2::<FieldElement<F>>::default((1, 2))
    }

    /// Evaluates the polynomial at the given x and y values
    pub fn evaluate<E>(&self, x: &FieldElement<E>, y: &FieldElement<E>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        // Iterate over the rows in reverse order
        let mut y_acc = FieldElement::zero();
        for y_row in self.coefficients.axis_iter(Axis(0)).rev() {
            let mut x_acc = FieldElement::zero();
            for x_coeff in y_row.iter().rev() {
                let x_coeff_as_e = x_coeff.clone(); // Convert FieldElement<F> to FieldElement<E>
                x_acc = x_coeff_as_e + x_acc * x.clone();
            }
            y_acc = x_acc + y_acc * y.clone();
        }
        y_acc
    }

    pub fn ruffini_division<L>(
        &self,
        a: &FieldElement<L>,
        b: &FieldElement<L>,
    ) -> (
        BivariatePolynomial<FieldElement<L>>,
        UnivariatePolynomial<FieldElement<L>>,
    )
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        // Initialize a 2D array with the appropriate size filled with zeros
        let mut q_xy_coeffs = Array2::<FieldElement<L>>::default((self.y_degree, self.x_degree));
        let mut remainder_y = UnivariatePolynomial::zero();

        for (y_index, y_row) in self.coefficients.axis_iter(Axis(0)).enumerate() {
            if let Some(c) = y_row.last() {
                // Convert the coefficient to the extension field L
                let mut c = c.clone().to_extension();
                let mut x_coeffs: Vec<FieldElement<L>> = Vec::with_capacity(self.x_degree);

                for coeff in y_row.iter().rev().skip(1) {
                    x_coeffs.push(c.clone());
                    c = coeff.clone().to_extension() + c * a;
                }

                // Reverse the coefficients to match the correct order
                x_coeffs.reverse();

                // Fill the q_xy_coeffs array with x_coeffs
                for (x_index, x_coeff) in x_coeffs.iter().enumerate() {
                    q_xy_coeffs[(y_index, x_index)] = x_coeff.clone();
                }

                // Create the remainder polynomial
                let remainder_poly = UnivariatePolynomial::new_monomial(c, y_index);
                remainder_y = remainder_y.add(remainder_poly);
            }
        }

        let q_xy = BivariatePolynomial {
            coefficients: q_xy_coeffs.clone(), // Clone here if you plan to use q_xy_coeffs later
            x_degree: q_xy_coeffs.ncols().max(0),
            y_degree: q_xy_coeffs.nrows().max(0),
        };

        // Perform Ruffini division on the univariate polynomial
        let q_y = remainder_y.ruffini_division(b);

        (q_xy, q_y)
    }
}

impl<F: IsField> Add for BivariatePolynomial<FieldElement<F>> {
    type Output = BivariatePolynomial<FieldElement<F>>;

    fn add(
        self,
        other: BivariatePolynomial<FieldElement<F>>,
    ) -> BivariatePolynomial<FieldElement<F>> {
        // Determine the maximum degrees in x and y directions
        let max_y_degree = self.y_degree.max(other.y_degree);
        let max_x_degree = self.x_degree.max(other.x_degree);

        // Resize self's coefficients to match the maximum degrees if necessary
        let mut self_extended_coeffs =
            Array2::<FieldElement<F>>::default((max_y_degree, max_x_degree));
        self_extended_coeffs
            .slice_mut(s![..self.y_degree, ..self.x_degree])
            .assign(&self.coefficients);

        // Resize other's coefficients to match the maximum degrees if necessary
        let mut other_extended_coeffs =
            Array2::<FieldElement<F>>::default((max_y_degree, max_x_degree));
        other_extended_coeffs
            .slice_mut(s![..other.y_degree, ..other.x_degree])
            .assign(&other.coefficients);

        // Perform element-wise addition of the coefficients
        let new_coefficients = self_extended_coeffs + other_extended_coeffs;

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: max_x_degree,
            y_degree: max_y_degree,
        }
    }
}

// Implementing the Add trait for references of BivariatePolynomial
impl<F: IsField> Add for &BivariatePolynomial<FieldElement<F>> {
    type Output = BivariatePolynomial<FieldElement<F>>;

    fn add(
        self,
        other: &BivariatePolynomial<FieldElement<F>>,
    ) -> BivariatePolynomial<FieldElement<F>> {
        let max_y_degree = self.y_degree.max(other.y_degree);
        let max_x_degree = self.x_degree.max(other.x_degree);

        // Create a new 2D array with the maximum dimensions
        let mut new_coefficients = Array2::<FieldElement<F>>::default((max_y_degree, max_x_degree));

        // Iterate over each coefficient and calculate the sum
        for y in 0..max_y_degree {
            for x in 0..max_x_degree {
                let self_coeff = if y < self.y_degree && x < self.x_degree {
                    self.coefficients[(y, x)].clone()
                } else {
                    FieldElement::zero()
                };

                let other_coeff = if y < other.y_degree && x < other.x_degree {
                    other.coefficients[(y, x)].clone()
                } else {
                    FieldElement::zero()
                };

                new_coefficients[(y, x)] = self_coeff + other_coeff;
            }
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: max_x_degree,
            y_degree: max_y_degree,
        }
    }
}

impl<F: IsField> Sub for BivariatePolynomial<FieldElement<F>> {
    type Output = BivariatePolynomial<FieldElement<F>>;

    fn sub(
        self,
        other: BivariatePolynomial<FieldElement<F>>,
    ) -> BivariatePolynomial<FieldElement<F>> {
        let max_y_degree = self.y_degree.max(other.y_degree);
        let max_x_degree = self.x_degree.max(other.x_degree);

        // Create a new 2D array to store the result coefficients
        let mut new_coefficients = Array2::<FieldElement<F>>::default((max_y_degree, max_x_degree));

        // Iterate over each coefficient and calculate the difference
        for y in 0..max_y_degree {
            for x in 0..max_x_degree {
                let self_coeff = if y < self.y_degree && x < self.x_degree {
                    self.coefficients[(y, x)].clone()
                } else {
                    FieldElement::zero()
                };

                let other_coeff = if y < other.y_degree && x < other.x_degree {
                    other.coefficients[(y, x)].clone()
                } else {
                    FieldElement::zero()
                };

                new_coefficients[(y, x)] = self_coeff - other_coeff;
            }
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: max_x_degree,
            y_degree: max_y_degree,
        }
    }
}

impl<F: IsField> Sub for &BivariatePolynomial<FieldElement<F>> {
    type Output = BivariatePolynomial<FieldElement<F>>;

    fn sub(
        self,
        other: &BivariatePolynomial<FieldElement<F>>,
    ) -> BivariatePolynomial<FieldElement<F>> {
        let max_y_degree = self.y_degree.max(other.y_degree);
        let max_x_degree = self.x_degree.max(other.x_degree);

        // Create a new Array2 for the result with the maximum size
        let mut new_coefficients = Array2::<FieldElement<F>>::default((max_y_degree, max_x_degree));

        for y in 0..max_y_degree {
            for x in 0..max_x_degree {
                let self_coeff = if y < self.coefficients.nrows() && x < self.coefficients.ncols() {
                    self.coefficients[(y, x)].clone()
                } else {
                    FieldElement::default()
                };

                let other_coeff =
                    if y < other.coefficients.nrows() && x < other.coefficients.ncols() {
                        other.coefficients[(y, x)].clone()
                    } else {
                        FieldElement::default()
                    };

                new_coefficients[(y, x)] = self_coeff - other_coeff;
            }
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: max_x_degree,
            y_degree: max_y_degree,
        }
    }
}

//Owned FieldElement minus Owned BivariatePolynomi
impl<F, L> core::ops::Sub<&BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(
        self,
        other: &BivariatePolynomial<FieldElement<L>>,
    ) -> BivariatePolynomial<FieldElement<L>> {
        // Clone the coefficients so that we can modify them
        let mut new_coefficients = other.coefficients.clone();

        // Subtract the FieldElement from the constant term (if it exists)
        if new_coefficients.nrows() > 0 && new_coefficients.ncols() > 0 {
            new_coefficients[(0, 0)] =
                self.to_owned().to_extension() - new_coefficients[(0, 0)].clone();
        } else {
            // If the polynomial has no constant term, we effectively add the FieldElement as the constant term
            let mut extended_coefficients = Array2::<FieldElement<L>>::default((1, 1));
            extended_coefficients[(0, 0)] = self.to_owned().to_extension();
            new_coefficients = extended_coefficients;
        }

        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: other.x_degree,
            y_degree: other.y_degree,
        }
    }
}

//Owned FieldElement minus Borrowed BivariatePolynomial
impl<F, L> Sub<&BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, poly: &BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Clone the coefficients to create a new set for the result
        let mut new_coefficients = poly.coefficients.clone();

        // Subtract the scalar from the constant term of the polynomial (if it exists)
        if new_coefficients.nrows() > 0 && new_coefficients.ncols() > 0 {
            new_coefficients[(0, 0)] = self.to_extension() - poly.coefficients[(0, 0)].clone();
        } else {
            // If the polynomial is effectively empty, create a new 1x1 array with the scalar as the constant term
            new_coefficients = Array2::from_elem((1, 1), self.to_extension());
        }

        // Return the new polynomial with updated coefficients
        BivariatePolynomial {
            coefficients: new_coefficients,
            x_degree: poly.x_degree,
            y_degree: poly.y_degree,
        }
    }
}

// Owned FieldElement minus Owned BivariatePolynomial
impl<F, L> Sub<BivariatePolynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Subtract the scalar from the constant term of the polynomial
        if poly.coefficients.nrows() > 0 && poly.coefficients.ncols() > 0 {
            poly.coefficients[(0, 0)] = self.to_extension() - poly.coefficients[(0, 0)].clone();
        } else {
            // If the polynomial has no terms, initialize it with the scalar as a constant term
            poly.coefficients = Array2::from_elem((1, 1), self.to_extension());
            poly.x_degree = 0;
            poly.y_degree = 0;
        }

        poly
    }
}

/// Implementing subtraction for a borrowed `FieldElement` minus an owned `BivariatePolynomial`
impl<F, L> Sub<BivariatePolynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = BivariatePolynomial<FieldElement<L>>;

    fn sub(self, mut poly: BivariatePolynomial<FieldElement<L>>) -> Self::Output {
        // Subtract the scalar from the constant term of the polynomial
        if poly.coefficients.nrows() > 0 && poly.coefficients.ncols() > 0 {
            // Subtract the scalar from the (0, 0) term
            poly.coefficients[(0, 0)] =
                self.clone().to_extension() - poly.coefficients[(0, 0)].clone();
        } else {
            // If the polynomial is empty, we initialize it with the scalar at (0, 0)
            poly.coefficients = Array2::from_elem((1, 1), self.clone().to_extension());
            poly.x_degree = 0;
            poly.y_degree = 0;
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
        if new_coefficients.nrows() > 0 && new_coefficients.ncols() > 0 {
            new_coefficients[(0, 0)] =
                new_coefficients[(0, 0)].clone() + self.clone().to_extension();
        } else {
            // If the polynomial has no constant term, initialize it with the FieldElement as the constant term
            new_coefficients = Array2::from_elem((1, 1), self.clone().to_extension());
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
        // Clone the coefficients to create a new set for the result
        let mut new_coefficients = poly.coefficients.clone();

        // Add the scalar to the constant term of the polynomial
        if new_coefficients.nrows() > 0 && new_coefficients.ncols() > 0 {
            new_coefficients[(0, 0)] = new_coefficients[(0, 0)].clone() + self.to_extension();
        } else {
            // If the polynomial is empty, initialize it with the scalar as the constant term
            new_coefficients = Array2::from_elem((1, 1), self.to_extension());
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
        if poly.coefficients.nrows() > 0 && poly.coefficients.ncols() > 0 {
            poly.coefficients[(0, 0)] = poly.coefficients[(0, 0)].clone() + self.to_extension();
        } else {
            // If the polynomial has no terms, initialize a 1x1 array with the FieldElement as the constant term
            poly.coefficients = Array2::from_elem((1, 1), self.to_extension());
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
        if poly.coefficients.nrows() > 0 && poly.coefficients.ncols() > 0 {
            poly.coefficients[(0, 0)] =
                poly.coefficients[(0, 0)].clone() + self.clone().to_extension();
        } else {
            // If the polynomial has no terms, initialize a 1x1 array with the FieldElement as the constant term
            poly.coefficients = Array2::from_elem((1, 1), self.clone().to_extension());
        }

        poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;
    use ndarray::array;

    // 3 + x + 2x*y + x^2*y + 4x*y^2
    // because we lexicography order is based on y and x the vector should represent like this
    // ( 3 + 1 + 0 ) , ( 0 , 2 , 1) , (0 , 4 , 0)
    fn polynomial_a() -> BivariatePolynomial<FE> {
        BivariatePolynomial::new(array![
            [FE::new(3), FE::new(1), FE::new(0)],
            [FE::new(0), FE::new(2), FE::new(1)],
            [FE::new(0), FE::new(4), FE::new(0)],
        ])
    }
    // 1 + 2x + 3y + 4xy
    fn polynomial_b() -> BivariatePolynomial<FE> {
        BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2), FE::new(0)],
            [FE::new(3), FE::new(4), FE::new(0)],
            [FE::new(0), FE::new(0), FE::new(0)],
        ])
    }

    #[test]
    fn test_bivariate_polynomial_new() {
        // Define the coefficients for the polynomial
        // Example: 3 + x + 2xy + x^2y + 4xy^2
        let coefficients = array![
            [FE::new(3), FE::new(1), FE::new(0)], // 3 + x
            [FE::new(0), FE::new(2), FE::new(1)], // 2xy + x^2y
            [FE::new(0), FE::new(4), FE::new(0)], // 4xy^2
        ];

        // Create the polynomial using the `new` method
        let poly = BivariatePolynomial::new(coefficients);

        // Expected 2D array of coefficients
        let expected_coeffs = array![
            [FE::new(3), FE::new(1), FE::zero()], // 3 + x + 0*x^2
            [FE::new(0), FE::new(2), FE::new(1)], // 0 + 2x + 1x^2
            [FE::new(0), FE::new(4), FE::zero()]  // 0 + 4x + 0*x^2
        ];

        // Verify the dimensions (degrees)
        assert_eq!(poly.x_degree, 3, "x_degree should be 3");
        assert_eq!(poly.y_degree, 3, "y_degree should be 3");

        // Verify the coefficients are as expected
        assert_eq!(
            poly.coefficients, expected_coeffs,
            "The coefficients matrix is incorrect."
        );
    }

    #[test]
    fn new_ndarray_2d_test() {
        let ploy_a = self::polynomial_a();

        assert_eq!(
            ploy_a,
            BivariatePolynomial::new(array![
                [FE::new(3), FE::new(1), FE::new(0)],
                [FE::new(0), FE::new(2), FE::new(1)],
                [FE::new(0), FE::new(4), FE::new(0)],
            ])
        )
    }

    #[test]
    fn test_flatten_out() {
        let coeffs = array![[FE::new(1), FE::new(2)], [FE::new(3), FE::new(4)]];

        let poly = BivariatePolynomial::new(coeffs);
        let flattened = poly.flatten_out();

        let expected = vec![FE::new(1), FE::new(2), FE::new(3), FE::new(4)];

        assert_eq!(flattened, expected);
    }

    #[test]
    fn test_sub_by_field_element() {
        let coeffs = array![[FE::new(5), FE::new(2)], [FE::new(3), FE::new(4)]];

        let poly = BivariatePolynomial::new(coeffs);
        let element_to_subtract = FE::new(3);

        let new_poly = poly.sub_by_field_element(&element_to_subtract);

        let expected_coeffs =
            Array::from_shape_vec((2, 2), vec![FE::new(2), FE::new(2), FE::new(3), FE::new(4)])
                .unwrap();

        assert_eq!(new_poly.coefficients, expected_coeffs);
    }

    #[test]
    fn test_evaluate() {
        let poly = polynomial_a();
        let x = FE::new(2);
        let y = FE::new(3);

        let result = poly.evaluate(&x, &y);

        // Manually compute the expected result
        // 3 + x + 2xy + x^2y + 4xy^2
        // = 3 + 2 + 2*2*3 + 2^2*3 + 4*2*3^2
        // = 3 + 2 + 12 + 12 + 72
        // = 101 mod 23
        let expected = FE::new(101 % ORDER);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero() {
        // Create a zero polynomial using the zero method
        let zero_poly = BivariatePolynomial::<FE>::zero();

        // Create an expected zero polynomial
        let expected_zero_poly = Array2::<FE>::default((1, 2));

        // Assert that the polynomial generated by the zero method matches the expected zero polynomial
        assert_eq!(zero_poly, expected_zero_poly);
    }

    // test ruffini implementation
    // Q = (x-1)[3 + x + 2xy + x^2*y + 4xy^2] + (y-2)[ y^2 + 3 ]
    // Q.ruffinit(1,2) => [3+x+2xy+x^2y+4xy^2] , [y^2+3]
    // bear in mind we test it in Z_23 => -9 => 14
    #[test]
    fn ruffini_test() {
        // Define the polynomial Q(x, y)
        let p = BivariatePolynomial::new(array![
            [FE::new(14), FE::new(2), FE::new(1), FE::zero()],
            [FE::new(3), FE::new(21), FE::new(1), FE::new(1)],
            [FE::new(21), FE::new(19), FE::new(4), FE::new(0)],
            [FE::new(1), FE::zero(), FE::zero(), FE::new(0)],
        ]);

        // Evaluate Q(1, 2) to ensure it's zero
        assert_eq!(FE::zero(), p.evaluate(&FE::new(1), &FE::new(2)));

        // Perform Ruffini division by (x - 1) and (y - 2)
        let (q_xy, q_y) = p.ruffini_division(&FE::new(1), &FE::new(2));

        // Define the expected quotient polynomial (3 + x + 2xy + x^2y + 4xy^2)
        let expected_q_xy = BivariatePolynomial::new(array![
            [FE::new(3), FE::new(1), FE::new(0), FE::new(0)],
            [FE::new(0), FE::new(2), FE::new(1), FE::new(0)],
            [FE::new(0), FE::new(4), FE::new(0), FE::new(0)],
            [FE::new(0), FE::zero(), FE::zero(), FE::new(0)],
        ]);

        // Define the expected remainder polynomial (y^2 + 3)
        let expected_q_y = UnivariatePolynomial::new(&[
            FE::new(3), // Constant term
            FE::new(0), // y term
            FE::new(1), // y^2 term
        ]);

        // Assert that the quotient and remainder are as expected
        assert_eq!(expected_q_xy, q_xy, "The quotient polynomial is incorrect.");
        assert_eq!(expected_q_y, q_y, "The remainder polynomial is incorrect.");
    }

    #[test]
    fn test_polynomial_addition_values() {
        // Polynomial p1: 1 + 2x + 3x^2 + 4y + 5xy + 6x^2y
        let p1 = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2), FE::new(3)],
            [FE::new(4), FE::new(5), FE::new(6)],
            [FE::new(4), FE::new(5), FE::new(6)],
        ]);

        // Polynomial p2: 6 + 5x + 4x^2 + 3y + 2xy + 1x^2y
        let p2 = BivariatePolynomial::new(array![
            [FE::new(6), FE::new(5), FE::new(4)],
            [FE::new(3), FE::new(2), FE::new(1)],
        ]);

        // Expected result: 7 + 7x + 7x^2 + 7y + 7xy + 7x^2y
        let expected = BivariatePolynomial::new(array![
            [FE::new(7), FE::new(7), FE::new(7)],
            [FE::new(7), FE::new(7), FE::new(7)],
            [FE::new(4), FE::new(5), FE::new(6)],
        ]);

        // Test the addition
        let result = p1 + p2;

        assert_eq!(expected, result);
    }

    #[test]
    fn test_polynomial_addition_references() {
        // Polynomial p1: 1 + 2x + 3x^2 + 4y + 5xy + 6x^2y
        let p1 = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2), FE::new(3)], // 1 + 2x + 3x^2
            [FE::new(4), FE::new(5), FE::new(6)], // 4y + 5xy + 6x^2y
            [FE::new(7), FE::new(8), FE::new(9)], // 7y^2 + 8xy^2 + 9x^2y^2
        ]);

        // Polynomial p2: 6 + 5x + 4x^2 + 3y + 2xy + 1x^2y
        let p2 = BivariatePolynomial::new(array![
            [FE::new(6), FE::new(5), FE::new(4)], // 6 + 5x + 4x^2
            [FE::new(3), FE::new(2), FE::new(1)], // 3y + 2xy + 1x^2y
            [FE::new(0), FE::new(0), FE::new(0)], // 0 + 0xy^2 + 0x^2y^2
        ]);

        // Expected result: p1 + p2
        // = (1 + 6) + (2 + 5)x + (3 + 4)x^2 + (4 + 3)y + (5 + 2)xy + (6 + 1)x^2y + (7 + 0)y^2 + (8 + 0)xy^2 + (9 + 0)x^2y^2
        let expected = BivariatePolynomial::new(array![
            [FE::new(7), FE::new(7), FE::new(7)], // 7 + 7x + 7x^2
            [FE::new(7), FE::new(7), FE::new(7)], // 7y + 7xy + 7x^2y
            [FE::new(7), FE::new(8), FE::new(9)], // 7y^2 + 8xy^2 + 9x^2y^2 (same as p1)
        ]);

        // Perform the addition
        let result = &p1 + &p2;

        // Assert that the result matches the expected polynomial
        assert_eq!(
            result, expected,
            "The polynomial addition result is incorrect."
        );
    }

    #[test]
    fn test_polynomial_subtraction_values() {
        // Polynomial p1: 3 + 2x + x^2 + 4y + 5xy + 6x^2y
        let p1 = BivariatePolynomial::new(array![
            [FE::new(3), FE::new(2), FE::new(1)], // 3 + 2x + x^2
            [FE::new(4), FE::new(5), FE::new(6)], // 4y + 5xy + 6x^2y
            [FE::new(7), FE::new(8), FE::new(9)], // 7y^2 + 8xy^2 + 9x^2y^2
        ]);

        // Polynomial p2: 1 + x + 2x^2 + 2y + 3xy + 4x^2y
        let p2 = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(1), FE::new(2)], // 1 + x + 2x^2
            [FE::new(2), FE::new(3), FE::new(4)], // 2y + 3xy + 4x^2y
            [FE::new(5), FE::new(6), FE::new(7)], // 5y^2 + 6xy^2 + 7x^2y^2
        ]);

        // Expected result: p1 - p2
        // = (3 - 1) + (2 - 1)x + (1 - 2)x^2 + (4 - 2)y + (5 - 3)xy + (6 - 4)x^2y + (7 - 5)y^2 + (8 - 6)xy^2 + (9 - 7)x^2y^2
        let expected = BivariatePolynomial::new(array![
            [FE::new(2), FE::new(1), FE::new(22)], // 2 + x + (-1)x^2 (22 mod 23 = -1)
            [FE::new(2), FE::new(2), FE::new(2)],  // 2y + 2xy + 2x^2y
            [FE::new(2), FE::new(2), FE::new(2)],  // 2y^2 + 2xy^2 + 2x^2y^2
        ]);

        // Perform the subtraction
        let result = p1 - p2;

        // Assert that the result matches the expected polynomial
        assert_eq!(
            result, expected,
            "The polynomial subtraction result is incorrect."
        );
    }

    #[test]
    fn test_bivariate_polynomial_subtraction_references() {
        let poly_a = polynomial_a();
        let poly_b = polynomial_b();

        let result = &poly_a - &poly_b;

        // Expected result: (2 - x + 3x^2) + (-3y + xy) + (-4xy^2)
        let expected_result = BivariatePolynomial::new(array![
            [FE::new(2), FE::new(22), FE::new(0)],
            [FE::new(20), FE::new(21), FE::new(1)],
            [FE::new(0), FE::new(4), FE::new(0)],
        ]);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_field_element_minus_polynomial() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x
        let polynomial =
            BivariatePolynomial::new(array![[FE::new(1), FE::new(2)], [FE::new(0), FE::new(0)],]);

        // Expected result: (5 - 1) + 2x = 4 + 2x
        let expected_coeffs = array![[FE::new(4), FE::new(2)], [FE::new(0), FE::new(0)],];

        let result = &element - &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_minus_borrowed_polynomial() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x
        let polynomial = BivariatePolynomial::new(array![[FE::new(1), FE::new(2)]]);

        // Expected result: (5 - 1) + 2x = 4 + 2x
        let expected_coeffs = array![[FE::new(4), FE::new(2)]];

        let result = element - &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_minus_owned_polynomial() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x
        let polynomial = BivariatePolynomial::new(array![[FE::new(1), FE::new(2)]]);

        // Expected result: (5 - 1) + 2x = 4 + 2x
        let expected_coeffs = array![[FE::new(4), FE::new(2)]];

        let result = element - polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_minus_empty_polynomial() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5
        let expected_coeffs = array![[FE::new(5)]];

        let result = element - polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_minus_owned_polynomial_2d() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x + 3y + 4xy (2D array)
        let polynomial = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2)], // 1 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ]);

        // Expected result: (5 - 1) + 2x + 3y + 4xy = 4 + 2x + 3y + 4xy
        let expected_coeffs = array![
            [FE::new(4), FE::new(2)], // 4 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ];

        let result = &element - polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_minus_empty_polynomial_2d() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5 (in a 1x1 2D array)
        let expected_coeffs = array![[FE::new(5)]];

        let result = &element - polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_minus_owned_polynomial_with_zeros() {
        let element = FE::new(5);

        // Polynomial: 0 + 0x + 0y + 0xy (all coefficients are zero)
        let polynomial = BivariatePolynomial::new(array![
            [FE::zero(), FE::zero()], // 0 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ]);

        // Expected result: 5 (in a 2x2 2D array with 5 in the top-left corner)
        let expected_coeffs = array![
            [FE::new(5), FE::zero()], // 5 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ];

        let result = &element - polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_plus_borrowed_polynomial_2d() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x + 3y + 4xy (2D array)
        let polynomial = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2)], // 1 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ]);

        // Expected result: (5 + 1) + 2x + 3y + 4xy = 6 + 2x + 3y + 4xy
        let expected_coeffs = array![
            [FE::new(6), FE::new(2)], // 6 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ];

        let result = &element + &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_plus_empty_polynomial_2d() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5 (in a 1x1 2D array)
        let expected_coeffs = array![[FE::new(5)]];

        let result = &element + &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_plus_polynomial_with_zeros() {
        let element = FE::new(5);

        // Polynomial: 0 + 0x + 0y + 0xy (all coefficients are zero)
        let polynomial = BivariatePolynomial::new(array![
            [FE::zero(), FE::zero()], // 0 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ]);

        // Expected result: 5 (in a 2x2 2D array with 5 in the top-left corner)
        let expected_coeffs = array![
            [FE::new(5), FE::zero()], // 5 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ];

        let result = &element + &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_plus_borrowed_polynomial_2d() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x + 3y + 4xy (2D array)
        let polynomial = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2)], // 1 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ]);

        // Expected result: (5 + 1) + 2x + 3y + 4xy = 6 + 2x + 3y + 4xy
        let expected_coeffs = array![
            [FE::new(6), FE::new(2)], // 6 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ];

        let result = element + &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_plus_empty_polynomial_2d_own_ref() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5 (in a 1x1 2D array)
        let expected_coeffs = array![[FE::new(5)]];

        let result = element + &polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_plus_owned_polynomial_2d_own_own() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x + 3y + 4xy (2D array)
        let polynomial = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2)], // 1 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ]);

        // Expected result: (5 + 1) + 2x + 3y + 4xy = 6 + 2x + 3y + 4xy
        let expected_coeffs = array![
            [FE::new(6), FE::new(2)], // 6 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ];

        let result = element + polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_plus_empty_polynomial_2d() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5 (in a 1x1 2D array)
        let expected_coeffs = array![[FE::new(5)]];

        let result = element + polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_owned_field_element_plus_polynomial_with_zeros() {
        let element = FE::new(5);

        // Polynomial: 0 + 0x + 0y + 0xy (all coefficients are zero)
        let polynomial = BivariatePolynomial::new(array![
            [FE::zero(), FE::zero()], // 0 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ]);

        // Expected result: 5 (in a 2x2 2D array with 5 in the top-left corner)
        let expected_coeffs = array![
            [FE::new(5), FE::zero()], // 5 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ];

        let result = element + polynomial;

        assert_eq!(expected_coeffs, result.coefficients);
    }

    #[test]
    fn test_borrowed_field_element_plus_owned_polynomial_2d() {
        let element = FE::new(5);

        // Polynomial: 1 + 2x + 3y + 4xy (2D array)
        let polynomial = BivariatePolynomial::new(array![
            [FE::new(1), FE::new(2)], // 1 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ]);

        // Expected result: (5 + 1) + 2x + 3y + 4xy = 6 + 2x + 3y + 4xy
        let expected_coeffs = array![
            [FE::new(6), FE::new(2)], // 6 + 2x
            [FE::new(3), FE::new(4)], // 3y + 4xy
        ];

        let result = &element + polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_plus_owned_empty_polynomial_2d() {
        let element = FE::new(5);

        // Empty polynomial
        let polynomial = BivariatePolynomial::new(array![[]]);

        // Expected result: 5 (in a 1x1 2D array)
        let expected_coeffs = array![[FE::new(5)]];

        let result = &element + polynomial;

        assert_eq!(result.coefficients, expected_coeffs);
    }

    #[test]
    fn test_borrowed_field_element_plus_owned_polynomial_with_zeros() {
        let element = FE::new(5);

        // Polynomial: 0 + 0x + 0y + 0xy (all coefficients are zero)
        let polynomial = BivariatePolynomial::new(array![
            [FE::zero(), FE::zero()], // 0 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ]);

        // Expected result: 5 (in a 2x2 2D array with 5 in the top-left corner)
        let expected_coeffs = array![
            [FE::new(5), FE::zero()], // 5 + 0x
            [FE::zero(), FE::zero()], // 0y + 0xy
        ];

        let result = &element + polynomial;

        assert_eq!(result.coefficients, expected_coeffs)
    }
}