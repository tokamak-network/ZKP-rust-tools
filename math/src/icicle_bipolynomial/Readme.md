# Bivariate Polynomial Implementation in Rust

This document provides an overview of the **`BivariatePolynomial`** structure and the accompanying methods for handling two-variable polynomials in Rust. The implementation relies on `DensePolynomial` for each row (in terms of `x`), combined to form a polynomial in `(x, y)`.

---
## Overview

We represent a bivariate polynomial 
<img src="https://latex.codecogs.com/svg.latex?p%28x%2Cy%29" title="p(x,y)" /> 
of degrees 
<img src="https://latex.codecogs.com/svg.latex?%5Cbigl%28d_x%2C%20d_y%5Cbigr%29" title="(d_x, d_y)" />
by storing one `DensePolynomial` for each power of 
<img src="https://latex.codecogs.com/svg.latex?y" title="y" />. Specifically:

<img src="https://latex.codecogs.com/svg.latex?p(x,y)&space;\;=\;&space;\sum_{i=0}^{y\_degree} " title="p(x,y) \;=\; \sum_{i=0}^{y\_degree} }" />
<img src="https://latex.codecogs.com/svg.latex?\Bigl(\text{DensePolynomial&space;in&space;}&space;x\Bigr)\;\cdot\;y^i." title="p(x,y) \;=\; \sum_{i=0}^{y\_degree} }" />


This approach allows us to leverage existing operations for one‐dimensional polynomials (`DensePolynomial`)—such as addition, subtraction, scalar multiplication—and extend them to two variables by stacking them row‐by‐row in terms of powers of \(y\).

---

## DensePolynomialExt

```rust
fn get_coefficients(&self) -> Vec<ScalarField>
fn scale(&self, scalar: &ScalarField) -> Self
fn sub_constant(&mut self, constant: ScalarField)
fn ruffini_division(&self, b: &ScalarField) -> Result<(DensePolynomial, ScalarField), &'static str>
fn add_polynomial(&self, other: &DensePolynomial) -> DensePolynomial
```

## BivariatePolynomial

```rust
pub struct BivariatePolynomial {
    pub coefficients: Vec<DensePolynomial>,
    pub x_degree: usize,
    pub y_degree: usize,
}

pub fn evaluate(&self, x: &ScalarField, y: &ScalarField) -> ScalarField

pub fn ruffini_division(
    &self,
    a: &ScalarField,
    b: &ScalarField,
) -> Result<(BivariatePolynomial, DensePolynomial), &'static str>

pub fn scale(&self, x_factor: &ScalarField, y_factor: &ScalarField) -> Self

impl Add for BivariatePolynomial { ... }
impl Sub for BivariatePolynomial { ... }
```

## Test Suite

Within the #[cfg(test)] module, there are several unit tests:

	1.	test_bivariate_polynomial_new: Checks if constructing a polynomial using new stores the correct coefficients and degrees.
	2.	test_evaluate: Evaluates a known polynomial at (x, y) = (2, 3) and compares against the expected numeric result.
	3.	test_zero: Ensures BivariatePolynomial::zero() is indeed the zero polynomial.
	4.	test_bivariate_polynomial_ruffini_division: Performs Ruffini division (x - a, y - b) and checks the returned quotient and remainder polynomials.
	5.	test_polynomial_addition: Validates that bivariate addition is correct.
	6.	test_polynomial_subtraction: Validates that bivariate subtraction is correct.
	7.	test_sub_by_field_element: Checks subtracting a scalar from each row’s constant term.
	8.	test_scale: Validates scaling in the x-direction and y-direction individually.


| Test Function                                   | Result |
|-------------------------------------------------|--------|
| **test_bivariate_polynomial_new**               | Pass   |
| **test_evaluate**                               | Pass   |
| **test_zero**                                   | Pass   |
| **test_bivariate_polynomial_ruffini_division**  | fail   |
| **test_polynomial_addition**                    | Pass   |
| **test_polynomial_subtraction**                 | fail   |
| **test_sub_by_field_element**                   | fail   |
| **test_scale**                                  | Pass   |