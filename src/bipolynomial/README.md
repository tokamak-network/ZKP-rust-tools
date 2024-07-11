# Bivariate Polynomials

Contains simple Bivariate polynomial, with ruffini algorithm for division. 
Because of this [problem](https://www.quora.com/How-do-I-divide-a-polynomial-with-two-variables-by-long-division) defining long division for Bivariate polynomials is not possible but divide them to degree 1 binomials make sense. 

TODO -> 2D-FFT/IFFT for evaluation/interpolation. operator overloading for ADD, NEG, SUB and MUL. based on above Div operation is not possible for general cases and only possible for degree 1 binomials. If needed we can search more on Grobner basis but the remainder is not deterministic if 2 polynomials don't have bezout identity. 
