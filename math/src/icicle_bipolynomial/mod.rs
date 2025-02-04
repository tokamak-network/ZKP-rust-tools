pub mod bifft;
pub mod dense_ext;
pub mod bipolynomial;


#[cfg(test)]
mod tests {
    use super::*;
    use bipolynomial::BivariatePolynomial;
    use dense_ext::DensePolynomialExt;
    use icicle_bls12_381::{curve::ScalarField, polynomials::DensePolynomial};
    use icicle_core::{polynomials::UnivariatePolynomial, traits::FieldImpl};
    use icicle_runtime::memory::HostSlice;

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