use lambdaworks_math::{
    field::{element::FieldElement, traits::{IsFFTField, IsField}},
    unsigned_integer::element::{U256, U64},

};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use ndarray::{Array2, concatenate, Axis};
use lambdaworks_groth16::{common::{FrElement,FrField}, r1cs::{self, ConstraintSystem, R1CS}};
use rand::Rng;


// as it is demonstration, we need to think about what should be the input of this struct, 
// the permutation of subcircuit wires is needed to calculate a_0, a_1 , s_0 , s_1, b. 
pub struct CopyConstraintProver {
    s_max: u32,
    l_d: u32,
    l: u32,


    a_0: Array2<u32>, 
    a_1: Array2<u32>,

    s_0: BivariatePolynomial<FrElement>,
    s_1: BivariatePolynomial<FrElement>,

    b: BivariatePolynomial<FrElement>,


}

impl CopyConstraintProver {
    pub fn new(s_max: u32, l_d: u32, l: u32) -> Self {
        let (a_0,s_0_evals) = random_permutation(l_d,s_max,s_max);
        let (a_1, s1_evals) =  random_permutation(l_d,s_max,l_d);
        let s_0 =  BivariatePolynomial::interpolate_fft::<FrField>(&s_0_evals).unwrap();
        let s_1 = BivariatePolynomial::interpolate_fft::<FrField>(&s1_evals).unwrap();
        let b = calculate_b_permutation_poly(&a_0,&a_1);
        CopyConstraintProver { s_max, l_d, l, a_0, a_1,s_0, s_1,b }
    }

    // pub fn set_random_challenges(theta_0: FrElement, theta_1: FrElement, theta_2: FrElement)

    pub fn prove(&self) -> (BivariatePolynomial<FrElement>, BivariatePolynomial<FrElement>){
        let tetha_0 = random_fr(); 
        let tetha_1 = random_fr(); 
        let tetha_2 = random_fr();
        
        let b = calculate_b_permutation_poly(&self.a_0, &self.a_1);
        let one = BivariatePolynomial::new(Array2::<FrElement>::from_elem((self.l_d as usize, self.s_max as usize), FrElement::one()));
        #[cfg(debug_assertions)]
        println!("one :: {}", one);

        let bib_bib = BivariatePolynomial::interpolate_fft::<FrField>(&Array2::<FrElement>::from_elem((self.l_d as usize, self.s_max as usize), FrElement::one())).unwrap(); 
        #[cfg(debug_assertions)]
        println!("bib_bib :: {}", bib_bib);


        let zero_matrix = Array2::<FrElement>::from_elem((self.l_d as usize, self.s_max as usize), FrElement::zero());
        
        let mut y_monomial_matrix = zero_matrix.clone(); 
        y_monomial_matrix[(0,1)] = FrElement::one() ;

        let y_monomial = BivariatePolynomial::new(y_monomial_matrix);
        #[cfg(debug_assertions)]
        println!("{}", y_monomial);


        let mut z_monomial_matrix = zero_matrix.clone(); 
        z_monomial_matrix[(1,0)] = FrElement::one() ;

        let z_monomial = BivariatePolynomial::new(z_monomial_matrix);


        let f = &b + &tetha_0 * &self.s_0 + &tetha_1* &self.s_1 + &tetha_2 * &one;
        let g = &b + &tetha_0 * &y_monomial + &tetha_1 * &z_monomial + &tetha_2 * &one;




        ///Part 3 
        let mut c = zero_matrix.clone(); 
        c[(self.l_d as usize - 1 , self.s_max as usize - 1 )] = FrElement::one(); 

        for i in 0..self.s_max as usize {
            c[(0, i)] = if i > 0 {
                c[(self.l_d as usize - 1 , i - 1)].clone()
            } else {
                c[(self.l_d as usize - 1 , self.s_max as usize - 1)].clone()
            };

            for j in 1..self.l_d as usize {
                c[(j, i)] = c[(j - 1, i)].clone() * f.coefficients[(j, i)].clone() / g.coefficients[(j, i)].clone();
            }
        }


        let r = BivariatePolynomial::interpolate_fft::<FrField>(&c).unwrap(); 


        // part4 
        let k = random_fr();
       
        // part5 
        let mut d = zero_matrix.clone(); 
        d[(self.l_d as usize - 1 , self.s_max as usize - 1 )] = FrElement::one(); 


        let e = BivariatePolynomial::interpolate_fft::<FrField>(&d).unwrap(); 

        let r_minus_one = -FrElement::one() + &r ;

        let r_minus_one_evaluation =  BivariatePolynomial::evaluate_fft::<FrField>(&r_minus_one, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        let e_evaluation =  BivariatePolynomial::evaluate_fft::<FrField>(&e, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();

        let p1_evaluation = r_minus_one_evaluation * e_evaluation; 

        let p_1 = BivariatePolynomial::interpolate_fft::<FrField>(&p1_evaluation).unwrap(); 

        let r_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&r, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        let g_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&g, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();

        let h_evaluation = r_evaluation * g_evaluation; 
        let h = BivariatePolynomial::interpolate_fft::<FrField>(&h_evaluation).unwrap(); 

        let w_z_inverse = FrField::get_primitive_root_of_unity(self.l_d as u64 - self.l as u64).unwrap().inv().unwrap();
        let w_y_inverse = FrField::get_primitive_root_of_unity(self.s_max as u64).unwrap().inv().unwrap();

        let r_scaled_1_inverse_w_z_evaluation = BivariatePolynomial::evaluate_offset_fft(&r, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 ),&FrElement::one() , &w_z_inverse).unwrap();
        let f_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&f, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        
        let i_evaluation = r_scaled_1_inverse_w_z_evaluation * &f_evaluation; 

        let i = BivariatePolynomial::interpolate_fft::<FrField>(&i_evaluation).unwrap(); 

        let r_scaled_inverse_w_y_inverse_w_z = BivariatePolynomial::evaluate_offset_fft(&r, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 ),&w_y_inverse , &w_z_inverse).unwrap();

        let j_evaluation = r_scaled_inverse_w_y_inverse_w_z * &f_evaluation; 

        let j = BivariatePolynomial::interpolate_fft::<FrField>(&j_evaluation).unwrap(); 


        // let z_minus_one = -FrElement::one() + &z ;


        let z_minus_one = -FrElement::one() + &z_monomial ;
        let h_minus_i = &h - &i ; 
        let h_minus_i_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&h_minus_i, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        
        let z_minuc_one_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&z_minus_one, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        
        let p_2_evaluation = z_minuc_one_evaluation * h_minus_i_evaluation; 
        let p_2 = BivariatePolynomial::interpolate_fft::<FrField>(&p_2_evaluation).unwrap(); 


        let mut k_0_evaluation = zero_matrix.clone() ;
        for j in 0..self.s_max as usize {
            k_0_evaluation[(0, j)] = FrElement::one();
        }

        let k_0 = BivariatePolynomial::interpolate_fft::<FrField>(&k_0_evaluation).unwrap(); 
        let k_0_resize_evaluations = BivariatePolynomial::evaluate_fft::<FrField>(&k_0, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();

        // it is not necessary 

        // let k_0 = BivariatePolynomial::interpolate_fft::<FrField>(&k_0_evaluation);
        let h_minus_j = &h - &j ; 
        let h_minus_i_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&h_minus_j, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();
        
        let p3_evaluation = h_minus_i_evaluation * k_0_resize_evaluations;
        let p_3 = BivariatePolynomial::interpolate_fft::<FrField>(&p3_evaluation).unwrap(); 


        let p = k.pow(3 as usize) * p_3 + k.pow(2 as usize) * p_2 + k * p_1 ;

        assert_eq!(p.polynomial_dimension() , (2 * self.s_max as usize - 2 , 3 * self.l_d as usize - 3));

        // part 6 
        let zeta: FieldElement<FrField> = FieldElement::from(3);
        let gamma: FieldElement<FrField> = FieldElement::from(5);
        // let p_coset_z_evaluation = BivariatePolynomial::evaluate_offset_fft::<FrField>(&p, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 ),&FrElement::one() , &gamma).unwrap();
        let indices_left: Vec<usize> = (0..self.s_max as usize).collect();
        let indices_right: Vec<usize> = ((self.s_max as usize)..p.coefficients.len_of(Axis(0))).collect();

        let p_left_coefficients = &p.coefficients.select(Axis(1), &indices_left );
        let p_right_coefficients = &p.coefficients.select(Axis(1), &indices_right);

        // need to check the correctness of dimensions 
        let p_left_roots_of_unity_evals = BivariatePolynomial::evaluate_offset_fft::<FrField>(&BivariatePolynomial::new(p_left_coefficients.to_owned()), 1, 1, Some(3*self.l_d as usize - 2  ), Some(self.s_max as usize  ), &FieldElement::one(), &zeta).unwrap();
        let p_right_roots_of_unity_evals = BivariatePolynomial::evaluate_offset_fft::<FrField>(&BivariatePolynomial::new(p_right_coefficients.to_owned()), 1, 1, Some(3*self.l_d as usize - 2  ), Some(self.s_max as usize  ) , &FieldElement::one() , &zeta).unwrap();

        let mut r_roots_of_unity_evals = p_left_roots_of_unity_evals + p_right_roots_of_unity_evals;


        let divisor_inv_zeta = (zeta.pow(self.s_max) - FrElement::one()).inv().unwrap(); 
     
        r_roots_of_unity_evals = r_roots_of_unity_evals.map_mut(|elem| elem.clone() * &divisor_inv_zeta);

        let pi_z_poly = BivariatePolynomial::interpolate_offset_fft::<FrField>(&r_roots_of_unity_evals, &FrElement::one(), &zeta).unwrap();

        assert_eq!(pi_z_poly.polynomial_dimension() , (self.s_max as usize - 1 , 2* self.l_d as usize - 3));

        let pi_z_coefficients_negated = pi_z_poly.coefficients.mapv(|elem| -elem);
        let pi_z_negated = BivariatePolynomial::new(pi_z_coefficients_negated);
        
        // this concatination should be revised
        let zero_l_d_minus_one_matrix = Array2::<FrElement>::from_elem((self.l_d as usize - 1, self.s_max as usize), FrElement::zero());

        //p_y should be replaced with p_z
        let p_z_times_t_z_coefficients = concatenate(Axis(0), &[zero_l_d_minus_one_matrix.view(), pi_z_poly.coefficients.view()]).unwrap();
        let p_z_times_t_z = BivariatePolynomial::new(p_z_times_t_z_coefficients);


        // let remainder_poly_coefficients = 


        let remainder_poly = pi_z_negated + p_z_times_t_z;
        

        // let h_minus_i_evaluation = BivariatePolynomial::evaluate_fft::<FrField>(&h_minus_i, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 )).unwrap();

        let remainder_poly_coset_y_evaluations = BivariatePolynomial::evaluate_offset_fft::<FrField>(&remainder_poly, 1, 1, Some(3*self.l_d as usize - 2  ), Some(2*self.s_max as usize -1 ), &gamma, &FieldElement::one()).unwrap();

        // let p_left_roots_of_unity_evals = BivariatePolynomial::evaluate_offset_fft::<FrField>(&BivariatePolynomial::new(p_left_coefficients.to_owned()), 1, 1, Some(3*self.l_d as usize - 2  ), Some(self.s_max as usize  ), &FieldElement::one(), &zeta).unwrap();

        let p_coset_y_evaluations = BivariatePolynomial::evaluate_offset_fft::<FrField>(&p, 1, 1, Some(3*self.l_d as usize - 2  ), Some(self.s_max as usize  ), &gamma, &FieldElement::one()).unwrap();

        let mut q_coset_y_evaluation = remainder_poly_coset_y_evaluations - p_coset_y_evaluations; 
        let divisor_inv_gamma = (gamma.pow(self.l_d - 1) - FrElement::one()).inv().unwrap(); 

        q_coset_y_evaluation = q_coset_y_evaluation.map_mut(|elem| elem.clone() * &divisor_inv_gamma);

        let pi_y_poly= BivariatePolynomial::interpolate_offset_fft::<FrField>(&q_coset_y_evaluation, &gamma, &FrElement::one()).unwrap();

        // todo!()     
        (pi_y_poly, pi_z_poly) 
    }
}



// to create A_0 and A_1 we need to now about the upper bound to create random element.
pub fn random_permutation(row: u32, col: u32, upper_bound: u32) ->(Array2<u32> ,Array2<FrElement>)
{
    let mut rng = rand::thread_rng();
    let mut to_power_random_evals = Array2::<FrElement>::from_elem((row as usize, col as usize), FrElement::zero());
    let mut random_evals = Array2::<u32>::from_elem((row as usize, col as usize), 0);

    let w = FrField::get_primitive_root_of_unity(upper_bound as u64).unwrap();

    for i in 0..to_power_random_evals.nrows() {
        for j in 0..to_power_random_evals.ncols() {
            let random = rng.gen_range(0..upper_bound);
            to_power_random_evals[(i, j)] = w.pow(random);
            random_evals[(i, j)] = random;

        }
    }
    (random_evals ,to_power_random_evals)
}


// in this function I create W first and after that calculate IFFT to get B Bipoly.
pub fn calculate_b_permutation_poly(a_0: &Array2<u32>, a_1: &Array2<u32>) -> BivariatePolynomial<FrElement> {
    let mut w = Array2::<FrElement>::from_elem((a_0.nrows() as usize, a_0.ncols() as usize), FrElement::zero());

    for i in 0..a_0.nrows() {
        for j in 0..a_0.ncols() {
            // Your code here
            if w[(a_1[(i,j)] as usize ,a_0[(i,j)] as usize)] == FrElement::zero() {
                w[(a_1[(i,j)] as usize ,a_0[(i,j)] as usize)]  = random_fr();
            } else {
                w[(i,j)] = w[(a_1[(i,j)] as usize ,a_0[(i,j)] as usize)].clone();
            }
        }
    }


    let b =  BivariatePolynomial::interpolate_fft::<FrField>(&w).unwrap();
    b 
}

/// Generate a random field element
pub fn random_fr_upper_bound(upper_bound: u64) -> FrElement {
    let mut rng = rand::thread_rng();
    FrElement::from(rng.gen_range(0..upper_bound))
}

/// Generate a random field element
pub fn random_fr() -> FrElement {
    let mut rng = rand::thread_rng();
    FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    })
}



// pub fn random_wire_bipoly()


#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn test_x(){
        let prover = CopyConstraintProver::new(8, 4, 0);
        let (pi_y, pi_z) = prover.prove();
    }

    // I need to write a test to check the permutation B is correct or not , for that I should write 
    // all the functionality together because current implementation is not compatible with it 
    #[test]
    fn test_b_permutaion_calculation(){
        let l_d = 4; 
        let s_max = 8 ; 
        let (a_0,s_0_evals) = random_permutation(l_d,s_max,s_max);
        let (a_1, s1_evals) =  random_permutation(l_d,s_max,l_d);
    

        let s_0 =  BivariatePolynomial::interpolate_fft::<FrField>(&s_0_evals).unwrap();
        let s_1 = BivariatePolynomial::interpolate_fft::<FrField>(&s1_evals).unwrap();

        let b = calculate_b_permutation_poly(&a_0,&a_1);


        let w_z = FrField::get_primitive_root_of_unity(l_d as u64 ).unwrap();
        let w_y = FrField::get_primitive_root_of_unity(s_max as u64).unwrap();


        for i in 0..s_max {
            for j in 0..l_d {
                let c = b.evaluate(&w_y.pow(i), &w_z.pow(j));
                let p_i_j_1 = s_0.evaluate(&w_y.pow(i), &w_z.pow(j));
                let p_i_j_2 = s_1.evaluate(&w_y.pow(i), &w_z.pow(j));

                let c_expected = b.evaluate(&p_i_j_1, &p_i_j_2); 

                #[cfg(debug_assertions)]
                println!("bib_bib :: {:?}", c.value());
                #[cfg(debug_assertions)]
                println!("bib_bib :: {:?}", c_expected.value());
                assert_eq!(c , c_expected);

            }
        }
        
    }

    #[test]
    fn test_one_ifft_is_one(){
        // let one = BivariatePolynomial::new(Array2::<FrElement>::from_elem((5, 7), FrElement::one()));
        // #[cfg(debug_assertions)]
        // println!("one :: {}", one);

        let bib_bib = BivariatePolynomial::interpolate_fft::<FrField>(&Array2::<FrElement>::from_elem((4,8), FrElement::one())).unwrap(); 
        #[cfg(debug_assertions)]
        println!("bib_bib :: {}", bib_bib);

        assert_eq!(bib_bib.coefficients.get((0,0)).unwrap(), &FrElement::one());
    }
}   