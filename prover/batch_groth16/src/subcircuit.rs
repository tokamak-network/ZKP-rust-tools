use lambdaworks_groth16::{common::{FrElement,FrField}, r1cs::{self, ConstraintSystem, R1CS}};
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;
use lambdaworks_math::{fft::errors::FFTError, polynomial::Polynomial as UnivariatePolynomial};
use ndarray::{s, Array, Array2, ArrayBase, Axis, IndexLonger, Ix2};

// 

#[derive(Debug)]
pub struct SubcircuitManager{
    pub num_subcircuits: usize, 
    pub max_constraints: usize, // 2^4 
    pub max_witness: usize, // maybe not neccessary 
    pub subcircuits: Vec<R1CS>,

}



impl SubcircuitManager {
    pub fn new(num_subcircuits: usize, max_constraints: usize, max_witness: usize, subcircuits: Vec<R1CS>) -> Self {
        let max_constraints = max_constraints.next_power_of_two();
        Self {
            num_subcircuits,
            max_constraints,
            max_witness,
            subcircuits,
        }
    } 
    // will return U V, W matrices 
    pub fn concatinate_subcircuits(&self ,ordered_idx: &[usize], witnesses: &[Vec<FrElement>]) -> Result<(BivariatePolynomial<FrElement>,BivariatePolynomial<FrElement>,BivariatePolynomial<FrElement>),FFTError> {

        let mut l_matrix = Array2::<FrElement>::default((self.max_constraints, witnesses.len()));
        let mut r_matrix = Array2::<FrElement>::default((self.max_constraints, witnesses.len()));
        let mut o_matrix = Array2::<FrElement>::default((self.max_constraints, witnesses.len()));

        for i in 0..witnesses.len() {
            for (constraint_cnt, constraint ) in self.subcircuits[ordered_idx[i]].constraints.iter().enumerate(){

                let u_i_j = constraint.a.iter().enumerate().fold(FrElement::zero(), |acc, (j, a_j)| acc + a_j * witnesses[i][j].clone());
                let v_i_j = constraint.b.iter().enumerate().fold(FrElement::zero(), |acc, (j, b_j)| acc + b_j * witnesses[i][j].clone());
                let w_i_j = constraint.c.iter().enumerate().fold(FrElement::zero(), |acc, (j, c_j)| acc + c_j * witnesses[i][j].clone());
                let v_entry = r_matrix.get_mut((constraint_cnt, i)).unwrap();
                *v_entry = v_i_j;

                let w_entry = o_matrix.get_mut((constraint_cnt, i)).unwrap();
                *w_entry = w_i_j;

                let u_entry = l_matrix.get_mut((constraint_cnt, i)).unwrap();
                *u_entry = u_i_j;

            }
        }
        let l_poly = BivariatePolynomial::interpolate_fft::<FrField>(&l_matrix)?;
        let r_poly = BivariatePolynomial::interpolate_fft::<FrField>(&r_matrix)?;
        let o_poly = BivariatePolynomial::interpolate_fft::<FrField>(&o_matrix)?;
    
        Ok((l_poly, r_poly, o_poly))
    }

    pub fn generate_proof(&self, u: BivariatePolynomial<FrElement>,v: BivariatePolynomial<FrElement>,w: BivariatePolynomial<FrElement>) {
        
            

        todo!()
    }
}

// #[inline]
// fn get_varialbe_lro_array_from_r1cs() -> []




// this function should calculate lro polynomials with zeropadding which is equal to max_constraints . 
pub fn lro_polys_from_r1cs(r1cs: R1CS, max_constraints: usize) -> [Vec<UnivariatePolynomial<FrElement>>;3] {
    let num_gates = r1cs.number_of_constraints();
    let pad_zeroes: usize = max_constraints - num_gates;

    let mut l: Vec<UnivariatePolynomial<FrElement>> = vec![];
    let mut r: Vec<UnivariatePolynomial<FrElement>> = vec![];
    let mut o: Vec<UnivariatePolynomial<FrElement>> = vec![];
    // 
    for i in 0..r1cs.witness_size() {
        let [l_poly, r_poly, o_poly] =
            get_variable_lro_polynomials_from_r1cs(&r1cs, i, pad_zeroes);
        l.push(l_poly);
        r.push(r_poly);
        o.push(o_poly);
    }

    [l,r,o]
    // todo!()
}   

#[inline]
fn get_variable_lro_polynomials_from_r1cs(
    r1cs: &R1CS,
    var_idx: usize,
    pad_zeroes: usize,
) -> [UnivariatePolynomial<FrElement>; 3] {
    let cap = r1cs.number_of_constraints() + pad_zeroes;
    let mut current_var_l = vec![FrElement::zero(); cap];
    let mut current_var_r = vec![FrElement::zero(); cap];
    let mut current_var_o = vec![FrElement::zero(); cap];

    for (i, c) in r1cs.constraints.iter().enumerate() {
        current_var_l[i] = c.a[var_idx].clone();
        current_var_r[i] = c.b[var_idx].clone();
        current_var_o[i] = c.c[var_idx].clone();
    }

    [current_var_l, current_var_r, current_var_o]
        .map(|e| UnivariatePolynomial::interpolate_fft::<FrField>(&e).unwrap())
}

// pub fn setup_subcircuits(
//     subcircuits_file_content: &[&str], 
// ){
//     todo!()
// }





#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_groth16::common::FrElement;

    use lazy_static::lazy_static;
    // ... existing code ...
    

    lazy_static! {
        static ref KEVIN_EXAMPLE_A: R1CS = R1CS::from_matrices(
            vec![
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)],
                vec![FrElement::from(5),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)]
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
            ],
            5,
        );
        //https://learn.0xparc.org/materials/circom/additional-learning-resources/r1cs%20explainer/
        static ref KEVIN_WITNESS: Vec<FrElement> = vec![FrElement::from(1),FrElement::from(3),FrElement::from(35),FrElement::from(9),FrElement::from(27),FrElement::from(30)];


        static ref THREE_FAC_MOONMATH: R1CS = R1CS::from_matrices(
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)]
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)]
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)],
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)]
            ],
            5,
        );
        // 2*3*4 = 24 
        static ref THREE_FAC_WITNESS: Vec<FrElement> = vec![FrElement::from(1),FrElement::from(24),FrElement::from(2),FrElement::from(3),FrElement::from(4),FrElement::from(6)];



        //four factorization https://www.rareskills.io/post/rank-1-constraint-system#:~:text=are%20as%20follows-,Checking,-our%20work%20for

        static ref FOUR_FAC: R1CS = R1CS::from_matrices(
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)]

            ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)]
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)],
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)]
            ],
            7,
        );
        static ref FOUR_FAC_WITNESS: Vec<FrElement> =  vec![FrElement::from(1),FrElement::from(1680),FrElement::from(5),FrElement::from(6),FrElement::from(7),FrElement::from(8),FrElement::from(30),FrElement::from(42)];

        // x^3 + y^3 = z^3 , doesnt have solution, just for fun :) . the only possible solution is zero vector :)
        static ref CUBIC_FERMA_LAST_THEOREM: R1CS = R1CS::from_matrices(
            vec![
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(1),FrElement::from(0)]
            ],
            vec![
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                ],
            vec![
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0)],
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0),FrElement::from(0)],  
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0),FrElement::from(0)],  
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1),FrElement::from(0)],  
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)],  
                vec![FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(0),FrElement::from(1)],  
                ],
            9,
        );

    }
    
    
    // #[test]
    // fn test1() {
    //     // SubcircuitManager::new(3, max_constraints, max_witness, subcircuits)
    // }
    
}