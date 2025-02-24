
use crate::error::Error;
use lambdaworks_groth16::common::{FrElement, FrField};
use lambdaworks_math::{
    field::traits::{IsField},
    polynomial::Polynomial as UnivariatePolynomial,
    fft,

};
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

use serde_json::{Value, from_str};
use std::{clone, default, fs, ops::Sub, result, vec};
use ndarray::{s, Array, Array2, Axis};

use std::sync::Mutex;



use rayon::{iter::ParallelIterator, prelude::{IndexedParallelIterator, IntoParallelRefIterator}};
#[derive(Debug,Clone)]
pub struct SubcircuitQAP{
    // pub subcircuit_id :usize, 
    pub u :Vec<UnivariatePolynomial<FrElement>>,
    pub v :Vec<UnivariatePolynomial<FrElement>>,
    pub w :Vec<UnivariatePolynomial<FrElement>>,
}





#[derive(Debug)]
pub struct SubcircuitR1CS {
    // pub subcircuit_id :usize, 

    pub a :Array2<FrElement>,
    pub b :Array2<FrElement>,
    pub c :Array2<FrElement>,
}

impl SubcircuitR1CS {
    pub fn from_path(max_constraint :usize, subcircuit_cnt :usize, r1cs_file_path :&str) -> Result<Vec<SubcircuitR1CS>,Error> {

        let mut result = Vec::with_capacity(subcircuit_cnt);

        for subcircuit_id in 0..subcircuit_cnt{

            let subcircuit_id_r1cs = format!("{r1cs_file_path}/json/subcircuit{subcircuit_id}.r1cs.json");
            
            let r1cs_file_content =
              &fs::read_to_string(subcircuit_id_r1cs).expect("Error reading the file");

            let circom_r1cs: Value = serde_json::from_str(r1cs_file_content).expect("Error parsing JSON");

            let num_of_vars = circom_r1cs["nVars"].as_u64().unwrap() as usize; // Includes "1"
           
           
            
            
            
            let mut l = Array2::<FrElement>::from_elem((max_constraint,num_of_vars), FrElement::zero());
            let mut r = Array2::<FrElement>::from_elem((max_constraint,num_of_vars), FrElement::zero());
            let mut o = Array2::<FrElement>::from_elem((max_constraint,num_of_vars), FrElement::zero());
            
            for (constraint_idx, constraint) in circom_r1cs["constraints"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            {
                let constraint = constraint.as_array().unwrap();
                for (var_idx, str_val) in constraint[0].as_object().unwrap() {
                    l[(constraint_idx ,var_idx.parse::<usize>().unwrap())] =
                        circom_str_to_lambda_field_element(str_val.as_str().unwrap());
                }
                for (var_idx, str_val) in constraint[1].as_object().unwrap() {
                    r[(constraint_idx ,var_idx.parse::<usize>().unwrap())] =
                        circom_str_to_lambda_field_element(str_val.as_str().unwrap());
                }
                for (var_idx, str_val) in constraint[2].as_object().unwrap() {
                    o[(constraint_idx ,var_idx.parse::<usize>().unwrap())] =
                        circom_str_to_lambda_field_element(str_val.as_str().unwrap());
                }
            }

            result.push( SubcircuitR1CS{
                a : l,
                b : r, 
                c : o,
            });

        }
        Ok(result)
    }



}

// max
impl SubcircuitQAP {
    pub fn default_vec(capacity :usize) -> Vec<SubcircuitQAP> {

        vec![ SubcircuitQAP { u: Vec::new(), v: Vec::new(), w: Vec::new() } ; capacity]

    }

    pub fn from_r1cs(r1cs_list :Vec<SubcircuitR1CS> ) -> Result<Vec<SubcircuitQAP>,Error> {
        // max_constraint
        // let result = Mutex::new(vec![SubcircuitQAP::default(); r1cs_list.len()]);
        
        let mut result = Mutex::new(SubcircuitQAP::default_vec(r1cs_list.len()));
        
        

        r1cs_list.par_iter().enumerate().for_each(|(id , subcircuit_r1cs)|  {
            let m_k = subcircuit_r1cs.a.dim().1 ; 

            let mut u_poly = Vec::with_capacity(m_k);
            let mut v_poly = Vec::with_capacity(m_k);
            let mut w_poly = Vec::with_capacity(m_k);

            for a_row in subcircuit_r1cs.a.axis_iter(Axis(1)) {
                u_poly.push(UnivariatePolynomial::interpolate_fft::<FrField>(&a_row.to_vec()).unwrap());// TODO :: capture error ?? 
            }
            for b_row in subcircuit_r1cs.b.axis_iter(Axis(1)) {
                v_poly.push(UnivariatePolynomial::interpolate_fft::<FrField>(&b_row.to_vec()).unwrap());
            }            
            for c_row in subcircuit_r1cs.c.axis_iter(Axis(1)) {
                w_poly.push(UnivariatePolynomial::interpolate_fft::<FrField>(&c_row.to_vec()).unwrap());
            }            

            let mut res = result.lock().unwrap();

            res[id] = SubcircuitQAP{
                u :u_poly,
                v :v_poly,
                w :w_poly,
            };
            // Ok(())

        });
        let dd = result.into_inner().unwrap(); 
        
    


        Ok(dd)

    }
}

#[inline]
fn circom_str_to_lambda_field_element(value: &str) -> FrElement {
    FrElement::from(&UnsignedInteger::<4>::from_dec_str(value).unwrap())
}
#[cfg(test)]
mod tests {
    use std::vec;

    use lambdaworks_groth16::qap;

    use super::*;

    #[test]
    fn test_qap_generation() {
        let subcircuit_r1cs_list = SubcircuitR1CS::from_path(8192, 30, "./subcircuits/r1cs").expect("read r1cs failed");
        let qap_list = SubcircuitQAP::from_r1cs(subcircuit_r1cs_list).expect("qap conversion failed");


        let add_subcircuit_qap = qap_list.get(2).expect("problem in qap"); 

    }

    


}
