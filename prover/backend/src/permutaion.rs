use std::fs;

use lambdaworks_groth16::common::{FrElement, FrField};
use lambdaworks_math::{
    field::traits::{IsField},
    polynomial::Polynomial as UnivariatePolynomial,
    field::element::FieldElement,

};

use wasmer::{Store, Module, Instance, imports};

use serde::{Deserialize,Serialize};
use serde_json::{Value as JsonValue, from_str};
use crate::error::Error;

use crate::wasm::Wasm;

const wasm_dir :&str = "subcircuits/wasm";


// This struct represent the permutation map for inter subcircuit wires. 
// this function is useful to create B(Y,Z) which is necessary for copy constraint part. 
#[derive(Serialize, Deserialize, Debug)]
pub struct PermutationRule {
    #[serde(rename = "row")] 
    y: i32,
    #[serde(rename = "col")]
    z: i32,
    #[serde(rename = "Y")] 
    py: i32,
    #[serde(rename = "Z")]
    pz: i32,
}


// This struct is help full to find which instances called respectively, and what was the input and output. 
// Should import this to respected wasm files to generate witnesses. 
#[derive(Serialize, Deserialize,Debug)]
pub struct PlacementInstance {
    #[serde(rename = "placementIndex")]
    pub placement_index: i32,     // Maps to "placementIndex" in JSON
    #[serde(rename = "subcircuitId")]
    pub subcircuit_id: i32,       // Maps to "subcircuitId" in JSON
    #[serde(rename = "instructionName")] 
    pub instruction_name: String, // Maps to "instructionName" in JSON
    #[serde(rename = "inValues")] 
    pub in_values: Vec<String>,   // Maps to "inValues" in JSON
    #[serde(rename = "outValues")] 
    pub out_values: Vec<String>,  // Maps to "outValues" in JSON
}



// create a struct for aggregated QAP subcircuit OR Subcircuit Library 
// 
#[derive(Debug)]
pub struct SubcircuitLibraryInfo {
    // pub wasm : &mut Wasm,


    pub l : usize, 
    pub l_d :usize, 
    pub m_d : usize, 


    // to create d for each subcircuit we need to use this map 
    // we have m_d elements in wire list and each of them is a map 
    // like , wire_list[i] 0<i<m_d is (s_id,inner_id) which 0<=s_id<30 in our case
    // because we have 30 subcircuits is like this and 0<= inner_id <s_id.n_wires
    pub wire_list: Vec<(usize, usize)>,


    // to create d for each subcircuit we need to use this struct 
    // with flatten map object we can creat general d_i for 0<i<s_max
    pub subcircuit_infos : Vec<SubcircuitInfo>,

    
}
#[derive(Serialize, Deserialize,Debug)]
pub struct SubcircuitInfo {
    pub id: usize, 
    pub name : String, 
    pub n_wires : usize, 
    #[serde(rename = "Out_idx")]
    pub out_idx : Vec<usize> , 
    #[serde(rename = "In_idx")]
    pub in_idx : Vec<usize>, 
    // flatten map is gonna represent the map to library index 
    // flatten_map[i] , 0<i<n_wires and the output is 0<j<m_d which in our case is 41509
    #[serde(rename = "flattenMap")]
    pub flatten_map: Vec<usize>,

}

#[derive(Serialize, Deserialize,Debug)]
struct GlobalWire {
    pub l : usize, 
    pub l_d :usize, 
    pub m_d : usize, 
    #[serde(rename = "wireList")]
    pub wire_list: Vec<(usize, usize)>,
}




impl SubcircuitLibraryInfo { 

    pub fn new(global_wire_path :&str , subcircuit_info_path :&str) -> Result<Self,Error> {
        
        let global_wire_content = fs::read_to_string(global_wire_path)?;
        let global_wire : GlobalWire = from_str(&global_wire_content)?;   

        let subcircuit_wire_content = fs::read_to_string(global_wire_path)?;
        let subcircuit_infos : Vec<SubcircuitInfo> = from_str(&subcircuit_wire_content)?;   


        Ok(SubcircuitLibraryInfo{
            l: global_wire.l,
            l_d :global_wire.l_d,
            m_d :global_wire.m_d,
            wire_list :global_wire.wire_list,
            subcircuit_infos,
        })

    }


    pub fn create_dy(&self, witnesses :Vec<Vec<FrElement>> ,placement_instances :&[PlacementInstance])
    -> Result<Vec<UnivariatePolynomial<FrElement>> ,Error>
    {

        assert_eq!(witnesses.len() , placement_instances.len());
        // length of subcircuits used to create proof .
        let s_max = placement_instances.len(); // TODO :: check that is it necessary to be next_power_of_two? 

        // if we calculate fft for each inner vector we receive d_j(y), 0<=j<m_d
        let mut matrix = vec![vec![FrElement::zero(); s_max]; self.m_d];


        // in this function the placement instance should be imported and we create d_i(y) 0<i<m_d
        // from the placement we have input and output of jth subcircuit that is invoked . 0<j<s_max
        // each d_i(y) is a interpolation of a vector with lenght of s_max. 
        for i in 0..s_max{
            let subcircuit_id = placement_instances[i].subcircuit_id;
            for (j, witness) in witnesses[i].iter().enumerate(){
                // this map shows how to map a jth witness to flatten_map_id which is between 0<=flatten_map_id<m_d
                let flatten_map_id = self.subcircuit_infos[subcircuit_id as usize].flatten_map[j];
                matrix[flatten_map_id][i] = witness.clone();
            } 
        }

        
        // let dy = vec![FieldElement::<F>::zero(); self.m_d];
        let mut result = Vec::new();
        for i in 0..self.m_d {

            let poly = UnivariatePolynomial::interpolate_fft::<FrField>(&matrix[i])?;
            result.push(poly);
        }
        
        Ok(result)
        
    }
}

