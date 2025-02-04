
use clap::Error;
use lambdaworks_math::{
    field::traits::{IsField},
    polynomial::Polynomial as UnivariatePolynomial,
    field::element::FieldElement,

};

use wasmer::{Store, Module, Instance, Value, imports};

use serde::{Deserialize};

const wasm_dir :&str = "subcircuits/wasm";


// This struct represent the permutation map for inter subcircuit wires. 
// this function is useful to create B(Y,Z) which is necessary for copy constraint part. 
#[derive(Debug)]
pub struct PermutationRule {
    // #[serde(rename = "row")] 
    y: i32,
    // #[serde(rename = "col")]
    z: i32,
    // #[serde(rename = "Y")] 
    py: i32,
    // #[serde(rename = "Z")]
    pz: i32,
}


// This struct is help full to find which instances called respectively, and what was the input and output. 
// Should import this to respected wasm files to generate witnesses. 
#[derive(Debug)]
pub struct PlacementInstance {
    pub placement_index: i32,     // Maps to "placementIndex" in JSON
    pub subcircuit_id: i32,       // Maps to "subcircuitId" in JSON
    pub instruction_name: String, // Maps to "instructionName" in JSON
    pub in_values: Vec<String>,   // Maps to "inValues" in JSON
    pub out_values: Vec<String>,  // Maps to "outValues" in JSON
}



// create a struct for aggregated QAP subcircuit OR Subcircuit Library 
// 
#[derive(Debug)]
struct SubcircuitLibraryInfo {
    pub l : usize, 
    pub l_d :usize, 
    pub m_d : usize, 

    pub wire_list: Vec<(usize, usize)>,
    // pub 
    
}



impl SubcircuitLibraryInfo { 
    pub fn create_dy<F>(&self, placement_instances :&[PlacementInstance])
    -> Vec<UnivariatePolynomial<FieldElement<F>>> 
    where F: IsField
    {
        // in this function the placement instance should be imported and we create d_i(y) 0<i<m_d
        // from the placement we have input and output of jth subcircuit that is invoked . 0<j<s_max
        // each d_i(y) is a interpolation of a vector with lenght of s_max. 
        // let mut witnesses = Vec::new();
        for (i, placement) in placement_instances.iter().enumerate(){
            // call each wasm and  get respected witness.


        }
        // first I need to run the wasm of each placment to get the witnesses for each placement

        // let dy = vec![FieldElement::<F>::zero(); self.m_d];
        let mut result = Vec::new();
        for (idx, pi) in placement_instances.iter().enumerate() {
            // Process each placement instance here
            // e.g., create d_i(y) based on pi and append to result
            todo!()
        }
        
        result    }
}

pub fn call_wasm<F:IsField>(placement_instance :PlacementInstance) -> Result<Vec<FieldElement<F>>,Error> {

    let subcircuit_id = placement_instance.subcircuit_id; 
    let subcircuit_id_wasm = format!("{wasm_dir}/subcirctui{subcircuit_id}.wasm");
   
    let wasm_bytes = std::fs::read(subcircuit_id_wasm)?;

    // Create a Wasmer store
    let mut store = Store::default();

    let module = Module::new(&store, wasm_bytes).unwrap();

    // Create an import object (empty in this case, but you can add functions here if needed)
    let import_object = imports! {};

    // Instantiate the module
    let instance = Instance::new(&mut store, &module, &import_object).unwrap();

    // Now you can call functions from the Wasm module
    let result = instance.exports.get_function("your_function_name").unwrap().call(&mut store, &[]).unwrap();



    todo!()
}