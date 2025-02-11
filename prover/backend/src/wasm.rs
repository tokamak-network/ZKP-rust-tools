use lambdaworks_groth16::common::FrElement;
use lambdaworks_math::traits::ByteConversion;
use wasmer::{imports, Value, Exports, Function, Instance, Memory, MemoryType, Module, RuntimeError, Store};

use crate::permutaion::PlacementInstance;
use crate::error::Error;
use std::path::PathBuf;

use crate::fnv::fnv;


use std;
use wasmer_wasix::WasiEnv;

// use thiserror::Error;


// pub type Result<T> = core::result::Result<T,Error>;

#[derive(Debug)]
pub struct Wasm {
    pub store :Store,
    pub subcircuits_insts :Vec<SubcircuitWasmInst>,
}

#[derive(Debug)]
pub struct SubcircuitWasmInst {
    pub exports: Exports, 
    pub memory: Memory,
}

impl SubcircuitWasmInst {
    fn func(&self, name: &str) -> &Function {
        self.exports
            .get_function(name)
            .unwrap_or_else(|_| panic!("function {} not found", name))
    }

    fn init(&self, store: &mut Store, sanity_check: bool) -> Result<(),Error> {
        let func = self.func("init");
        func.call(store, &[Value::I32(sanity_check as i32)])?;
        Ok(())
    }
    fn get_field_num_len32(&self, store: &mut Store) -> Result<u32,Error> {
        self.get_u32(store, "getFieldNumLen32")
    }
    fn get_u32(&self, store: &mut Store, name: &str) -> Result<u32,Error> {
        let func = &self.func(name);
        let result = func.call(store, &[])?;
        Ok(result[0].unwrap_i32() as u32)
    }
    fn get_n_vars(&self, store: &mut Store) -> Result<u32,Error> {
        self.get_u32(store, "getNVars")
    }

    fn read_shared_rw_memory(&self, store: &mut Store, i: u32) -> Result<u32,Error> {
        let func = self.func("readSharedRWMemory");
        let result = func.call(store, &[i.into()])?;
        Ok(result[0].unwrap_i32() as u32)
    }

    fn write_shared_rw_memory(&self, store: &mut Store, i: u32, v: u32) -> Result<(),Error> {
        let func = self.func("writeSharedRWMemory");
        func.call(store, &[i.into(), v.into()])?;
        Ok(())
    }

    fn set_input_signal(&self, store: &mut Store, hmsb: u32, hlsb: u32, pos: u32) -> Result<(),Error> {
        let func = self.func("setInputSignal");
        func.call(store, &[hmsb.into(), hlsb.into(), pos.into()])?;
        Ok(())
    }
    fn get_witness_size(&self, store: &mut Store) -> Result<u32,Error> {
        self.get_u32(store, "getWitnessSize")
    }
    fn get_witness(&self, store: &mut Store, i: u32) -> Result<(),Error> {
        let func = self.func("getWitness");
        func.call(store, &[i.into()])?;
        Ok(())
    }




}

pub fn fr_element_to_u32(fr : &FrElement)-> Vec<u32> {
    bytes_to_u32_le(&fr.to_bytes_le())
}

fn fr_element_from_array_u32(arr: &Vec<u32>) -> Result<FrElement,Error> {
    let element = FrElement::from_bytes_le(&u32_to_bytes_le(&arr))?;
    Ok(element)
}


fn bytes_to_u32_le(bytes: &[u8]) -> Vec<u32> {
    // Ensure the byte slice length is a multiple of 4 (since 4 bytes = 1 u32)
    assert!(bytes.len() % 4 == 0, "Byte slice length must be a multiple of 4");

    // Convert the slice of u8 to a slice of u32 in little-endian order
    bytes
        .chunks_exact(4) // Split into chunks of 4 bytes
        .map(|chunk| {
            u32::from_le_bytes(chunk.try_into().unwrap()) // Convert each chunk to u32
        })
        .collect() // Collect into a Vec<u32>
}

fn u32_to_bytes_le(u32_vec: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(u32_vec.len() * 4); // Pre-allocate space for efficiency

    for &value in u32_vec {
        // Convert each u32 to its little-endian byte representation
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes
}


impl Wasm {
    pub fn new(subcircuit_cnt :usize ,dir: impl AsRef<std::path::Path>) -> Result<Self,Error> {
        let mut store = Store::default();
        let mut subcircuits_insts =  Vec::with_capacity(subcircuit_cnt);
        

        for subcircuit_id in 0..subcircuit_cnt{
            let module = Module::from_file(&store, dir.as_ref().join(format!("subcircuit{}.wasm", subcircuit_id)))?;
            let (export, memory) = Wasm::make_wasm_runtime(&mut store, module)?;


            subcircuits_insts.push(SubcircuitWasmInst{exports: export, memory: memory});
        }


        Ok(Wasm { store: store, subcircuits_insts: subcircuits_insts })
    }

    pub fn make_wasm_runtime(store: &mut Store, module: Module) -> Result<(Exports, Memory),Error> {
        let memory = Memory::new(store, MemoryType::new(2000, None, false)).unwrap();
        let import_object = imports! {
            "env" => {
                "memory" => memory.clone(),
            },
            // Host function callbacks from the WASM
            "runtime" => {
                "error" => runtime::error(store),
                "logSetSignal" => runtime::log_signal(store),
                "logGetSignal" => runtime::log_signal(store),
                "logFinishComponent" => runtime::log_component(store),
                "logStartComponent" => runtime::log_component(store),
                "log" => runtime::log_component(store),
                "exceptionHandler" => runtime::exception_handler(store),
                "showSharedRWMemory" => runtime::show_memory(store),
                "printErrorMessage" => runtime::print_error_message(store),
                "writeBufferMessage" => runtime::write_buffer_message(store),
            }
        };
        let instance = Instance::new(store, &module, &import_object)?;
        let exports = instance.exports.clone();
        let mut wasi_env = WasiEnv::builder("calculateWitness").finalize(store)?;
        wasi_env.initialize_with_memory(store, instance, Some(memory.clone()), false)?;
        Ok((exports,memory))
    }



    pub fn calculate_witness(&mut self,placements :&[PlacementInstance]) -> Result<Vec<Vec<FrElement>> ,Error >
    {   
        let n32 = self.subcircuits_insts[0].get_field_num_len32(&mut self.store)?;
        let mut result = Vec::new();
        let (msb_id, lsb_id) = fnv("in");
        for placement in placements {

            self.subcircuits_insts[placement.subcircuit_id as usize].init(&mut self.store, true)?;
            for (pos,in_values) in placement.in_values.iter().enumerate() {
                let input = FrElement::from_hex(&in_values)?;
                for (i , u_32_i) in  fr_element_to_u32(&input).into_iter().enumerate() {
                    self.subcircuits_insts[placement.subcircuit_id as usize].write_shared_rw_memory(
                        &mut self.store,
                        i as u32 ,
                        u_32_i,
                    )?;
                }
                self.subcircuits_insts[placement.subcircuit_id as usize].set_input_signal(&mut self.store, msb_id, lsb_id, pos as u32)?;
            }
            let mut w = Vec::new();
            let witness_size = self.subcircuits_insts[placement.subcircuit_id as usize].get_witness_size(&mut self.store )?;
            for i in 0..witness_size {
                self.subcircuits_insts[placement.subcircuit_id as usize].get_witness(&mut self.store, i)?;
                let mut arr = vec![0; n32 as usize];
                for j in 0..n32 {
                    arr[(n32 as usize) - 1 - (j as usize)] =
                    self.subcircuits_insts[placement.subcircuit_id as usize].read_shared_rw_memory(&mut self.store, j)?;
                }
                let element = fr_element_from_array_u32(&arr)?;
                w.push(element);
            }
            result.push(w);
        }

        Ok(result)
    }
}


// Error type to signal end of execution.
// From https://docs.wasmer.io/integrations/examples/exit-early
#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("{0}")]
struct ExitCode(u32);


// callback hooks for debugging
mod runtime {
    use super::*;

    pub fn error(store: &mut Store) -> Function {
        #[allow(unused)]
        #[allow(clippy::many_single_char_names)]
        fn func(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> Result<(), RuntimeError> {
            // NOTE: We can also get more information why it is failing, see p2str etc here:
            // https://github.com/iden3/circom_runtime/blob/master/js/witness_calculator.js#L52-L64
            println!("runtime error, exiting early: {a} {b} {c} {d} {e} {f}",);
            Err(RuntimeError::user(Box::new(ExitCode(1))))
        }
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn exception_handler(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32) {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn show_memory(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn print_error_message(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn write_buffer_message(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    pub fn log_signal(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32, b: i32) {}
        Function::new_typed(store, func)
    }

    pub fn log_component(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32) {}
        Function::new_typed(store, func)
    }
}


#[cfg(test)]
mod tests {
    use std::vec;
    use super:: * ;

    #[tokio::test]
    async fn test_wasm() {
        let wasm = Wasm::new(30, "./subcircuits/wasm").unwrap();
        // wasm.calculate_witness(placements)
    }


    #[test]
    fn test_bytes_to_u32_le() {
        let bytes: Vec<u8> = vec![0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]; // Example little-endian bytes
        let u32_values = bytes_to_u32_le(&bytes);
        assert_eq!(u32_values[0], 1);
        assert_eq!(u32_values[1], 2);

        // println!("{:?}", u32_values); // Output: [1, 2]
    }


    #[test]
    fn test_u32_to_bytes_le() {
        let u32_values = vec![0x12345678, 0xAABBCCDD, 0xDEADBEEF];

        // Convert u32 values to bytes
        let bytes = u32_to_bytes_le(&u32_values);
        
        // Convert the bytes back to u32 values to verify
        let reconstructed_u32_values = bytes_to_u32_le(&bytes);

        assert_eq!(reconstructed_u32_values, u32_values);
    }
}