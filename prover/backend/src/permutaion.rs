


#[derive(Debug, Deserialize)]
struct PermutationRule {
    #[serde(rename = "row")] 
    y: i32,
    #[serde(rename = "col")]
    z: i32,
    #[serde(rename = "Y")] 
    py: i32,
    #[serde(rename = "Z")]
    pz: i32,
}



#[derive(Debug, Deserialize)]
struct PlacementInstance {
    placement_index: i32,     // Maps to "placementIndex" in JSON
    subcircuit_id: i32,       // Maps to "subcircuitId" in JSON
    instruction_name: String, // Maps to "instructionName" in JSON
    in_values: Vec<String>,   // Maps to "inValues" in JSON
    out_values: Vec<String>,  // Maps to "outValues" in JSON
}