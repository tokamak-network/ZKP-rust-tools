# Define the range of IDs (0 to 29)
IDS := $(shell seq 0 29)

# Define the input and output file patterns
INPUT_FILES := $(addprefix subcircuits/r1cs/subcircuit,$(addsuffix .r1cs,$(IDS)))
OUTPUT_FILES := $(addprefix subcircuits/r1cs/json/subcircuit,$(addsuffix .r1cs.json,$(IDS)))

# Ensure the output directory exists
$(shell mkdir -p subcircuits/r1cs/json)

# Default target
all: $(OUTPUT_FILES)

# Rule to convert .r1cs to .r1cs.json and place it in the "files" directory
subcircuits/r1cs/json/subcircuit%.r1cs.json: subcircuits/r1cs/subcircuit%.r1cs
	snarkjs r1cs export json $< $@

# Clean up generated JSON files in the "files" directory
clean:
	rm -f $(OUTPUT_FILES)

.PHONY: all clean
