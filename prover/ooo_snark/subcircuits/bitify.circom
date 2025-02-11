pragma circom 2.0.0;

template Bitify32() {
    signal input in;          // Input signal (assumed to be in the range 0 to 2^32 - 1)
    signal output bits[32];   // Output array of 32 bits

    // Decompose the input into 32 bits
    for (var i = 0; i < 32; i++) {
        bits[i] <-- (in >> i) & 1;  // Extract the i-th bit
        bits[i] * (bits[i] - 1) === 0; // Constrain bits[i] to be binary (0 or 1)
    }

    // Reconstruct the input from the bits to ensure correctness
    var reconstructed = 0;
    for (var i = 0; i < 32; i++) {
        reconstructed += bits[i] * (1 << i);
    }
    reconstructed === in; // Constrain the reconstructed value to match the input
}

// component main = Bitify32();