pragma circom 2.1.6;
include "divider.circom";

template Adder32 () {
    signal input in[2], carry_in;
    var NUM_BITS = 32;

    component divider = Divider(NUM_BITS);
    divider.in <== in[0] + in[1] + carry_in;

    signal output carry_out <== divider.q;
    signal output sum <== divider.r;
}

template Divider (n) {
    //add assert
    signal input in;

    var divisor = 2**n;

    signal output r <-- in % divisor;
    signal output q <-- in \ divisor;

    in === q * divisor + r;

    // Ensure r < divisor;
    component lt_divisor = LessThan(n);
    lt_divisor.in[0] <== r;
    lt_divisor.in[1] <== divisor;

    lt_divisor.out === 1;
}

template LessThan(n) {
    assert(n <= 252);
    signal input in[2];
    signal output out;

    component n2b = Num2Bits(n+1);

    n2b.in <== in[0]+ (1<<n) - in[1];

    out <== 1-n2b.out[n];
}



template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc1=0;

    var e2=1;
    for (var i = 0; i<n; i++) {
        out[i] <-- (in >> i) & 1; // there in no sign . 
        out[i] * (out[i] -1 ) === 0; // row in r1cs
        lc1 += out[i] * e2;
        e2 = e2+e2;
    }

    lc1 === in;
}
