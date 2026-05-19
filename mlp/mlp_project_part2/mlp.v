//// MLP Inference (Sequential Version) - Fixed Matrix Indexing
//// Precision: 16-bit Fixed-point (Q8.8)

//module mlp_inference (
//    input clk,
//    input reset,
//    input start,
//    input [9:0] sample_idx,
//    output reg [3:0] predicted_class,
//    output reg done
//);

//    parameter INPUT_SIZE  = 784;
//    parameter HIDDEN_SIZE = 128;
//    parameter OUTPUT_SIZE = 10;

//    // Buffer for memories
//    reg signed [15:0] x_input [0:500*INPUT_SIZE-1]; 
//    reg signed [15:0] w1 [0:INPUT_SIZE*HIDDEN_SIZE-1];
//    reg signed [15:0] b1 [0:HIDDEN_SIZE-1];
//    reg signed [15:0] w2 [0:HIDDEN_SIZE*OUTPUT_SIZE-1];
//    reg signed [15:0] b2 [0:OUTPUT_SIZE-1];
    
//    reg signed [15:0] hidden_layer [0:HIDDEN_SIZE-1];
//    reg signed [15:0] output_layer [0:OUTPUT_SIZE-1];

//    // FSM States
//    parameter IDLE        = 3'b000;
//    parameter CALC_HIDDEN = 3'b001;
//    parameter CALC_OUTPUT = 3'b010;
//    parameter FIND_MAX    = 3'b011;
//    parameter DONE_STATE  = 3'b100;

//    reg [2:0] state;
//    reg [9:0] i_cnt; 
//    reg [7:0] j_cnt; 
//    reg signed [31:0] acc;

//    // Load Hex data
//    initial begin
//        $readmemh("x_test.mem", x_input);
//        $readmemh("W1.mem", w1);
//        $readmemh("b1.mem", b1);
//        $readmemh("W2.mem", w2);
//        $readmemh("b2.mem", b2);
//    end

//    always @(posedge clk or posedge reset) begin
//        if (reset) begin
//            state <= IDLE;
//            done <= 0;
//            i_cnt <= 0;
//            j_cnt <= 0;
//            acc <= 0;
//            predicted_class <= 0;
//        end else begin
//            case (state)
//                IDLE: begin
//                    done <= 0;
//                    if (start) begin
//                        state <= CALC_HIDDEN;
//                        i_cnt <= 0;
//                        j_cnt <= 0;
//                        acc <= b1[0] << 8; 
//                    end
//                end


//                CALC_HIDDEN: begin
//                    acc <= acc + (x_input[sample_idx * INPUT_SIZE + i_cnt] * w1[i_cnt * HIDDEN_SIZE + j_cnt]);
                    
//                    if (i_cnt == INPUT_SIZE - 1) begin
//                        i_cnt <= 0;
                       
//                        hidden_layer[j_cnt] <= (acc[31] == 1'b0) ? acc[23:8] : 16'd0;
                        
//                        if (j_cnt == HIDDEN_SIZE - 1) begin
//                            j_cnt <= 0;
//                            state <= CALC_OUTPUT;
//                            acc <= b2[0] << 8; 
//                        end else begin
//                            j_cnt <= j_cnt + 1;
//                            acc <= b1[j_cnt + 1] << 8; 
//                        end
//                    end else begin
//                        i_cnt <= i_cnt + 1;
//                    end
//                end

              
//                CALC_OUTPUT: begin
                   
//                    acc <= acc + (hidden_layer[i_cnt[7:0]] * w2[i_cnt[7:0] * OUTPUT_SIZE + j_cnt]);
                    
//                    if (i_cnt[7:0] == HIDDEN_SIZE - 1) begin
//                        i_cnt <= 0;
//                        output_layer[j_cnt] <= acc[23:8]; 
                        
//                        if (j_cnt == OUTPUT_SIZE - 1) begin
//                            state <= FIND_MAX;
//                            j_cnt <= 0;
//                            i_cnt <= 0;
//                        end else begin
//                            j_cnt <= j_cnt + 1;
//                            acc <= b2[j_cnt + 1] << 8; 
//                        end
//                    end else begin
//                        i_cnt <= i_cnt + 1;
//                    end
//                end

//                FIND_MAX: begin
//                    if (i_cnt == 0) begin
//                        predicted_class <= 0;
//                        i_cnt <= 1;
//                    end else if (i_cnt < OUTPUT_SIZE) begin
//                        if ($signed(output_layer[i_cnt[3:0]]) > $signed(output_layer[predicted_class])) begin
//                            predicted_class <= i_cnt[3:0];
//                        end
//                        i_cnt <= i_cnt + 1;
//                    end else begin
//                        state <= DONE_STATE;
//                    end
//                end

//                DONE_STATE: begin
//                    done <= 1'b1;
//                    if (!start) state <= IDLE;
//                end
                
//                default: state <= IDLE;
//            endcase
//        end
//    end
//endmodule








// MLP Inference (Sequential Version) - Fixed Matrix Indexing
// Precision: 16-bit Fixed-point (Q8.8)

`timescale 1ns / 1ps

module mlp_inference (
    input clk,                          // System clock (Design heart beat)
    input reset,                        // Synchronous active-high reset signal
    input start,                        // Trigger signal to start hardware inference
    input [9:0] sample_idx,             // Index of the test image to process (0 to 499)
    output reg [3:0] predicted_class,   // 4-bit output indicating the predicted digit (0-9)
    output reg done                     // High when inference process is completed
);

    // Network Architecture Parameters
    parameter INPUT_SIZE  = 784;        // 28x28 flattened image pixels
    parameter HIDDEN_SIZE = 128;        // Neurons in the hidden layer
    parameter OUTPUT_SIZE = 10;         // Output classes (Digits 0-9)

    // Hardware Memory Buffers (Inferred as Block RAMs or Distributed RAM)
    reg signed [15:0] x_input [0:500*INPUT_SIZE-1];     // Storage for 500 test images
    reg signed [15:0] w1 [0:INPUT_SIZE*HIDDEN_SIZE-1];  // Layer 1 flattened weight matrix
    reg signed [15:0] b1 [0:HIDDEN_SIZE-1];             // Layer 1 bias vector
    reg signed [15:0] w2 [0:HIDDEN_SIZE*OUTPUT_SIZE-1]; // Layer 2 flattened weight matrix
    reg signed [15:0] b2 [0:OUTPUT_SIZE-1];             // Layer 2 bias vector
    
    // Intermediate Layer Registers
    reg signed [15:0] hidden_layer [0:HIDDEN_SIZE-1];   // Buffers outputs of Layer 1 (after ReLU)
    reg signed [15:0] output_layer [0:OUTPUT_SIZE-1];   // Buffers raw outputs of Layer 2 (Logits)

    // Finite State Machine (FSM) States Definition
    parameter IDLE        = 3'b000;     // Waiting for start trigger
    parameter CALC_HIDDEN = 3'b001;     // Computing Layer 1 matrix multiplication & ReLU
    parameter CALC_OUTPUT = 3'b010;     // Computing Layer 2 matrix multiplication
    parameter FIND_MAX    = 3'b011;     // Hardwired Argmax operation to find highest logit
    parameter DONE_STATE  = 3'b100;     // Assertion of done flag and waiting for handshake

    // Control and Internal Registers
    reg [2:0] state;                    // Current FSM state register
    reg [9:0] i_cnt;                    // Row/Input loop counter (counts up to 784)
    reg [7:0] j_cnt;                    // Column/Hidden loop counter (counts up to 128)
    reg signed [31:0] acc;              // 32-bit Accumulator to prevent overflow during MAC operations

    // Initial Block for Loading Pre-trained Weights and Test Data
    initial begin
        $readmemh("x_test.mem", x_input); // Load hex pixel data generated by Python
        $readmemh("W1.mem", w1);          // Load Layer 1 Q8.8 quantized weights
        $readmemh("b1.mem", b1);          // Load Layer 1 Q8.8 quantized biases
        $readmemh("W2.mem", w2);          // Load Layer 2 Q8.8 quantized weights
        $readmemh("b2.mem", b2);          // Load Layer 2 Q8.8 quantized biases
    end

    // Main Sequential Control Logic Block
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset all internal states, flags, counters, and registers
            state <= IDLE;
            done <= 0;
            i_cnt <= 0;
            j_cnt <= 0;
            acc <= 0;
            predicted_class <= 0;
        end else begin
            case (state)
                
                // IDLE State: System rests here until "start" signal is caught
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= CALC_HIDDEN;
                        i_cnt <= 0;
                        j_cnt <= 0;
                        // Pre-load accumulator with the first bias shifted by 8 bits
                        // This aligns the Q8.8 bias with the 32-bit (Q16.16) multiplication results
                        acc <= b1[0] << 8; 
                    end
                end

                // CALC_HIDDEN State: Computes Hidden Layer activation values sequentially
                CALC_HIDDEN: begin
                    // Multiply-Accumulate (MAC) Step: Image pixel * Weight + Accumulator
                    acc <= acc + (x_input[sample_idx * INPUT_SIZE + i_cnt] * w1[i_cnt * HIDDEN_SIZE + j_cnt]);
                    
                    // Check if all 784 inputs for the current hidden neuron have been accumulated
                    if (i_cnt == INPUT_SIZE - 1) begin
                        i_cnt <= 0;
                        
                        // Hardware ReLU Activation Function & Rescaling (Quantization Downshift)
                        // If MSB (acc[31]) is 0, the number is positive -> extract middle 16 bits [23:8] to get Q8.8 format
                        // If MSB is 1, the number is negative -> clamp to 16'd0 (ReLU)
                        hidden_layer[j_cnt] <= (acc[31] == 1'b0) ? acc[23:8] : 16'd0;
                        
                        // Check if all 128 hidden layer neurons are calculated
                        if (j_cnt == HIDDEN_SIZE - 1) begin
                            j_cnt <= 0;
                            state <= CALC_OUTPUT;           // Move to the next layer
                            acc <= b2[0] << 8;              // Pre-load the first bias of Layer 2
                        end else begin
                            j_cnt <= j_cnt + 1;             // Move to the next hidden neuron
                            acc <= b1[j_cnt + 1] << 8;      // Pre-load next hidden neuron's bias
                        end
                    end else begin
                        i_cnt <= i_cnt + 1;                 // Increment input pixel counter
                    end
                end

                // CALC_OUTPUT State: Computes Final Layer Raw Scores (Logits)
                CALC_OUTPUT: begin
                    // MAC Step: Hidden layer activation * Layer 2 weight + Accumulator
                    acc <= acc + (hidden_layer[i_cnt[7:0]] * w2[i_cnt[7:0] * OUTPUT_SIZE + j_cnt]);
                    
                    // Check if all 128 hidden neuron connections for the current output neuron are accumulated
                    if (i_cnt[7:0] == HIDDEN_SIZE - 1) begin
                        i_cnt <= 0;
                        // Rescale 32-bit accumulation result back to 16-bit Q8.8 format
                        // No ReLU is applied here because output logits can be negative
                        output_layer[j_cnt] <= acc[23:8]; 
                        
                        // Check if all 10 output classes (digits 0-9) are calculated
                        if (j_cnt == OUTPUT_SIZE - 1) begin
                            state <= FIND_MAX;              // Proceed to Argmax stage
                            j_cnt <= 0;
                            i_cnt <= 0;
                        end else begin
                            j_cnt <= j_cnt + 1;             // Move to the next output neuron
                            acc <= b2[j_cnt + 1] << 8;      // Pre-load next output neuron's bias
                        end
                    end else begin
                        i_cnt <= i_cnt + 1;                 // Increment hidden layer loop counter
                    end
                end

                // FIND_MAX State: Hardware Argmax Implementation
                FIND_MAX: begin
                    if (i_cnt == 0) begin
                        // Initialize baseline: assume class 0 has the highest value
                        predicted_class <= 0;
                        i_cnt <= 1;
                    end else if (i_cnt < OUTPUT_SIZE) begin
                        // Signed comparison to safely handle negative values
                        // If current class score is greater than the recorded maximum, update index
                        if ($signed(output_layer[i_cnt[3:0]]) > $signed(output_layer[predicted_class])) begin
                            predicted_class <= i_cnt[3:0];  // Capture the index of the highest score
                        end
                        i_cnt <= i_cnt + 1;                 // Move to the next class index
                    end else begin
                        state <= DONE_STATE;                // All comparisons complete
                    end
                end

                // DONE_STATE: Asserts execution completion flag
                DONE_STATE: begin
                    done <= 1'b1;                           // Broadcast completion signal
                    if (!start) state <= IDLE;              // Handshake: return to IDLE once start drops
                end
                
                // Safety fallback to prevent latch generation or stuck state machine
                default: state <= IDLE;
            endcase
        end
    end
endmodule