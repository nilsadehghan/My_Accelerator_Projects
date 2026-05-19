//`timescale 1ns / 1ps

//module mlp_tb();
//    reg clk;
//    reg reset;
//    reg start;
//    reg [9:0] sample_idx;
//    wire [3:0] predicted_class;
//    wire done;

//    reg [3:0] y_test [0:499];
//    integer i;
//    integer correct_count = 0;
//    integer total_samples = 500; 
//    real accuracy;

//    mlp_inference uut (
//        .clk(clk), .reset(reset), .start(start),
//        .sample_idx(sample_idx),
//        .predicted_class(predicted_class), .done(done)
//    );

   
//    initial begin
//        clk = 0;
//        forever #5 clk = ~clk; 
//    end

//    initial begin
      
//        $readmemh("y_test.mem", y_test);
        
//        reset = 1; start = 0; sample_idx = 0;
//        #100 reset = 0;
        
//        $display("\n=======================================");
//        $display("STARTING HARDWARE INFERENCE TEST...");
//        $display("Testing first %d samples", total_samples);
//        $display("=======================================\n");

//        for (i = 0; i < total_samples; i = i + 1) begin
//            sample_idx = i;
//            #20 start = 1;
//            #20 start = 0;
            
//            wait(done); 
            
//            if (predicted_class == y_test[i]) begin
//                correct_count = correct_count + 1;
//                $display("[OK] Sample %d: Pred=%d, Actual=%d", i, predicted_class, y_test[i]);
//            end else begin
//                $display("[WRONG] Sample %d: Pred=%d, Actual=%d", i, predicted_class, y_test[i]);
//            end
            
//            #100; 
//        end

//        accuracy = (correct_count * 100.0) / total_samples;
        
//        $display("\n=======================================");
//        $display("       FINAL HARDWARE RESULTS          ");
//        $display("=======================================");
//        $display("Total Samples    : %d", total_samples);
//        $display("Correct Matches  : %d", correct_count);
//        $display("Hardware Accuracy: %f %%", accuracy);
//        $display("=======================================\n");
        
//        $finish;
//    end
//endmodule



`timescale 1ns / 1ps

module mlp_tb();
    reg clk;
    reg reset;
    reg start;
    reg [9:0] sample_idx;
    wire [3:0] predicted_class;
    wire done;

    reg [3:0] y_test [0:499];
    integer i;
    integer correct_count = 0;
    integer total_samples = 500; 
    integer fp;   // Added for file writing
    real accuracy;

    mlp_inference uut (
        .clk(clk), .reset(reset), .start(start),
        .sample_idx(sample_idx),
        .predicted_class(predicted_class), .done(done)
    );

   
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    initial begin
      
        $readmemh("y_test.mem", y_test);

        // Open output file
        fp = $fopen("hardware_predictions.txt", "w");
        $fwrite(fp, "Predicted True\n");
        
        reset = 1; start = 0; sample_idx = 0;
        #100 reset = 0;
        
        $display("\n=======================================");
        $display("STARTING HARDWARE INFERENCE TEST...");
        $display("Testing first %d samples", total_samples);
        $display("=======================================\n");

        for (i = 0; i < total_samples; i = i + 1) begin
            sample_idx = i;
            #20 start = 1;
            #20 start = 0;
            
            wait(done); 

            // Save prediction to file
            $fwrite(fp, "%d %d\n", predicted_class, y_test[i]);
            
            if (predicted_class == y_test[i]) begin
                correct_count = correct_count + 1;
                $display("[OK] Sample %d: Pred=%d, Actual=%d", i, predicted_class, y_test[i]);
            end else begin
                $display("[WRONG] Sample %d: Pred=%d, Actual=%d", i, predicted_class, y_test[i]);
            end
            
            #100; 
        end

        // Close file
        $fclose(fp);

        accuracy = (correct_count * 100.0) / total_samples;
        
        $display("\n=======================================");
        $display("       FINAL HARDWARE RESULTS          ");
        $display("=======================================");
        $display("Total Samples    : %d", total_samples);
        $display("Correct Matches  : %d", correct_count);
        $display("Hardware Accuracy: %f %%", accuracy);
        $display("=======================================\n");
        
        $finish;
    end
endmodule

