//! cortex-cli — the black-box pipeline demo.
//!
//! ```text
//! cargo run -p cortex-core -- <text-file>
//! ```
//!
//! Reads a text file, simulates the DeepSeek-OCR encoder (`tid{n}` IDs),
//! runs the cortex pipeline (gate → TF-LUT → materialize), and simulates
//! the decoder to reconstruct the text.

use std::env;

use cortex_core::{CortexPipeline, SimCodec};

fn main() {
    let default_path =
        "/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-cortex/corpus/conversation.txt";
    let path = env::args().nth(1).unwrap_or_else(|| default_path.to_string());
    let text = std::fs::read_to_string(&path).expect("failed to read input file");

    // Simulated DeepSeek-OCR boundary: text ↔ opaque encoding IDs.
    let codec = SimCodec::from_text(&text);

    let mut pipeline = CortexPipeline::new();
    pipeline.set_gate(codec.vocab().iter().cloned());

    let ids = codec.encode_text(&text);
    let result = pipeline.process(&ids);
    let restored_text = codec.decode_ids(&result.restored_ids);

    println!("cortex-core — enhanced hllset-cortex pipeline");
    println!("  file           : {path}");
    println!("  words          : {}", ids.len());
    println!("  vocab (gate)   : {}", codec.vocab().len());
    println!("  doc bits       : {}", result.doc_bits);
    println!("  gated bits     : {}", result.gated_bits);
    println!("  restored ids   : {}", result.restored_ids.len());
    println!("  leaks          : {} {}", result.leaks.len(), if result.ok() { "✓" } else { "✗" });
    println!("  coverage       : {:.2}", result.coverage);
    println!("  LUT size       : {}", pipeline.lut_len());
    println!("  passes         : {}", pipeline.processed());
    println!();
    println!("  reconstructed (bag of words, TF-ranked):");
    let preview: String = restored_text.chars().take(200).collect();
    println!("  {preview}…");
}
