//! M1 exit-criterion integration test: the black-box pipeline round-trips
//! the conversation corpus with 0 leaks and coverage 1.0.

use cortex_core::{CortexPipeline, SimCodec};

#[test]
fn m1_roundtrip_on_conversation_corpus() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../corpus/conversation.txt"
    );
    let text = std::fs::read_to_string(path).expect("corpus file");

    // Simulated DeepSeek-OCR boundary.
    let codec = SimCodec::from_text(&text);

    let mut pipeline = CortexPipeline::new();
    pipeline.set_gate(codec.vocab().iter().cloned());

    let ids = codec.encode_text(&text);
    let result = pipeline.process(&ids);

    assert!(result.leaks.is_empty(), "M1: 0 out-of-vocab leaks");
    assert_eq!(result.coverage, 1.0, "M1: coverage invariant");
    assert_eq!(
        result.materialized_ids.len(),
        result.restored_ids.len(),
        "M1: output gate must not drop in-vocab ids"
    );
    assert_eq!(
        result.restored_ids.len() as u64,
        result.doc_bits,
        "M1: one token per resolved bit (collision groups disambiguated by TF)"
    );
}
