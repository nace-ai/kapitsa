//! Explicit real-model lifecycle check; never part of the CPU test run.
#[cfg(not(feature = "fixture"))]
#[test]
#[ignore = "loads a local model and holds the native GPU lock; requires explicit authorization"]
fn model_lifecycle() {
    use kapitsa::{Engine, ErrorKind};
    use std::sync::mpsc;
    use std::thread;
    let options = std::env::var("KAPITSA_TEST_OPTIONS")
        .expect("set JSON with model_dir, slots=2, spec_depth=0 and dflash=none");
    let engine = Engine::new(&options).unwrap();
    let input = r#"{"messages":[{"role":"user","content":"Count from one to twenty."}],"max_tokens":32,"temperature":0,"seed":1}"#;
    let mut request = engine.request(input).unwrap();
    let mut nested = engine.request(input).unwrap();
    let mut deltas = 0;
    let result = request
        .generate_with(|delta| {
            assert!(!delta.is_empty());
            deltas += 1;
            assert_eq!(nested.generate().unwrap_err().kind, ErrorKind::Busy);
            true
        })
        .unwrap();
    assert!(deltas > 0 && result.contains("completion_tokens") && result.contains("content"));
    assert_eq!(request.generate().unwrap_err().kind, ErrorKind::Busy);
    drop(request);
    assert!(nested.generate().is_ok());
    drop(nested);
    let mut request = engine.request(input).unwrap();
    request.cancellation().cancel();
    let err = request.generate().unwrap_err();
    assert_eq!(err.kind, ErrorKind::Cancelled);
    assert!(err.partial_result.is_none());
    drop(request);
    let mut request = engine.request(input).unwrap();
    assert_eq!(
        request.generate_with(|_| false).unwrap_err().kind,
        ErrorKind::Cancelled
    );
    drop(request);
    let mut a = engine.request(input).unwrap();
    let mut b = engine.request(input).unwrap();
    let cancel = a.cancellation();
    let (tx, rx) = mpsc::channel();
    let (ack_tx, ack_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let mut first = true;
        a.generate_with(|_| {
            if first {
                first = false;
                tx.send(()).unwrap();
                ack_rx.recv().unwrap();
            }
            true
        })
    });
    rx.recv().unwrap();
    cancel.cancel();
    ack_tx.send(()).unwrap();
    let b_result = b.generate().unwrap();
    assert!(b_result.contains("completion_tokens"));
    assert_eq!(
        worker.join().unwrap().unwrap_err().kind,
        ErrorKind::Cancelled
    );
    drop(cancel);
    drop(b);
    let mut request = engine.request(input).unwrap();
    let cancel = request.cancellation();
    drop(engine);
    assert!(request.generate().is_ok());
    drop(request);
    drop(cancel);
    // Creation checks that all native ownership and the global lock were released.
    drop(Engine::new(&options).unwrap());
}
