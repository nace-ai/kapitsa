use kapitsa::{Engine, ErrorKind};

#[test]
fn inputs_reject_nul_before_entering_native_code() {
    assert!(matches!(Engine::new("{}\0"), Err(e) if e.kind == ErrorKind::InvalidArgument));
}

#[cfg(not(feature = "fixture"))]
#[test]
fn native_invalid_json_needs_no_model_or_gpu() {
    for json in ["{", "{}", r#"{"model_dir":false}"#] {
        assert!(matches!(Engine::new(json), Err(e) if e.kind == ErrorKind::InvalidArgument));
    }
}

#[cfg(feature = "fixture")]
#[test]
fn lifecycle_through_c_abi() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::mpsc;
    use std::thread;
    extern "C" {
        fn fixture_live() -> i32;
    }
    const NORMAL: &str = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let live = || unsafe { fixture_live() };
    assert_eq!(live(), 0);
    assert!(matches!(Engine::new(r#"{"model_dir":"bad"}"#), Err(e) if e.kind == ErrorKind::Native));
    assert_eq!(live(), 0);
    let engine = Engine::new("{}").unwrap();
    assert!(matches!(Engine::new("{}"), Err(e) if e.kind == ErrorKind::Busy));
    assert!(matches!(engine.request("{}"), Err(e) if e.kind == ErrorKind::InvalidArgument));
    assert!(matches!(engine.request("\0"), Err(e) if e.kind == ErrorKind::InvalidArgument));
    {
        let mut r = engine.request(NORMAL).unwrap();
        let mut count = 0;
        let result = r
            .generate_with(|delta| {
                assert!(delta.contains("猫🐈") && delta.contains("北京"));
                count += 1;
                true
            })
            .unwrap();
        assert_eq!(count, 6);
        assert!(result.contains("猫🐈猫🐈猫🐈猫🐈猫🐈猫🐈"));
        assert!(result.contains(r#""finish_reason":"stop""#));
        assert_eq!(r.generate().unwrap_err().kind, ErrorKind::Busy);
    }
    {
        let mut r = engine.request(NORMAL).unwrap();
        r.cancellation().cancel();
        let e = r.generate().unwrap_err();
        assert_eq!(e.kind, ErrorKind::Cancelled);
        assert!(e.partial_result.is_none());
    }
    {
        let mut r = engine.request(NORMAL).unwrap();
        let e = r.generate_with(|_| false).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Cancelled);
        assert!(e.partial_result.unwrap().contains("猫🐈"));
    }
    {
        let mut r = engine.request(NORMAL).unwrap();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            r.generate_with(|_| panic!("callback panic"))
        }));
        assert!(panic.is_err());
        assert_eq!(r.generate().unwrap_err().kind, ErrorKind::Busy);
    }
    {
        let mut outer = engine.request(NORMAL).unwrap();
        let mut inner = engine.request(NORMAL).unwrap();
        let mut calls = 0;
        outer
            .generate_with(|_| {
                calls += 1;
                assert_eq!(inner.generate().unwrap_err().kind, ErrorKind::Busy);
                true
            })
            .unwrap();
        assert!(calls > 0);
        assert!(inner.generate().is_ok()); // rejection did not consume the request
    }
    assert_eq!(live(), 1); // no output strings or requests leaked on any error path
    {
        let mut r = engine.request(NORMAL).unwrap();
        let cancel = r.cancellation();
        let (tx, rx) = mpsc::channel();
        let (ack_tx, ack_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut first = true;
            r.generate_with(|_| {
                if first {
                    first = false;
                    tx.send(()).unwrap();
                    ack_rx.recv().unwrap();
                }
                true
            })
        });
        rx.recv().unwrap(); // cancellation actually races a live generation
        cancel.cancel();
        ack_tx.send(()).unwrap();
        assert_eq!(
            worker.join().unwrap().unwrap_err().kind,
            ErrorKind::Cancelled
        );
        assert_eq!(live(), 2); // cancellation handle retains the finished request
        drop(cancel);
    }
    {
        let mut a = engine.request(NORMAL).unwrap();
        let mut b = engine.request(NORMAL).unwrap();
        thread::scope(|s| {
            let aa = s.spawn(|| a.generate());
            let bb = s.spawn(|| b.generate());
            assert!(aa.join().unwrap().is_ok());
            assert!(bb.join().unwrap().is_ok());
        });
    }
    {
        let mut bad = engine
            .request(r#"{"messages":[{"role":"user","content":"deferred"}]}"#)
            .unwrap();
        let mut later = engine.request(NORMAL).unwrap();
        let e = bad.generate().unwrap_err();
        assert_eq!(e.kind, ErrorKind::Native);
        assert!(e.partial_result.unwrap().contains("猫🐈"));
        assert_eq!(later.generate().unwrap_err().kind, ErrorKind::Native);
    }
    drop(engine);
    assert_eq!(live(), 0);
    let engine = Engine::new("{}").unwrap();
    let mut r = engine.request(NORMAL).unwrap();
    let cancel = r.cancellation();
    drop(engine); // retained request survives its Engine handle
    assert!(r.generate().is_ok());
    drop(r);
    assert_eq!(live(), 2);
    cancel.cancel(); // valid even after the Request wrapper has been dropped
    drop(cancel);
    assert_eq!(live(), 0);
}
