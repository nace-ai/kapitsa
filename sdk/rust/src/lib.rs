//! Owning Rust handles for Kapitsa's in-process C ABI, version 1.
//!
//! JSON options, requests, deltas and results follow `sdk/c/kapitsa.h`. Calls block;
//! move them to application-owned worker threads when needed. No runtime or JSON
//! dependency is required. The native engine retains its cooperative GPU lock
//! until the engine, all requests and all cancellation handles have been dropped.

use std::any::Any;
use std::cell::Cell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::sync::Arc;

mod ffi {
    use super::*;
    pub type Callback = unsafe extern "C" fn(*const c_char, usize, *mut c_void) -> c_int;
    extern "C" {
        pub fn kapitsa_abi_version() -> u32;
        pub fn kapitsa_engine_create(
            json: *const c_char,
            out: *mut *mut c_void,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn kapitsa_engine_destroy(engine: *mut c_void);
        pub fn kapitsa_request_create(
            engine: *mut c_void,
            json: *const c_char,
            out: *mut *mut c_void,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn kapitsa_generate(
            request: *mut c_void,
            callback: Option<Callback>,
            user: *mut c_void,
            result: *mut *mut c_char,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn kapitsa_request_cancel(request: *mut c_void);
        pub fn kapitsa_request_destroy(request: *mut c_void);
        pub fn kapitsa_string_free(string: *mut c_char);
    }
}

/// Native status, or a failure detected before entering the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    Busy,
    Cancelled,
    Native,
    AbiMismatch,
    Contract,
    Unknown(i32),
}

/// A failed call can still contain a partial response, including on cancellation.
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub partial_result: Option<String>,
}
impl Error {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            partial_result: None,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for Error {}

struct NativeString(*mut c_char);
impl NativeString {
    fn empty() -> Self {
        Self(ptr::null_mut())
    }
    fn text(&self) -> Option<String> {
        // ABI output strings are NUL-terminated and remain owned until string_free.
        (!self.0.is_null())
            .then(|| unsafe { CStr::from_ptr(self.0).to_string_lossy().into_owned() })
    }
}
impl Drop for NativeString {
    fn drop(&mut self) {
        // Every non-null output is freed even on failure or callback panic.
        unsafe { ffi::kapitsa_string_free(self.0) }
    }
}
fn input(json: &str) -> Result<CString, Error> {
    CString::new(json)
        .map_err(|_| Error::new(ErrorKind::InvalidArgument, "JSON contains a NUL byte"))
}
fn status(code: c_int, error: &NativeString, partial_result: Option<String>) -> Result<(), Error> {
    let kind = match code {
        0 => return Ok(()),
        1 => ErrorKind::InvalidArgument,
        2 => ErrorKind::Busy,
        3 => ErrorKind::Cancelled,
        4 => ErrorKind::Native,
        x => ErrorKind::Unknown(x),
    };
    Err(Error {
        kind,
        message: error
            .text()
            .unwrap_or_else(|| format!("native status {code}")),
        partial_result,
    })
}

/// A loaded native engine. Requests retain the native engine independently.
///
/// Create at most one live engine per process. Fatal native errors outside the
/// ABI's recovery boundary can terminate the process; use process isolation if
/// your application must survive those failures.
pub struct Engine(NonNull<c_void>);
// The C ABI allows distinct requests on different threads. &Engine only creates
// retained requests; Rust prevents destruction while such a call borrows it.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}
impl Engine {
    pub fn new(options_json: &str) -> Result<Self, Error> {
        let json = input(options_json)?;
        // A version mismatch must be rejected before interpreting any handle.
        let version = unsafe { ffi::kapitsa_abi_version() };
        if version != 1 {
            return Err(Error::new(
                ErrorKind::AbiMismatch,
                format!("expected ABI 1, received {version}"),
            ));
        }
        let mut out = ptr::null_mut();
        let mut error = NativeString::empty();
        // Input lives through this synchronous call; both output slots are distinct.
        let code = unsafe { ffi::kapitsa_engine_create(json.as_ptr(), &mut out, &mut error.0) };
        status(code, &error, None)?;
        NonNull::new(out)
            .map(Self)
            .ok_or_else(|| Error::new(ErrorKind::Contract, "native success without an engine"))
    }

    pub fn request(&self, request_json: &str) -> Result<Request, Error> {
        let json = input(request_json)?;
        let mut out = ptr::null_mut();
        let mut error = NativeString::empty();
        // self remains borrowed until the native request has retained the engine.
        let code = unsafe {
            ffi::kapitsa_request_create(self.0.as_ptr(), json.as_ptr(), &mut out, &mut error.0)
        };
        status(code, &error, None)?;
        let raw = NonNull::new(out)
            .ok_or_else(|| Error::new(ErrorKind::Contract, "native success without a request"))?;
        Ok(Request {
            inner: Arc::new(RequestInner(raw)),
            used: false,
        })
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::kapitsa_engine_destroy(self.0.as_ptr()) }
    }
}

struct RequestInner(NonNull<c_void>);
// The only shared operation is cancellation, explicitly thread-safe in the C ABI.
// Generation holds an Arc and requires &mut Request. The last Arc alone destroys
// the request, so destruction cannot race generation or cancellation.
unsafe impl Send for RequestInner {}
unsafe impl Sync for RequestInner {}
impl Drop for RequestInner {
    fn drop(&mut self) {
        unsafe { ffi::kapitsa_request_destroy(self.0.as_ptr()) }
    }
}

/// A one-shot request. May be moved to a worker; it cannot be cloned.
pub struct Request {
    inner: Arc<RequestInner>,
    used: bool,
}
/// Cooperative cancellation from any thread. Clones retain the request and engine.
#[derive(Clone)]
pub struct CancelHandle(Arc<RequestInner>);
impl CancelHandle {
    pub fn cancel(&self) {
        unsafe { ffi::kapitsa_request_cancel(self.0 .0.as_ptr()) }
    }
}

thread_local! {
    static GENERATING: Cell<bool> = const { Cell::new(false) };
}
struct GenerationScope;
impl GenerationScope {
    fn enter() -> Result<Self, Error> {
        if GENERATING.with(|active| active.replace(true)) {
            return Err(Error::new(
                ErrorKind::Busy,
                "generation cannot reenter a callback",
            ));
        }
        Ok(Self)
    }
}
impl Drop for GenerationScope {
    fn drop(&mut self) {
        GENERATING.with(|active| active.set(false));
    }
}

struct CallbackState<'a> {
    callback: &'a mut dyn FnMut(&str) -> bool,
    panic: Option<Box<dyn Any + Send>>,
    invalid_utf8: bool,
}
unsafe extern "C" fn delta(bytes: *const c_char, length: usize, user: *mut c_void) -> c_int {
    // generate invokes callbacks synchronously, never after it returns. The
    // mutable context lives on that call's stack and is not accessed concurrently.
    let state = unsafe { &mut *user.cast::<CallbackState<'_>>() };
    if state.panic.is_some() || state.invalid_utf8 {
        return 0;
    }
    let bytes = if length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), length) }
    };
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            state.invalid_utf8 = true;
            return 0;
        }
    };
    // Unwinding through C is forbidden. Cancel now, then resume after native
    // generation has returned and all native output strings have been freed.
    match catch_unwind(AssertUnwindSafe(|| (state.callback)(text))) {
        Ok(keep_going) => c_int::from(keep_going),
        Err(panic) => {
            state.panic = Some(panic);
            0
        }
    }
}
impl Request {
    pub fn cancellation(&self) -> CancelHandle {
        CancelHandle(self.inner.clone())
    }
    pub fn generate(&mut self) -> Result<String, Error> {
        self.run(None)
    }
    /// Stream borrowed UTF-8 JSON deltas on the calling thread. Return false to
    /// cancel. Starting another generation from the callback returns Busy.
    /// A callback panic is resumed after native generation returns
    /// (with panic=unwind); panic=abort retains Rust's normal abort behavior.
    pub fn generate_with(
        &mut self,
        mut callback: impl FnMut(&str) -> bool,
    ) -> Result<String, Error> {
        self.run(Some(&mut callback))
    }
    fn run(&mut self, callback: Option<&mut dyn FnMut(&str) -> bool>) -> Result<String, Error> {
        if self.used {
            return Err(Error::new(ErrorKind::Busy, "request is one-shot"));
        }
        // Native callbacks can hold the engine turn. Reject nested generation
        // before consuming this request, so it can run after the callback ends.
        let _scope = GenerationScope::enter()?;
        self.used = true;
        let mut state = callback.map(|callback| CallbackState {
            callback,
            panic: None,
            invalid_utf8: false,
        });
        let (cb, user) = match state.as_mut() {
            Some(s) => (
                Some(delta as ffi::Callback),
                (s as *mut CallbackState<'_>).cast(),
            ),
            None => (None, ptr::null_mut()),
        };
        let mut result = NativeString::empty();
        let mut error = NativeString::empty();
        // &mut self excludes another generate; cancellation handles keep the
        // same native request alive and may set its flag concurrently.
        let code = unsafe {
            ffi::kapitsa_generate(self.inner.0.as_ptr(), cb, user, &mut result.0, &mut error.0)
        };
        if let Some(panic) = state.as_mut().and_then(|s| s.panic.take()) {
            drop(result);
            drop(error);
            resume_unwind(panic);
        }
        let text = result.text();
        if state.is_some_and(|s| s.invalid_utf8) {
            return Err(Error {
                kind: ErrorKind::Contract,
                message: "native callback returned invalid UTF-8".into(),
                partial_result: text,
            });
        }
        status(code, &error, text.clone())?;
        text.ok_or_else(|| Error::new(ErrorKind::Contract, "native success without a result"))
    }
}

#[cfg(all(test, not(feature = "fixture")))]
mod native_tests {
    use super::*;

    #[test]
    fn null_request_contract() {
        let json = CString::new("{}").unwrap();
        let mut request = ptr::null_mut();
        let mut error = NativeString::empty();
        let code = unsafe {
            ffi::kapitsa_request_create(ptr::null_mut(), json.as_ptr(), &mut request, &mut error.0)
        };
        assert_eq!(code, 1);
        assert!(request.is_null());
        assert!(error.text().is_some_and(|s| !s.is_empty()));

        let mut result = NativeString::empty();
        let mut error = NativeString::empty();
        let code = unsafe {
            ffi::kapitsa_generate(request, None, ptr::null_mut(), &mut result.0, &mut error.0)
        };
        assert_eq!(code, 1);
        assert!(result.0.is_null());
        assert!(error.text().is_some_and(|s| !s.is_empty()));
        unsafe {
            ffi::kapitsa_request_cancel(request);
            ffi::kapitsa_request_destroy(request);
            ffi::kapitsa_engine_destroy(ptr::null_mut());
            ffi::kapitsa_string_free(ptr::null_mut());
        }
    }
}
