//! Plain C-ABI exports (no wasm-bindgen) so the runtime can be driven from any
//! WASM host — e.g. `wasmtime` in Python. The host allocates buffers with
//! `rt_alloc`, copies the bundle bytes in, calls `rt_load` to get an opaque
//! pipeline handle, then `rt_process(handle, text)` which returns a pointer to
//! UTF-8 JSON (the full Doc) and writes its length through `out_len`. Free
//! returned/own buffers with `rt_free`.
//!
//! Build: `cargo build --target wasm32-unknown-unknown --release \
//!         --no-default-features --features capi`

use crate::pipeline::Pipeline;
use std::alloc::{alloc, dealloc, Layout};
use std::slice;

fn layout(len: usize) -> Layout {
    Layout::from_size_align(len.max(1), 1).unwrap()
}

/// Allocate `len` bytes in WASM linear memory; returns the pointer.
#[no_mangle]
pub extern "C" fn rt_alloc(len: usize) -> *mut u8 {
    unsafe { alloc(layout(len)) }
}

/// Free a buffer previously returned by `rt_alloc`/`rt_process`.
#[no_mangle]
pub extern "C" fn rt_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe { dealloc(ptr, layout(len)) }
    }
}

/// Build a pipeline from bundle bytes. `k_*` may be (null, 0) when the model
/// has no static vectors. Returns an opaque handle (leaked `Box<Pipeline>`),
/// or null on failure. Free with `rt_drop`.
///
/// # Safety
/// Pointers must reference `*_len` valid bytes in WASM memory.
#[no_mangle]
pub unsafe extern "C" fn rt_load(
    m_ptr: *const u8,
    m_len: usize,
    st_ptr: *const u8,
    st_len: usize,
    k_ptr: *const u8,
    k_len: usize,
    r_ptr: *const u8,
    r_len: usize,
) -> *mut Pipeline {
    let manifest = match std::str::from_utf8(slice::from_raw_parts(m_ptr, m_len)) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let st = slice::from_raw_parts(st_ptr, st_len);
    let k2r = if k_len > 0 {
        std::str::from_utf8(slice::from_raw_parts(k_ptr, k_len)).ok()
    } else {
        None
    };
    let r2w = if r_len > 0 {
        std::str::from_utf8(slice::from_raw_parts(r_ptr, r_len)).ok()
    } else {
        None
    };
    let pipe = Pipeline::from_bytes_full(manifest, st, k2r, r2w);
    Box::into_raw(Box::new(pipe))
}

/// Free a pipeline handle from `rt_load`.
///
/// # Safety
/// `handle` must be a live pointer from `rt_load`.
#[no_mangle]
pub unsafe extern "C" fn rt_drop(handle: *mut Pipeline) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Process `text` and return a pointer to UTF-8 JSON (the full Doc), writing
/// its byte length through `out_len`. Free the returned buffer with `rt_free`.
///
/// # Safety
/// `handle` must be from `rt_load`; `text_ptr`/`text_len` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn rt_process(
    handle: *mut Pipeline,
    text_ptr: *const u8,
    text_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let pipe = &*handle;
    let text = match std::str::from_utf8(slice::from_raw_parts(text_ptr, text_len)) {
        Ok(s) => s,
        Err(_) => {
            *out_len = 0;
            return std::ptr::null_mut();
        }
    };
    let doc = pipe.process(text);
    let json = serde_json::to_string(&doc).unwrap_or_else(|_| "{}".into());
    let bytes = json.into_bytes();
    let len = bytes.len();
    let out = rt_alloc(len);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, len);
    *out_len = len;
    out
}

/// Process `text` and return spaCy `Doc.to_json()`-shaped JSON (see `rt_process`).
///
/// # Safety
/// Same as `rt_process`.
#[no_mangle]
pub unsafe extern "C" fn rt_to_json(
    handle: *mut Pipeline,
    text_ptr: *const u8,
    text_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let pipe = &*handle;
    let text = match std::str::from_utf8(slice::from_raw_parts(text_ptr, text_len)) {
        Ok(s) => s,
        Err(_) => {
            *out_len = 0;
            return std::ptr::null_mut();
        }
    };
    let json = pipe.process(text).to_spacy_json().to_string();
    let bytes = json.into_bytes();
    let len = bytes.len();
    let out = rt_alloc(len);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, len);
    *out_len = len;
    out
}
