#![allow(unused_variables)]

#[no_mangle]
pub extern "C" fn enqueue(frame: *mut u8) {
    // No-op stub for tests
}

#[no_mangle]
pub extern "C" fn sched_read(
    fd: i64,
    buf: *mut u8,
    len: usize,
    k: extern "C" fn(*mut u8, usize),
    frame: *mut u8,
) {
    // For tests, call the continuation immediately with len=0
    unsafe {
        k(frame, 0);
    }
}
