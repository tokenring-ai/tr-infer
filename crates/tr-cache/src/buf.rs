//! A 4 KiB-aligned host buffer (O_DIRECT needs aligned addresses; the engine's export/import
//! writes it from every tile in parallel).
use crate::object::ALIGN;
use std::alloc::{alloc_zeroed, dealloc, Layout};

pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// A zeroed buffer of `len` bytes (capacity rounded up to the alignment).
    pub fn new(len: usize) -> AlignedBuf {
        let cap = len.div_ceil(ALIGN).max(1) * ALIGN;
        let layout = Layout::from_size_align(cap, ALIGN).expect("layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory allocating {cap} bytes");
        AlignedBuf { ptr, len, cap }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
    /// Shrink or grow the visible length within the capacity (bytes beyond the old length are
    /// whatever was there; the allocation started zeroed).
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.cap, "len {len} > capacity {}", self.cap);
        self.len = len;
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl std::ops::Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, Layout::from_size_align(self.cap, ALIGN).expect("layout")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligned_and_zeroed() {
        let mut b = AlignedBuf::new(5000);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
        assert_eq!(b.len(), 5000);
        assert_eq!(b.capacity(), 8192);
        assert!(b.iter().all(|&x| x == 0));
        b[4999] = 7;
        b.set_len(8192);
        assert_eq!(b[4999], 7);
        let e = AlignedBuf::new(0);
        assert!(e.is_empty());
        assert_eq!(e.capacity(), ALIGN);
    }
}
