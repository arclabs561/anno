#![allow(dead_code)]
use std::sync::{Arc, Mutex, MutexGuard};

struct Pool(Mutex<()>);

impl Pool {
    fn acquire(&self) -> Result<MutexGuard<'_, ()>, ()> {
        self.0.lock().map_err(|_| ())
    }
}

fn repeated_clone(values: &[String], label: &String) {
    for value in values {
        // ruleid: unnecessary-clone-in-loop
        let copied = label.clone();
        // ok: unnecessary-clone-in-loop
        let per_item = value.clone();
        println!("{copied} {per_item}");
    }
}

fn cache_snapshot(keys: &[usize], cache: &Mutex<Vec<String>>) {
    // ok: cache-clone-in-loop
    let before = cache.lock().unwrap().clone();
    for key in keys {
        // ruleid: cache-clone-in-loop
        let copy = cache.lock().unwrap().clone();
        println!("{:?} {:?}", before.get(*key), copy.get(*key));
    }
}

fn session_cleanup(pool: &Pool) -> Result<(), ()> {
    let lease = pool.acquire()?;
    // ruleid: session-acquire-without-release
    std::mem::forget(lease);
    let another = pool.acquire()?;
    // ok: session-acquire-without-release
    drop(another);
    Ok(())
}

fn redundant_owner(data: String) {
    // ruleid: arc-clone-unnecessary
    let first = Arc::new(data);
    let second = first.clone();
    drop(first);
    println!("{second}");
}

fn shared_owner(data: String) {
    // ok: arc-clone-unnecessary
    let first = Arc::new(data);
    let second = first.clone();
    println!("{first} {second}");
}

fn copy_values(items: Vec<String>) {
    // ruleid: large-vec-clone
    let copied = items.clone();
    println!("{copied:?}");
    let local: Vec<String> = copied;
    // ruleid: large-vec-clone
    let another_copy = local.clone();
    println!("{another_copy:?}");
    let active = true;
    // ok: large-vec-clone
    let other = active.clone();
    println!("{other}");
}
