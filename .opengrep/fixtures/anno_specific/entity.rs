#![allow(dead_code)]

struct Entity {
    start: usize,
    end: usize,
    label: &'static str,
}

fn unchecked(begin: usize, finish: usize) -> Entity {
    // ruleid: entity-offset-validation-missing
    Entity { label: "entry", start: begin, end: finish }
}

fn guarded(begin: usize, finish: usize) -> Result<Entity, ()> {
    if begin > finish {
        return Err(());
    }
    // ok: entity-offset-validation-missing
    Ok(Entity { start: begin, end: finish, label: "entry" })
}

fn ordered(begin: usize, finish: usize) -> Option<Entity> {
    if begin <= finish {
        // ok: entity-offset-validation-missing
        Some(Entity { start: begin, end: finish, label: "entry" })
    } else {
        None
    }
}

fn independent_guard(start: usize, end: usize, other: usize) -> Result<Entity, ()> {
    if other > end {
        return Err(());
    }
    // ruleid: entity-offset-validation-missing
    Ok(Entity { start, end, label: "entry" })
}

fn normalized(left: usize, right: usize) -> Entity {
    let (start, end) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    // ok: entity-offset-validation-missing
    Entity { start, end, label: "entry" }
}
